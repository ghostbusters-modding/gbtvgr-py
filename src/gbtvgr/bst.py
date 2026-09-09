#!/usr/bin/env python3
"""
.bst sets: the wire codec (parse, verify, rebuild).

The set is the world: render geometry, vis cells and portals, collision BSP and
per-section BVTs, skybox layers, lights, breakers, navmesh.

Commands:
  info <file.bst>                  summary
  verify <file.bst ...>            parse->rebuild, must be byte-identical
  roundtrip-test <dir>             verify every .bst under dir
"""
import argparse, glob, os, struct, sys

from .smb import R, W, read_deps, write_deps, read_bvt_node, write_bvt_node, \
    TYPE_SIZE, decl_stride, read_material, write_material, want

# bst, texture, material, mesh, renderpacket, breaker, breakerprops, event
VERS = (0x7F, 7, 0x19E, 2, 9, 0xB, 3, 1)
# 9th version field: light properties, reader accepts 0x28 or 0x29


def _read(path):
    with open(path, 'rb') as fh:
        return fh.read()


def read_str(r):
    """NUL-terminated, then padded to a multiple of 4 -- counted from the string's
    own start, not from the file offset."""
    s = r.cstr()
    pad = r.take((-(len(s) + 1)) % 4)
    return s, pad

def write_str(w, sp):
    s, pad = sp
    w.cstr(s); w.w(pad)


# --- render packet header/data: the .smb part header minus name, matidx
#     and meshver ------------------------------------------------------------
def read_rendpkt_hdr(r):
    p = {}
    want(r.u32(), 9, 'renderpacket ver')
    p['datasize'] = r.u32()
    p['decl'] = [(r.u32(), r.u32(), r.u32()) for _ in range(22)]
    p['nverts'], p['nprims'], p['f18'] = r.u32(), r.u32(), r.u32()
    p['morphs'] = [{'idx': r.u16(), 'chan': r.u32(), 'count': r.u32()}
                   for _ in range(r.u32())]
    return p

def write_rendpkt_hdr(w, p):
    w.u32(9); w.u32(p['datasize'])
    for a, b, c in p['decl']: w.u32(a); w.u32(b); w.u32(c)
    w.u32(p['nverts']); w.u32(p['nprims']); w.u32(p['f18'])
    w.u32(len(p['morphs']))
    for m in p['morphs']:
        w.u16(m['idx']); w.u32(m['chan']); w.u32(m['count'])

def read_rendpkt_data(r, p):
    p['vdata'] = r.take(p['nverts'] * decl_stride(p['decl']))
    p['idata'] = r.take(p['nprims'] * 6)
    p['mdata'] = [r.take(m['count'] * 0x34) for m in p['morphs']]

def write_rendpkt_data(w, p):
    w.w(p['vdata']); w.w(p['idata'])
    for b in p['mdata']: w.w(b)


# --- CMesh: ver + bbox + render packet header --------------------------------
def read_mesh_hdr(r):
    want(r.u32(), 2, 'mesh ver')
    bbox = r.take(24)
    p = read_rendpkt_hdr(r)
    p['bbox'] = bbox
    return p

def write_mesh_hdr(w, p):
    w.u32(2); w.w(p['bbox']); write_rendpkt_hdr(w, p)


# --- skybox layers (SkyBox and its layer types) ------------------------------
# Every layer body is one fixed-size raw block: a 0x148 prologue (name, material
# string buffer, 2 u32) plus per-type extras, all fixed-size reads.
SKY_SIZES = {1: 0x148 + 0xc + 0x80 + 0x18,        # Dome
             2: 0x148 + 0x18 + 0x1c + 8,          # Billboard
             3: 0x148 + 0x18 + 0x1c + 4 + 8,      # Billboard variant
             4: 0x148 + 4,                        # StarField
             5: 0x148 + 0x80 + 0x18 + 0x1c}       # Model

def read_sky_layer(r, typ):
    return {'type': typ, 'raw': r.take(SKY_SIZES[typ])}

def write_sky_layer(w, layer):
    w.w(layer['raw'])


def read_skybox(r):
    sk = {'layers': []}
    n = r.u32()
    for _ in range(n):
        typ = r.u32()
        sk['layers'].append(read_sky_layer(r, typ) if typ else {'type': 0})
    sk['fogvec'] = r.take(12)
    sk['fog3'] = r.take(12)
    sk['order'] = r.take(50 * 4)
    return sk

def write_skybox(w, sk):
    w.u32(len(sk['layers']))
    for l in sk['layers']:
        w.u32(l['type'])
        if l['type']: write_sky_layer(w, l)
    w.w(sk['fogvec']); w.w(sk['fog3']); w.w(sk['order'])


# --- CPortalLight ------------------------------------------------------------
def read_light(r):
    l = {'a': r.take(0x20), 'pos': r.take(12), 'b': r.take(12),
         'scal': r.take(12 * 4), 'vec': r.take(12), 'f48': r.take(4)}
    n = r.u32()
    l['idx'] = r.take(n * 4)
    l['name1'] = r.take(0x40); l['name2'] = r.take(0x40); l['name3'] = r.take(0x40)
    return l

def write_light(w, l):
    w.w(l['a']); w.w(l['pos']); w.w(l['b']); w.w(l['scal']); w.w(l['vec'])
    w.w(l['f48']); w.u32(len(l['idx']) // 4); w.w(l['idx'])
    w.w(l['name1']); w.w(l['name2']); w.w(l['name3'])


# --- CBreaker (ver 0xB; props ver 3; events ver 1) ---------------------------
def read_event(r):
    want(r.u32(), 1, 'event ver')
    return read_str(r)

def write_event(w, e):
    w.u32(1); write_str(w, e)

def read_breaker(r):
    b = {}
    want(r.u32(), 0xB, 'breaker ver')
    b['name'] = r.take(0x20)
    want(r.u32(), 3, 'breaker props ver')
    b['pf'] = r.take(12)
    b['ev1'] = read_event(r); b['ev2'] = read_event(r)
    b['s1'] = read_str(r); b['s2'] = read_str(r); b['s3'] = read_str(r)
    b['p84'] = r.take(4)
    b['fe8'] = r.take(12); b['fcc'] = r.take(4); b['fd0'] = r.take(0x18)
    nv, nt = r.u32(), r.u32()
    b['verts'] = [r.take(0x2c) for _ in range(nv)]
    b['tris'] = r.take(nt * 6)
    return b

def write_breaker(w, b):
    w.u32(0xB); w.w(b['name']); w.u32(3); w.w(b['pf'])
    write_event(w, b['ev1']); write_event(w, b['ev2'])
    write_str(w, b['s1']); write_str(w, b['s2']); write_str(w, b['s3'])
    w.w(b['p84']); w.w(b['fe8']); w.w(b['fcc']); w.w(b['fd0'])
    w.u32(len(b['verts'])); w.u32(len(b['tris']) // 6)
    for v in b['verts']: w.w(v)
    w.w(b['tris'])


# --- section = portal cell (CPortalSet) --------------------------------------
def read_section(r):
    s = {'name': r.take(0x40)}
    s['b1'], s['b2'], s['b3'] = r.u32(), r.u32(), r.u32()
    s['f3b4'], s['f3b8'], s['skip'], s['f3c0'] = r.u32(), r.u32(), r.u32(), r.u32()
    s['names3'] = r.take(300)
    n = r.u32()
    s['extnames'] = [r.take(100) for _ in range(n)]
    n = r.u32()
    s['portidx'] = r.take(n * 4)
    nmesh, nlref = r.u32(), r.u32()
    s['meshes'] = []
    for _ in range(nmesh):
        m = {'h70': r.take(2), 'name': r.take(0x20),
             'flags': r.u32(), 'f6c': r.take(4)}
        if m['flags'] & 0x20000:
            m['f90'] = r.take(4)
        else:
            m['pkt'] = read_mesh_hdr(r)
        s['meshes'].append(m)
    s['lrefs'] = [{'name': r.take(0x20), 'v': r.take(0x20)} for _ in range(nlref)]
    # per-section collision BVT (ver 3)
    want(r.u32(), 3, 'bvt ver')
    s['bvtflag'] = r.u32()   # arena byte size for the tree (0 = no tree)
    s['bvt'] = read_bvt_node(r) if s['bvtflag'] else None
    n = r.u32(); s['arr4f4'] = r.take(n * 4)
    n = r.u32(); s['arr818'] = r.take(n * 4)
    n = r.u32(); s['arr470'] = r.take(n * 4)
    s['dataofs'], s['fa20'] = r.u32(), r.u32()
    s['bbox'] = r.take(24)
    return s

def write_section(w, s):
    w.w(s['name'])
    for k in ('b1', 'b2', 'b3', 'f3b4', 'f3b8', 'skip', 'f3c0'): w.u32(s[k])
    w.w(s['names3'])
    w.u32(len(s['extnames']))
    for e in s['extnames']: w.w(e)
    w.u32(len(s['portidx']) // 4); w.w(s['portidx'])
    w.u32(len(s['meshes'])); w.u32(len(s['lrefs']))
    for m in s['meshes']:
        w.w(m['h70']); w.w(m['name']); w.u32(m['flags']); w.w(m['f6c'])
        if m['flags'] & 0x20000:
            w.w(m['f90'])
        else:
            write_mesh_hdr(w, m['pkt'])
    for l in s['lrefs']: w.w(l['name']); w.w(l['v'])
    w.u32(3); w.u32(s['bvtflag'])
    if s['bvt'] is not None:
        write_bvt_node(w, s['bvt'])
    for k in ('arr4f4', 'arr818', 'arr470'):
        w.u32(len(s[k]) // 4); w.w(s[k])
    w.u32(s['dataofs']); w.u32(s['fa20'])
    w.w(s['bbox'])


# --- navmesh (CNavMesh, ver 0x3E) --------------------------------------------
def read_nav(r):
    nv, nn, f, ne, np_ = r.u32(), r.u32(), r.u32(), r.u32(), r.u32()
    nav = {'f': f, 'empty': nv == 0 and nn == 0,
           'nverts': nv, 'nnodes': nn, 'nextra': ne, 'nparts': np_}
    if nav['empty']:
        return nav
    nav['verts'] = r.take(nv * 12)
    nav['nodes'] = []
    for _ in range(nn):
        d = {'hdr': r.take(12 * 4)}
        ec = r.u32()
        d['v'] = r.take(ec * 4); d['nb'] = r.take(ec * 4); d['d0'] = r.take(ec * 4)
        d['f15c'] = r.take(4)
        nav['nodes'].append(d)
    nav['extra'] = []
    for _ in range(ne):
        e = {'a': r.take(4), 'b': r.take(4), 'c': r.take(4)}
        c2 = r.u32()
        e['rows'] = r.take(c2 * 16)
        nav['extra'].append(e)
    nav['parts'] = [r.take(28) for _ in range(np_)]
    return nav

def write_nav(w, nav):
    if nav['empty']:
        w.u32(0); w.u32(0); w.u32(nav['f'])
        w.u32(nav['nextra']); w.u32(nav['nparts'])
        return
    w.u32(nav['nverts']); w.u32(nav['nnodes']); w.u32(nav['f'])
    w.u32(nav['nextra']); w.u32(nav['nparts'])
    w.w(nav['verts'])
    for d in nav['nodes']:
        w.w(d['hdr']); w.u32(len(d['v']) // 4)
        w.w(d['v']); w.w(d['nb']); w.w(d['d0']); w.w(d['f15c'])
    for e in nav['extra']:
        w.w(e['a']); w.w(e['b']); w.w(e['c'])
        w.u32(len(e['rows']) // 16); w.w(e['rows'])
    for p in nav['parts']: w.w(p)


# --- whole set ---------------------------------------------------------------
def parse(d, upto=None):
    r = R(d)
    v = struct.unpack_from('<9I', d, 0)
    if v[:8] != VERS or v[8] not in (0x28, 0x29):
        raise ValueError('bad version block %s' % (v,))
    r.o = 36
    m = {'lightver': v[8], 'hash': r.take(16)}
    m['deps'], m['deppad'] = read_deps(r)
    m['datasize'] = r.u32()
    m['boolA'] = r.u32()
    m['fogA'] = r.take(12); m['fogB'] = r.take(12)
    m['boolB'] = r.u32()
    m['f2f08'] = r.u32()
    m['sky'] = read_skybox(r)
    m['f62d0'] = r.u32()
    m['hdr2a'] = r.take(11 * 4)
    m['vec6310'] = r.take(12)
    m['hdr2b'] = r.take(6 * 4)
    m['f6358'] = r.u32()
    m['s6360'] = read_str(r)
    m['hdr2c'] = r.take(7 * 4)
    n = r.u32()
    m['materials'] = []
    for _ in range(n):
        s = read_str(r)
        # empty ref = an embedded material follows, same wire format as in .smb
        m['materials'].append({'ref': s,
                               'embedded': read_material(r) if not s[0] else None})
    m['mat16'] = r.take(n * 2)
    n = r.u32()
    m['sections'] = [read_section(r) for _ in range(n)]
    n = r.u32()
    m['lights'] = [read_light(r) for _ in range(n)]
    n = r.u32()
    m['portals'] = [r.take(0xEC) for _ in range(n)]  # 3 vec3 + 100B (8B + name) + 100B
    n = r.u32()
    m['fx'] = r.take(n * 0x118)
    n = r.u32()
    m['breakers'] = [read_breaker(r) for _ in range(n)]
    n1, n2 = r.u32(), r.u32()
    m['wblobs'] = [r.take(52) for _ in range(n2)]
    m['wmesh'] = []
    for _ in range(n1):
        p = read_mesh_hdr(r)
        read_rendpkt_data(r, p)
        m['wmesh'].append(p)
    m['watervis'] = r.take(len(m['sections']))
    n = r.u32()
    m['bspglob'] = r.take(4)
    m['bsp'] = r.take(n * 0x30)
    want(r.u32(), 0x3E, 'nav ver')
    m['nav'] = read_nav(r)
    # Each section's blob runs from its dataofs to the next section's. Vertex data
    # is at the front; the engine streams the rest lazily, so keep blobs raw.
    for s in m['sections']:
        if r.o > s['dataofs']:
            raise ValueError('section data @%#x already past %#x' % (r.o, s['dataofs']))
        s['datagap'] = r.take(s['dataofs'] - r.o)
        s['blob'] = r.take(s['fa20'])
        rb = R(s['blob'])
        for me in s['meshes']:
            if not (me['flags'] & 0x20000):
                read_rendpkt_data(rb, me['pkt'])
    if r.o != len(d):
        raise ValueError('trailing bytes: %#x != %#x' % (r.o, len(d)))
    return m


def build(m):
    w = W()
    for x in VERS: w.u32(x)
    w.u32(m['lightver'])
    w.w(m['hash'])
    write_deps(w, m['deps'])
    w.u32(m['datasize']); w.u32(m['boolA'])
    w.w(m['fogA']); w.w(m['fogB'])
    w.u32(m['boolB']); w.u32(m['f2f08'])
    write_skybox(w, m['sky'])
    w.u32(m['f62d0']); w.w(m['hdr2a']); w.w(m['vec6310']); w.w(m['hdr2b'])
    w.u32(m['f6358']); write_str(w, m['s6360']); w.w(m['hdr2c'])
    w.u32(len(m['materials']))
    for e in m['materials']:
        write_str(w, e['ref'])
        if e['embedded'] is not None:
            write_material(w, e['embedded'])
    w.w(m['mat16'])
    w.u32(len(m['sections']))
    for s in m['sections']: write_section(w, s)
    w.u32(len(m['lights']))
    for l in m['lights']: write_light(w, l)
    w.u32(len(m['portals']))
    for p in m['portals']: w.w(p)
    w.u32(len(m['fx']) // 0x118); w.w(m['fx'])
    w.u32(len(m['breakers']))
    for b in m['breakers']: write_breaker(w, b)
    w.u32(len(m['wmesh'])); w.u32(len(m['wblobs']))
    for b in m['wblobs']: w.w(b)
    for p in m['wmesh']:
        write_mesh_hdr(w, p); write_rendpkt_data(w, p)
    w.w(m['watervis'])
    w.u32(len(m['bsp']) // 0x30); w.w(m['bspglob']); w.w(m['bsp'])
    w.u32(0x3E); write_nav(w, m['nav'])
    for s in m['sections']:
        w.w(s['datagap']); w.w(s['blob'])
    return w.data()


def cmd_info(args):
    d = _read(args.file)
    m = parse(d)
    print('%s: %d sections, %d materials, %d lights, %d portals, %d breakers, '
          '%d sky layers, bsp %d nodes, nav %s' %
          (os.path.basename(args.file), len(m['sections']), len(m['materials']),
           len(m['lights']), len(m['portals']), len(m['breakers']),
           len(m['sky']['layers']), len(m['bsp']) // 0x30,
           'empty' if m['nav']['empty'] else '%dv/%dn' %
           (m['nav']['nverts'], m['nav']['nnodes'])))


def cmd_verify(args):
    ok = True
    for f in args.files:
        d = _read(f)
        try:
            out = build(parse(d))
            if out == d:
                print('OK   %s' % f)
            else:
                n = next(i for i in range(min(len(d), len(out)))
                         if d[i] != out[i]) if out != d else 0
                print('DIFF %s @%#x (len %d vs %d)' % (f, n, len(d), len(out)))
                ok = False
        except Exception as e:
            print('FAIL %s: %s' % (f, e))
            ok = False
    sys.exit(0 if ok else 1)


def cmd_roundtrip(args):
    files = sorted(glob.glob(os.path.join(args.dir, '**', '*.bst'),
                             recursive=True))
    args.files = files
    cmd_verify(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('info'); p.add_argument('file'); p.set_defaults(fn=cmd_info)
    p = sub.add_parser('verify'); p.add_argument('files', nargs='+')
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser('roundtrip-test'); p.add_argument('dir')
    p.set_defaults(fn=cmd_roundtrip)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()

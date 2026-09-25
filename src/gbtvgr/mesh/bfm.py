#!/usr/bin/env python3
"""
Skinned character meshes (.bfm) and the bone list of their skeletons (.skb).

Layout: gbtvgr-docs formats/RE_CHARACTER_MESH_FORMAT.md.

Commands:
  info <file.bfm> [--skb file.skb]   summary; with --skb, palettes print bone names
  verify <file.bfm ...>              parse->rebuild, must be byte-identical
  roundtrip-test <dir>               verify every .bfm under dir
  vertex-test <dir>                  decode->encode every packet's vertices, must be byte-identical
  skb-info <file.skb>                bone names, parents and mirrors
"""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import argparse, glob, os, struct, sys

from ..wire import R, W, f32_to_half, read_deps, read_str, want
from .mtb import read_material, write_material

VERS = (20, 0x19E, 9, 6)   # bfm, material, renderpacket, bonepacket
MAX_BONES, MAX_PARTS, MAX_PALETTE = 200, 200, 64
NO_SLOT = 0xFFFFFFFF

# type code -> (struct format, component count)
TYPE_FMT = {1: ('f', 1), 2: ('f', 2), 3: ('f', 3), 4: ('f', 4), 5: ('e', 2),
            6: ('e', 4), 7: ('B', 4), 8: ('H', 4)}
SLOT_NAME = {0: 'pos', 1: 'normal', 2: 'uv0', 3: 'uv1', 5: 'tangent', 8: 'binormal',
             15: 'colour', 16: 'bones', 17: 'weights', 18: 'uv1f'}


def _decl(pairs):
    d = [(NO_SLOT, 0, 0)] * 22
    for slot, off, t in pairs:
        d[slot] = (off, t, t)
    return tuple(d)

# The only three the corpus uses. The engine asserts on any declaration not in its table.
DECL_A = _decl([(0, 0, 3), (1, 12, 3), (2, 24, 5), (5, 32, 3), (8, 44, 3), (16, 56, 8), (17, 64, 6)])
DECL_B = _decl([(0, 0, 3), (1, 12, 3), (2, 24, 5), (3, 28, 5), (5, 36, 3), (8, 48, 3),
                (16, 60, 8), (17, 68, 6)])
DECL_C = _decl([(15, 0, 7), (0, 4, 3), (17, 16, 4), (1, 32, 3), (16, 44, 8), (2, 52, 2),
                (5, 60, 3), (8, 72, 3), (18, 84, 2)])
STRIDE = {DECL_A: 72, DECL_B: 76, DECL_C: 92}


def _read(path):
    with open(path, 'rb') as fh:
        return fh.read()

def nm(b):
    return b.split(b'\0')[0].decode('latin1')


def _slots(decl):
    # mayorMP's uv slots carry type 0 with the real type in the third field
    return [(s, off, t or t2) for s, (off, t, t2) in enumerate(decl) if off != NO_SLOT]


# --- packet header (CBonePacket wrapping CRenderPacket) ----------------------
def read_packet(r):
    want(r.u32(), 6, 'bone packet ver')
    p = {'mat': r.i32(), 'list': r.i32()}
    p['parts'] = [r.u16() for _ in range(r.i32())]
    p['extra'] = [r.u16() for _ in range(r.i32())]
    want(r.u32(), 9, 'render packet ver')
    p['datasize'] = r.u32()
    p['decl'] = tuple((r.u32(), r.u32(), r.u32()) for _ in range(22))
    p['nverts'], p['ntris'], p['palette'] = r.i32(), r.i32(), r.i32()
    p['morphs'] = [{'idx': r.u16(), 'chan': r.u32(), 'count': r.u32()} for _ in range(r.i32())]
    return p

def write_packet(w, p):
    w.u32(6); w.w(struct.pack('<ii', p['mat'], p['list']))
    for lst in (p['parts'], p['extra']):
        w.w(struct.pack('<i', len(lst)))
        for x in lst: w.u16(x)
    w.u32(9); w.u32(p['datasize'])
    for a, b, c in p['decl']: w.u32(a); w.u32(b); w.u32(c)
    w.w(struct.pack('<iiii', p['nverts'], p['ntris'], p['palette'], len(p['morphs'])))
    for m in p['morphs']:
        w.u16(m['idx']); w.u32(m['chan']); w.u32(m['count'])

def stride(p):
    # From the decl, not dataSize: mayorMP's edited packets kept the shipped dataSize
    return max(off + struct.calcsize('<%d%s' % TYPE_FMT[t][::-1]) for _, off, t in _slots(p['decl']))


# --- whole model -------------------------------------------------------------
def parse(d):
    r = R(d)
    if struct.unpack_from('<4I', d, 0) != VERS:
        raise ValueError('bad version block %s' % (struct.unpack_from('<4I', d, 0),))
    r.o = 16
    m = {'hash': r.take(16)}
    m['deps'], m['deppad'] = read_deps(r)
    nlod, npart, nbone, nmat, ntag, nlist, nmorph = (r.u32() for _ in range(7))
    m['lods'] = [r.take(8) for _ in range(nlod)]
    m['skeleton'] = r.take(128)
    m['parts'] = [{'name': r.take(30), 'bone': r.i32(), 'bbox': r.take(24)} for _ in range(npart)]
    m['tags'] = [{'name': r.take(24), 'bone': r.i32(), 'xform': r.take(64)} for _ in range(ntag)]
    m['materials'] = []
    for _ in range(nmat):
        name, pad = read_str(r)
        e = {'ref': name, 'refpad': pad, 'embedded': None}
        if not name:
            e['embedded'] = read_material(r)
        m['materials'].append(e)
    m['bone_a'], m['bone_b'] = r.take(12 * nbone), r.take(24 * nbone)
    m['bone_c'], m['bone_d'] = r.take(4 * nbone), r.take(4 * nbone)
    m['lists'] = [[r.i32() for _ in range(r.i32())] for _ in range(nlist)]
    m['morphnames'] = [r.take(64) for _ in range(nmorph)]
    m['packets'] = [[read_packet(r) for _ in range(r.u32())] for _ in range(nlod)]
    if any(r.align(16)):
        raise ValueError('non-zero header padding @%#x' % r.o)
    for p in (p for lod in m['packets'] for p in lod):
        p['vdata'] = r.take(p['nverts'] * stride(p))
        p['idata'] = r.take(p['ntris'] * 6)
        p['mdata'] = [r.take(mo['count'] * 0x34) for mo in p['morphs']]
    if r.o != len(d):
        raise ValueError('trailing bytes: %#x != %#x' % (r.o, len(d)))
    return m

def build(m):
    w = W()
    for v in VERS: w.u32(v)
    w.w(m['hash'])
    for name, h in m['deps']:
        w.cstr(name); w.w(h)
    w.w(b'\0'); w.w(m['deppad'])
    nbone = len(m['bone_c']) // 4
    for n in (len(m['lods']), len(m['parts']), nbone, len(m['materials']), len(m['tags']),
              len(m['lists']), len(m['morphnames'])):
        w.u32(n)
    for x in m['lods']: w.w(x)
    w.w(m['skeleton'])
    for p in m['parts']: w.w(p['name']); w.w(struct.pack('<i', p['bone'])); w.w(p['bbox'])
    for t in m['tags']: w.w(t['name']); w.w(struct.pack('<i', t['bone'])); w.w(t['xform'])
    for e in m['materials']:
        w.cstr(e['ref']); w.w(e['refpad'])
        if e['embedded'] is not None:
            write_material(w, e['embedded'])
    for k in ('bone_a', 'bone_b', 'bone_c', 'bone_d'): w.w(m[k])
    for lst in m['lists']: w.w(struct.pack('<%di' % (len(lst) + 1), len(lst), *lst))
    for x in m['morphnames']: w.w(x)
    for lod in m['packets']:
        w.u32(len(lod))
        for p in lod: write_packet(w, p)
    w.align(16)   # zeros in every shipped file, and it must move when the header grows
    for p in (p for lod in m['packets'] for p in lod):
        w.w(p['vdata']); w.w(p['idata'])
        for md in p['mdata']: w.w(md)
    return w.data()


# --- vertices ----------------------------------------------------------------

def decode_vertices(p):
    """{'pos': [(x,y,z)...], 'bones': [(i,i,i,i)...], ...}; halves come back as floats."""
    st, out = stride(p), {}
    for s, off, t in _slots(p['decl']):
        f, n = TYPE_FMT[t]
        fmt = struct.Struct('<%d%s' % (n, f))
        out[SLOT_NAME.get(s, 'slot%d' % s)] = [fmt.unpack_from(p['vdata'], i * st + off)
                                               for i in range(p['nverts'])]
    return out

def encode_vertices(decl, verts, st=None):
    """Inverse of decode_vertices. Unused bytes (format A's hole) are zero."""
    st = st or STRIDE[decl]
    n = len(verts['pos'])
    buf = bytearray(n * st)
    for s, off, t in _slots(decl):
        f, k = TYPE_FMT[t]
        col = verts[SLOT_NAME.get(s, 'slot%d' % s)]
        if f == 'e':
            # struct's 'e' rounds; the shipped halves are truncated
            fmt = struct.Struct('<%dH' % k)
            for i, v in enumerate(col): fmt.pack_into(buf, i * st + off, *(f32_to_half(x) for x in v))
        else:
            fmt = struct.Struct('<%d%s' % (k, f))
            for i, v in enumerate(col): fmt.pack_into(buf, i * st + off, *v)
    return bytes(buf)

def set_geometry(m, p, verts, tris, decl=DECL_A):
    """Replace a packet's vertices and triangles; dataSize and paletteSize follow."""
    if len(verts['pos']) > 0xFFFF:
        raise ValueError('%d vertices: a packet indexes with u16' % len(verts['pos']))
    if len(m['lists'][p['list']]) > MAX_PALETTE:
        raise ValueError('palette %d has more than %d bones' % (p['list'], MAX_PALETTE))
    p['decl'] = decl
    p['vdata'] = encode_vertices(decl, verts)
    p['idata'] = b''.join(struct.pack('<3H', *t) for t in tris)
    p['nverts'], p['ntris'] = len(verts['pos']), len(tris)
    p['datasize'] = len(p['vdata']) + len(p['idata'])
    p['palette'] = len(m['lists'][p['list']])

def decode_tris(p):
    return list(struct.iter_unpack('<3H', p['idata']))


# --- .skb bone list (read only) -----------------------------------------------
def read_skb_bones(d):
    r = R(d)
    want(r.u32(), 30, 'skb ver')
    r.take(16); read_deps(r)
    bones = []
    for _ in range(r.i32()):
        b = r.take(0x4C)
        name = nm(b[:24])
        h, parent, mirror = struct.unpack_from('<Iii', b, 24)
        bones.append({'name': name, 'hash': h, 'parent': parent, 'mirror': mirror, 'raw': b})
    return bones


# --- commands ----------------------------------------------------------------
def cmd_info(a):
    d = _read(a.infile)
    m = parse(d)
    names = [b['name'] for b in read_skb_bones(_read(a.skb))] if a.skb else None
    print('%s: %d bytes, skeleton %s, %d bone(s), %d part(s), %d tag(s), %d material(s), %d palette(s)'
          % (a.infile, len(d), nm(m['skeleton']), len(m['bone_c']) // 4, len(m['parts']),
             len(m['tags']), len(m['materials']), len(m['lists'])))
    for e in m['materials']:
        print('  material %s' % ('<embedded>' if e['embedded'] is not None else e['ref'].decode('latin1')))
    for i, p in enumerate(m['parts']):
        print('  part %2d %-30s bone=%d' % (i, nm(p['name']), p['bone']))
    for t in m['tags']:
        print('  tag  %-24s bone=%d' % (nm(t['name']), t['bone']))
    for i, lst in enumerate(m['lists']):
        shown = ' '.join(names[j] if names else str(j) for j in lst)
        print('  palette %d (%d): %s' % (i, len(lst), shown))
    for p in m['packets'][0] if m['packets'] else ():
        print('  packet part=%s mat=%d palette=%d verts=%-5d tris=%-5d stride=%d'
              % (p['parts'], p['mat'], p['list'], p['nverts'], p['ntris'], stride(p)))

def _check(fn):
    d = _read(fn)
    out = build(parse(d))
    if out == d: return None
    i = next((i for i, (x, y) in enumerate(zip(out, d)) if x != y), min(len(out), len(d)))
    return 'len %d->%d first diff @%#x' % (len(d), len(out), i)

def _run(files, check, what):
    fails = []
    for fn in files:
        try:
            e = check(fn)
        except Exception as ex:
            e = 'PARSE FAIL %s' % ex
        if e: fails.append((fn, e))
    print('%d/%d %s' % (len(files) - len(fails), len(files), what))
    for fn, e in fails[:15]:
        print('  FAIL %s: %s' % (fn, e))
    return 1 if fails else 0

def _corpus(root):
    return sorted(glob.glob(os.path.join(root, '**', '*.bfm'), recursive=True))

def cmd_verify(a):
    return _run(a.infiles, _check, 'byte-identical')

def cmd_roundtrip_test(a):
    return _run(_corpus(a.dir), _check, 'byte-identical round-trip')

def _vertex_check(fn):
    for lod in parse(_read(fn))['packets']:
        for i, p in enumerate(lod):
            got = encode_vertices(p['decl'], decode_vertices(p), stride(p))
            if got != p['vdata']:
                j = next(j for j, (x, y) in enumerate(zip(got, p['vdata'])) if x != y)
                return 'packet %d vertex %d byte %d' % (i, j // stride(p), j % stride(p))
    return None

def cmd_vertex_test(a):
    return _run(_corpus(a.dir), _vertex_check, 'vertex decode->encode byte-identical')

def cmd_skb_info(a):
    bones = read_skb_bones(_read(a.infile))
    print('%s: %d bone(s)' % (a.infile, len(bones)))
    for i, b in enumerate(bones):
        parent = bones[b['parent']]['name'] if b['parent'] >= 0 else '-'
        mirror = bones[b['mirror']]['name'] if b['mirror'] >= 0 else '-'
        print('  %3d %-24s parent=%-24s mirror=%s' % (i, b['name'], parent, mirror))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('info'); p.add_argument('infile'); p.add_argument('--skb')
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser('verify'); p.add_argument('infiles', nargs='+'); p.set_defaults(fn=cmd_verify)

    p = sub.add_parser('roundtrip-test'); p.add_argument('dir'); p.set_defaults(fn=cmd_roundtrip_test)

    p = sub.add_parser('vertex-test'); p.add_argument('dir'); p.set_defaults(fn=cmd_vertex_test)

    p = sub.add_parser('skb-info'); p.add_argument('infile'); p.set_defaults(fn=cmd_skb_info)

    a = ap.parse_args()
    rc = a.fn(a)
    sys.exit(rc or 0)


if __name__ == '__main__':
    main()

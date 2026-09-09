#!/usr/bin/env python3
"""
.bst geometry: meshes, collision, relayout, OBJ export.

The wire codec hands back opaque blobs -- vertex data as bytes, collision as a
packed BVT, offsets the caller has to keep consistent. This turns those into
editable geometry and back, in pure python so the Blender add-on stays a thin
wrapper. decode -> encode is byte-exact and relayout() of an unmodified parse
reproduces the whole file, so an open-and-save pass is a no-op.

The slot index carries the vertex semantic: 0 position, 1 normal, 2-4 texture
uvs, 5 tangent, 8 binormal, 11 colour, 12 lightmap uv, 19-21 rnm basis. Two
lighting families, never mixed: {11, 12}, or {11, 19, 20, 21} for a per-vertex
3-basis bake. Unknown slots and colour channel order round-trip as raw bytes.

Commands:
  info <file.bst>                    per-section geometry summary
  check <file.bst ...>               decode->encode + relayout must be exact
  roundtrip-test <dir>               check every .bst under dir
  to-obj <file.bst> <out.obj>        dump geometry (--section, --collision)
"""
import argparse
import glob
import hashlib
import math
import os
import struct
import sys

from . import bst
from .smb import build_bvt, decl_stride, f32_to_half, half_to_f32, TYPE_SIZE

NOSLOT = 0xFFFFFFFF

S_POS, S_NORMAL, S_TANGENT, S_BINORMAL = 0, 1, 5, 8
UV_SLOTS = (2, 3, 4, 12)
COLOR_SLOTS = (11, 19, 20, 21)
SLOT_NAME = {0: 'pos', 1: 'normal', 2: 'uv0', 3: 'uv1', 4: 'uv2', 5: 'tangent',
             8: 'binormal', 11: 'col0', 12: 'lmuv', 19: 'rnm0', 20: 'rnm1',
             21: 'rnm2'}


def _read(path):
    with open(path, 'rb') as fh:
        return fh.read()


def slot_name(slot):
    return SLOT_NAME.get(slot, 'slot%d' % slot)


def make_decl(entries):
    """entries: [(slot, offset, type)] -> a 22-slot declaration tuple."""
    d = [(NOSLOT, 0, 0)] * 22
    for slot, off, t in entries:
        d[slot] = (off, t, t)
    return tuple(d)


# Fallback declaration for a mesh built from scratch, copied from a shipped
# donor: position/normal/uv0/uv1/tangent/binormal + 4 colours.
DECL72 = make_decl([(0, 0, 3), (1, 12, 3), (2, 24, 5), (3, 28, 5), (5, 32, 3),
                    (8, 44, 3), (11, 56, 9), (19, 60, 9), (20, 64, 9),
                    (21, 68, 9)])


# --- axes ---------------------------------------------------------------------
# Game is Y-up right-handed, Blender Z-up right-handed. A pure rotation (det +1),
# so winding survives and normals go through the same function.
def g2b(p):
    """game (x, y, z) -> blender (x, -z, y)."""
    return (p[0], -p[2], p[1])


def b2g(p):
    """blender (x, y, z) -> game (x, z, -y)."""
    return (p[0], p[2], -p[1])


# --- vertex declaration --------------------------------------------------------
def decl_slots(decl):
    """{slot: (offset, type)} for the populated slots."""
    return {i: (o, t) for i, (o, t, _u) in enumerate(decl) if o != NOSLOT}


def decl_pad(decl):
    """Byte offsets inside one vertex that no slot covers (none in the corpus,
    carried anyway so an unknown declaration still round-trips)."""
    covered = set()
    for off, t in decl_slots(decl).values():
        covered.update(range(off, off + TYPE_SIZE[t]))
    return [i for i in range(decl_stride(decl)) if i not in covered]


def check_decl(decl):
    sl = decl_slots(decl)
    if S_POS not in sl or sl[S_POS] != (0, 3):
        raise ValueError('declaration has no float3 position at offset 0')
    for slot, (off, t) in sl.items():
        if t not in TYPE_SIZE:
            raise ValueError('slot %d: unknown element type %d' % (slot, t))
    return sl


# --- render mesh: packed bytes <-> editable geometry ---------------------------
def decode_mesh(pkt):
    """One CMesh render packet -> geometry dict, keyed by slot, in game space.
    Shipped NaN uvs decode to 0.0; their bits survive in uvraw/vecraw for encode."""
    decl = pkt['decl']
    sl = check_decl(decl)
    st = decl_stride(decl)
    n = pkt['nverts']
    vd = pkt['vdata']
    if len(vd) != n * st:
        raise ValueError('vertex data is %d bytes, declaration wants %d'
                         % (len(vd), n * st))
    geo = {'decl': decl, 'nverts': n, 'vec': {}, 'uv': {}, 'color': {},
           'raw': {}, 'vecraw': {}, 'uvraw': {}, 'pad': None, 'f18': pkt['f18'],
           'morphs': pkt['morphs'], 'mdata': pkt.get('mdata', [])}
    for slot, (off, t) in sorted(sl.items()):
        if t == 3:
            vals, odd = [], {}
            for v in range(n):
                p = struct.unpack_from('<3f', vd, v * st + off)
                if not all(math.isfinite(c) for c in p):
                    odd[v] = vd[v * st + off:v * st + off + 12]
                    p = tuple(c if math.isfinite(c) else 0.0 for c in p)
                vals.append(p)
            geo['vec'][slot] = vals
            if odd:
                geo['vecraw'][slot] = odd
        elif t == 5:
            vals, odd = [], {}
            for v in range(n):
                pair = struct.unpack_from('<2H', vd, v * st + off)
                if any(h & 0x7C00 == 0x7C00 for h in pair):
                    odd[v] = pair
                    vals.append(tuple(0.0 if h & 0x7C00 == 0x7C00
                                      else half_to_f32(h) for h in pair))
                else:
                    vals.append(tuple(half_to_f32(h) for h in pair))
            geo['uv'][slot] = vals
            if odd:
                geo['uvraw'][slot] = odd
        elif t == 9:
            geo['color'][slot] = [struct.unpack_from('<4B', vd, v * st + off)
                                  for v in range(n)]
        else:
            sz = TYPE_SIZE[t]
            geo['raw'][slot] = [vd[v * st + off:v * st + off + sz]
                                for v in range(n)]
    pad = decl_pad(decl)
    if pad:
        geo['pad'] = [bytes(vd[v * st + i] for i in pad) for v in range(n)]
    idata = pkt['idata']
    geo['tris'] = [struct.unpack_from('<3H', idata, i * 6)
                   for i in range(pkt['nprims'])]
    return geo


def encode_mesh(geo):
    """Geometry dict -> a CMesh render packet (inverse of decode_mesh)."""
    decl = geo['decl']
    sl = check_decl(decl)
    st = decl_stride(decl)
    n = geo['nverts']
    buf = bytearray(n * st)
    for slot, (off, t) in sorted(sl.items()):
        if t == 3:
            src = geo['vec'].get(slot)
            odd = geo.get('vecraw', {}).get(slot, {})
            for v in range(n):
                if v in odd:
                    buf[v * st + off:v * st + off + 12] = odd[v]
                else:
                    struct.pack_into('<3f', buf, v * st + off, *src[v])
        elif t == 5:
            src = geo['uv'].get(slot)
            odd = geo.get('uvraw', {}).get(slot, {})
            for v in range(n):
                if v in odd:
                    struct.pack_into('<2H', buf, v * st + off, *odd[v])
                else:
                    u, w = src[v]
                    struct.pack_into('<2H', buf, v * st + off,
                                     f32_to_half(u), f32_to_half(w))
        elif t == 9:
            src = geo['color'].get(slot)
            for v in range(n):
                struct.pack_into('<4B', buf, v * st + off,
                                 *(c & 0xFF for c in src[v]))
        else:
            src = geo['raw'].get(slot)
            for v in range(n):
                buf[v * st + off:v * st + off + TYPE_SIZE[t]] = src[v]
    pad = decl_pad(decl)
    if pad and geo.get('pad'):
        for v in range(n):
            for k, i in enumerate(pad):
                buf[v * st + i] = geo['pad'][v][k]
    idata = b''.join(struct.pack('<3H', *t) for t in geo['tris'])
    mdata = geo.get('mdata') or []
    pkt = {'decl': decl, 'nverts': n, 'nprims': len(geo['tris']),
           'f18': geo.get('f18', 0), 'morphs': geo.get('morphs') or [],
           'vdata': bytes(buf), 'idata': idata, 'mdata': list(mdata),
           'bbox': pack_bbox(bbox_of(geo['vec'][S_POS])) if n else
                   struct.pack('<6f', 0, 0, 0, 0, 0, 0)}
    pkt['datasize'] = len(pkt['vdata']) + len(idata) + sum(len(b) for b in mdata)
    return pkt


def mesh_data_size(pkt):
    """Bytes this mesh occupies inside its section's streamed data blob."""
    return (pkt['nverts'] * decl_stride(pkt['decl']) + pkt['nprims'] * 6
            + sum(m['count'] * 0x34 for m in pkt['morphs']))


def bbox_of(points):
    xs, ys, zs = zip(*points)
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def pack_bbox(bb):
    return struct.pack('<6f', *bb)


def union_bbox(boxes):
    bs = [struct.unpack('<6f', b) if isinstance(b, (bytes, bytearray)) else b
          for b in boxes]
    return tuple([min(b[k] for b in bs) for k in range(3)]
                 + [max(b[k] for b in bs) for k in range(3, 6)])


# --- geometry signature (unchanged geometry passes through untouched) ----------
def geo_sig(geo):
    """Stable digest of everything decode_mesh recovers. Triangles rotate to start
    at their lowest index, so re-cornering a triangle does not count as an edit."""
    h = hashlib.sha1()
    h.update(struct.pack('<II', geo['nverts'], len(geo['tris'])))
    for slot in sorted(geo['vec']):
        h.update(b'v%d' % slot)
        for p in geo['vec'][slot]:
            h.update(struct.pack('<3f', *p))
    for slot in sorted(geo['uv']):
        h.update(b'u%d' % slot)
        for u, v in geo['uv'][slot]:
            h.update(struct.pack('<2H', f32_to_half(u), f32_to_half(v)))
    for slot in sorted(geo['color']):
        h.update(b'c%d' % slot)
        for c in geo['color'][slot]:
            h.update(bytes(x & 0xFF for x in c))
    for slot in sorted(geo['raw']):
        h.update(b'r%d' % slot)
        for b in geo['raw'][slot]:
            h.update(b)
    for t in geo['tris']:
        i = min(range(3), key=lambda k: t[k])
        h.update(struct.pack('<3H', t[i], t[(i + 1) % 3], t[(i + 2) % 3]))
    return h.hexdigest()


def coll_sig(verts, tris, surf, tflags):
    h = hashlib.sha1()
    h.update(struct.pack('<II', len(verts), len(tris)))
    for p in verts:
        h.update(struct.pack('<3f', *p))
    for t, s, f in zip(tris, surf, tflags):
        i = min(range(3), key=lambda k: t[k])
        h.update(struct.pack('<3IHH', t[i], t[(i + 1) % 3], t[(i + 2) % 3],
                             s & 0xFFFF, f & 0x3F))
    return h.hexdigest()


# --- collision: BVT <-> triangle soup ------------------------------------------
# Leaf triangles pack into one u32: three 5-bit leaf-local vertex indices, so a
# leaf holds at most 32 vertices, then surface flags and a surface-table index.
TRI_SURF_SHIFT = 21
TRI_FLAG_SHIFT = 15
TRI_FLAG_MASK = 0x3F


def bvt_collect(node):
    """BVT -> (verts, tris, surf, tflags), vertices merged across leaves.
    Merging is bit-exact, so collecting the same tree twice gives the same lists."""
    verts, tris, surf, tflags = [], [], [], []
    pool = {}

    def walk(n):
        if 'kids' in n:
            for k in n['kids']:
                walk(k)
            return
        local = []
        for i in range(len(n['verts']) // 12):
            p = struct.unpack_from('<3f', n['verts'], i * 12)
            key = struct.pack('<3f', *p)
            gi = pool.get(key)
            if gi is None:
                gi = pool[key] = len(verts)
                verts.append(p)
            local.append(gi)
        for i in range(len(n['tris']) // 4):
            k = struct.unpack_from('<I', n['tris'], i * 4)[0]
            tris.append((local[k & 31], local[(k >> 5) & 31],
                         local[(k >> 10) & 31]))
            tflags.append((k >> TRI_FLAG_SHIFT) & TRI_FLAG_MASK)
            surf.append(k >> TRI_SURF_SHIFT)

    if node is not None:
        walk(node)
    return verts, tris, surf, tflags


def bvt_build(verts, tris, surf, tflags, inflate=1.0):
    """Returns (node, arena size). Undersizing the arena silently corrupts the
    tree and the section stops colliding; boxes are inflated so flat geometry works."""
    if not tris:
        return None, 0
    packed = [((s & 0x7FF) << 6) | (f & TRI_FLAG_MASK)
              for s, f in zip(surf, tflags)]
    root, arena = build_bvt(verts, tris, packed)
    if inflate:
        _inflate(root, inflate)
    return root, arena


def _inflate(n, d):
    b = struct.unpack('<6f', n['bbox'])
    n['bbox'] = struct.pack('<6f', b[0] - d, b[1] - d, b[2] - d,
                            b[3] + d, b[4] + d, b[5] + d)
    for k in n.get('kids', ()):
        _inflate(k, d)


def bvt_bbox(node):
    return struct.unpack('<6f', node['bbox']) if node else None


# --- navmesh / breakers / bsp (read-only views) --------------------------------
def nav_polys(nav):
    """One n-gon per navmesh node. Neighbour links and node headers are only
    partly resolved, so this is a read-only view, not an authoring path."""
    if nav.get('empty'):
        return [], []
    verts = [struct.unpack_from('<3f', nav['verts'], i * 12)
             for i in range(nav['nverts'])]
    polys = []
    for d in nav['nodes']:
        ring = [struct.unpack_from('<I', d['v'], i * 4)[0]
                for i in range(len(d['v']) // 4)]
        if len(ring) >= 3 and all(v < len(verts) for v in ring):
            polys.append(ring)
    return verts, polys


def breaker_geom(b):
    """(verts, tris) for one CBreaker. Only the leading float3 of each 0x2C-byte
    vertex is resolved, so breakers import read-only."""
    verts = [struct.unpack_from('<3f', v, 0) for v in b['verts']]
    tris = [struct.unpack_from('<3H', b['tris'], i * 6)
            for i in range(len(b['tris']) // 6)]
    return verts, tris


def bsp_nodes(m):
    """[(plane, section, front, back, bbox)] for the collision BSP."""
    out = []
    for i in range(len(m['bsp']) // 0x30):
        rec = m['bsp'][i * 0x30:(i + 1) * 0x30]
        plane = struct.unpack_from('<4f', rec, 0)
        sec, front, back, _pad = struct.unpack_from('<4h', rec, 16)
        out.append((plane, sec, front, back, struct.unpack_from('<6f', rec, 24)))
    return out


# --- names ---------------------------------------------------------------------
def name_str(raw):
    """Fixed-width NUL-terminated engine name -> str."""
    e = raw.find(b'\0')
    return raw[:e if e >= 0 else len(raw)].decode('latin1')


def name_bytes(s, width):
    b = s.encode('latin1')[:width - 1]
    return b + b'\0' * (width - len(b))


def material_name(entry, index):
    """Display name for one entry of the set's material table."""
    ref = name_str(entry['ref'][0])
    return '%03d_%s' % (index, ref.replace('\\', '/') if ref else 'embedded')


# --- layout --------------------------------------------------------------------
BLOB_ALIGN = 32


def capture(m):
    """Call once right after bst.parse() and before editing: it splits each section's
    blob into mesh data and the trailing ballast that must be copied through."""
    for s in m['sections']:
        pre = sum(mesh_data_size(me['pkt']) for me in s['meshes']
                  if not (me['flags'] & 0x20000))
        pre += (-pre) % BLOB_ALIGN
        s['ballast'] = bytes(s['blob'][pre:])
        s['origbbox'] = s['bbox']
    m['origdatasize'] = m['datasize']
    return m


def relayout(m, tight=False):
    """Rebuild what bst.build() will not: per-section blob, dataofs/fa20, the
    streaming-buffer size, mesh bboxes. relayout(capture(parse(d))) rebuilds d."""
    if 'origdatasize' not in m:
        raise ValueError('call capture(m) right after bst.parse()')
    if len(m['watervis']) != len(m['sections']):
        raise ValueError('watervis is %d bytes for %d sections'
                         % (len(m['watervis']), len(m['sections'])))
    for s in m['sections']:
        parts = []
        for me in s['meshes']:
            if me['flags'] & 0x20000:
                continue
            p = me['pkt']
            own = [p['vdata'], p['idata']] + list(p.get('mdata') or [])
            p['datasize'] = sum(len(b) for b in own)
            parts.extend(own)
        body = b''.join(parts)
        body += b'\0' * ((-len(body)) % BLOB_ALIGN)
        s['blob'] = body + s.get('ballast', b'')
        s['fa20'] = len(s['blob'])
    biggest = max([s['fa20'] for s in m['sections']] or [0])
    m['datasize'] = biggest if tight else max(m['origdatasize'], biggest)
    # Fixed part first, then each blob 32-aligned end to end -- the rule every
    # shipped set follows exactly.
    for s in m['sections']:
        s['dataofs'], s['datagap'] = 0, b''
    fixed = len(bst.build(m)) - sum(s['fa20'] for s in m['sections'])
    cur = fixed
    for s in m['sections']:
        pad = (-cur) % BLOB_ALIGN
        s['datagap'] = b'\0' * pad
        s['dataofs'] = cur + pad
        cur = s['dataofs'] + s['fa20']
    return m


def section_bbox(s, pad=0.0):
    """Union of a section's mesh and collision boxes.  The engine unions these
    into the set bounds; a camera outside every section renders nothing."""
    boxes = [me['pkt']['bbox'] for me in s['meshes']
             if not (me['flags'] & 0x20000) and me['pkt']['nverts']]
    if s.get('bvt') is not None:
        boxes.append(bvt_bbox(s['bvt']))
    if not boxes:
        return None
    bb = union_bbox(boxes)
    if pad:
        bb = tuple([v - pad for v in bb[:3]] + [v + pad for v in bb[3:]])
    return bb


# --- summary / OBJ dump --------------------------------------------------------
def section_stats(s):
    nv = nt = 0
    for me in s['meshes']:
        if me['flags'] & 0x20000:
            continue
        nv += me['pkt']['nverts']
        nt += me['pkt']['nprims']
    _cv, ct, _cs, _cf = bvt_collect(s['bvt']) if s['bvt'] else ([], [], [], [])
    return {'name': name_str(s['name']), 'meshes': len(s['meshes']),
            'verts': nv, 'tris': nt, 'colltris': len(ct),
            'lights': len(s['arr818']) // 4, 'probes': len(s['portidx']) // 4,
            'surfaces': len(s['lrefs'])}


def cmd_info(a):
    m = bst.parse(_read(a.file))
    print('%s: %d sections, %d materials, %d lights, %d probes, %d breakers'
          % (os.path.basename(a.file), len(m['sections']), len(m['materials']),
             len(m['lights']), len(m['portals']), len(m['breakers'])))
    print('%-4s %-24s %6s %9s %9s %9s %5s %5s'
          % ('#', 'section', 'meshes', 'verts', 'tris', 'colltris', 'lit',
             'surf'))
    for i, s in enumerate(m['sections']):
        st = section_stats(s)
        print('%-4d %-24s %6d %9d %9d %9d %5d %5d'
              % (i, st['name'][:24], st['meshes'], st['verts'], st['tris'],
                 st['colltris'], st['lights'], st['surfaces']))
    if a.materials:
        for i, e in enumerate(m['materials']):
            print('  material %s' % material_name(e, i))


def cmd_check(a):
    ok = True
    for f in a.files:
        d = _read(f)
        try:
            m = capture(bst.parse(d))
            nm = 0
            for s in m['sections']:
                for me in s['meshes']:
                    if me['flags'] & 0x20000:
                        continue
                    p = me['pkt']
                    q = encode_mesh(decode_mesh(p))
                    for k in ('vdata', 'idata', 'nverts', 'nprims'):
                        if q[k] != p[k]:
                            raise ValueError('%s mesh %s: %s differs'
                                             % (name_str(s['name']),
                                                name_str(me['name']), k))
                    nm += 1
            out = bst.build(relayout(m))
            if out != d:
                n = next((i for i in range(min(len(d), len(out)))
                          if d[i] != out[i]), min(len(d), len(out)))
                raise ValueError('relayout differs @%#x (len %d vs %d)'
                                 % (n, len(d), len(out)))
            print('OK   %-24s %d meshes decode/encode exact, relayout exact'
                  % (os.path.basename(f), nm))
        except Exception as e:
            print('FAIL %s: %s' % (f, e))
            ok = False
    sys.exit(0 if ok else 1)


def cmd_roundtrip(a):
    a.files = sorted(glob.glob(os.path.join(a.dir, '**', '*.bst'),
                               recursive=True))
    if not a.files:
        sys.exit('no .bst under %s' % a.dir)
    cmd_check(a)


def cmd_to_obj(a):
    m = bst.parse(_read(a.file))
    want = set(a.section) if a.section else None
    out = ['# %s -> obj by gbtvgr bst-geom (game axes, Y up)'
           % os.path.basename(a.file)]
    base = 1
    for si, s in enumerate(m['sections']):
        if want is not None and si not in want and name_str(s['name']) not in want:
            continue
        for mi, me in enumerate(s['meshes']):
            if me['flags'] & 0x20000:
                continue
            geo = decode_mesh(me['pkt'])
            if not geo['nverts']:
                continue
            out.append('g s%02d_%s_%s' % (si, name_str(s['name']),
                                          name_str(me['name'])))
            out.append('usemtl %s' % material_name(
                m['materials'][struct.unpack('<H', me['h70'])[0]],
                struct.unpack('<H', me['h70'])[0]))
            for p in geo['vec'][S_POS]:
                out.append('v %r %r %r' % p)
            uvs = geo['uv'].get(2)
            if uvs:
                for u, v in uvs:
                    out.append('vt %r %r' % (u, v))
            for t in geo['tris']:
                if uvs:
                    out.append('f %d/%d %d/%d %d/%d'
                               % (base + t[0], base + t[0], base + t[1],
                                  base + t[1], base + t[2], base + t[2]))
                else:
                    out.append('f %d %d %d'
                               % (base + t[0], base + t[1], base + t[2]))
            base += geo['nverts']
        if a.collision and s['bvt']:
            cv, ct, _cs, _cf = bvt_collect(s['bvt'])
            out.append('g s%02d_%s_collision' % (si, name_str(s['name'])))
            for p in cv:
                out.append('v %r %r %r' % p)
            for t in ct:
                out.append('f %d %d %d' % (base + t[0], base + t[1], base + t[2]))
            base += len(cv)
    with open(a.outfile, 'w') as fh:
        fh.write('\n'.join(out) + '\n')
    print('wrote %s (%d vertices)' % (a.outfile, base - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('info')
    p.add_argument('file')
    p.add_argument('--materials', action='store_true')
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser('check')
    p.add_argument('files', nargs='+')
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser('roundtrip-test')
    p.add_argument('dir')
    p.set_defaults(fn=cmd_roundtrip)
    p = sub.add_parser('to-obj')
    p.add_argument('file')
    p.add_argument('outfile')
    p.add_argument('--section', action='append', default=[],
                   help='section index or name (repeatable; default all)')
    p.add_argument('--collision', action='store_true')
    p.set_defaults(fn=cmd_to_obj)
    a = ap.parse_args()
    if getattr(a, 'section', None):
        a.section = [int(s) if s.isdigit() else s for s in a.section]
    a.fn(a)


if __name__ == '__main__':
    main()

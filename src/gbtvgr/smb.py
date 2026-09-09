#!/usr/bin/env python3
"""
Static meshes and .mtb material tables: OBJ both ways.

File is a 16-aligned header tree followed by per-part vertex/index/morph data.
The OBJ writer follows sakis720's obj2smb.

Commands:
  info <file.smb>                  summary
  verify <file.smb ...>            parse->rebuild, must be byte-identical
  roundtrip-test <dir>             verify every .smb under dir
  mtb-info <file.mtb>              material record summary
  mtb-roundtrip-test <dir>         verify every .mtb under dir
  to-obj <file.smb> <out.obj>      geometry+uv+normals (+ .json sidecar, .mtl, collision obj)
  from-obj <in.obj> <out.smb>      build a new .smb (see the subcommand's --help)
"""
import argparse, glob, io, json, os, struct, sys

VERS = (0x13, 0x19E, 2, 9, 6)   # smb, material, mesh, renderpacket, collisionmesh
TYPE_SIZE = {1: 4, 2: 8, 3: 12, 4: 16, 5: 4, 9: 4}   # vertex decl element byte sizes


def _read(path):
    with open(path, 'rb') as fh:
        return fh.read()


def want(got, expect, what):
    """Wire-format version gate. A bare assert would vanish under `python -O`."""
    if got != expect:
        raise ValueError('%s: expected %#x, got %#x' % (what, expect, got))
    return got


class R:
    def __init__(self, d):
        self.d = d; self.o = 0
    def take(self, n):
        b = self.d[self.o:self.o+n]
        if len(b) != n: raise ValueError('EOF @%#x (+%d)' % (self.o, n))
        self.o += n; return b
    def u32(self): return struct.unpack('<I', self.take(4))[0]
    def i32(self): return struct.unpack('<i', self.take(4))[0]
    def u16(self): return struct.unpack('<H', self.take(2))[0]
    def f32(self): return struct.unpack('<f', self.take(4))[0]
    def cstr(self):
        e = self.d.index(b'\0', self.o)
        s = self.d[self.o:e]; self.o = e + 1; return s
    def align(self, n):
        pad = (-self.o) % n
        return self.take(pad)


class W:
    def __init__(self):
        self.b = io.BytesIO()
    def w(self, b): self.b.write(b)
    def u32(self, v): self.w(struct.pack('<I', v))
    def u16(self, v): self.w(struct.pack('<H', v))
    def f32(self, v): self.w(struct.pack('<f', v))
    def cstr(self, s): self.w(s + b'\0')
    def align(self, n, fill=b'\0'):
        pad = (-self.b.tell()) % n
        self.w(fill * pad)
    def data(self): return self.b.getvalue()


def read_deps(r):
    """Repeat { path + 16-byte hash } until an empty path, then align 4.
    The enclosing record's own hash was already read by the caller."""
    deps = []
    while r.d[r.o] != 0:
        name = r.cstr()
        deps.append((name, r.take(16)))
    r.o += 1
    pad = r.align(4)
    return deps, pad

def write_deps(w, deps):
    for name, h in deps:
        w.cstr(name); w.w(h)
    w.w(b'\0'); w.align(4)


def read_matref_name(r):
    s = r.cstr(); pad = r.align(4)
    return s, pad


# --- embedded material (CMaterialGen3) ---------------------------------------
def read_c2dtransform(r):
    t = {}
    want(r.u32(), 1, 'C2DTransform ver')
    t['f0'] = r.take(4)
    waves = []
    for _ in range(6):
        want(r.u32(), 1, 'CWaveformControl ver')
        waves.append(r.take(0x14))
    t['waves'] = waves
    t['f7c'] = r.take(4); t['f80'] = r.take(4)
    n = r.u32()
    t['pairs'] = [(r.take(4), r.take(4)) for _ in range(n)]
    return t

def write_c2dtransform(w, t):
    w.u32(1); w.w(t['f0'])
    for wv in t['waves']: w.u32(1); w.w(wv)
    w.w(t['f7c']); w.w(t['f80']); w.u32(len(t['pairs']))
    for a, b in t['pairs']: w.w(a); w.w(b)

def read_shader_core(r):
    c = {'ver': r.u32()}
    if c['ver'] not in (2, 3):
        raise ValueError('shader core ver: expected 2 or 3, got %#x' % c['ver'])
    if c['ver'] == 3: c['crc'] = r.u32()
    n = r.u32()
    c['code'] = r.take(n)
    c['regs'] = [r.take(4) for _ in range(r.u32())]
    return c

def write_shader_core(w, c):
    w.u32(c['ver'])
    if c['ver'] == 3: w.u32(c['crc'])
    w.u32(len(c['code'])); w.w(c['code'])
    w.u32(len(c['regs']))
    for x in c['regs']: w.w(x)

def read_shader_set(r):
    s = {}
    want(r.u32(), 0x19E, 'shader set ver')
    nvs, nps, npair = r.u32(), r.u32(), r.u32()
    s['flags'] = r.u32()
    s['vs'] = []
    for _ in range(nvs):
        core = read_shader_core(r); core['sig'] = r.take(4)   # VS: +1 u32
        s['vs'].append(core)
    s['ps'] = []
    for _ in range(nps):
        core = read_shader_core(r)                            # PS: +u32 n + n*u32
        core['tex'] = [r.take(4) for _ in range(r.u32())]
        s['ps'].append(core)
    s['pairs'] = [(r.u32(), r.u32(), r.u32()) for _ in range(npair)]
    return s

def write_shader_set(w, s):
    w.u32(0x19E); w.u32(len(s['vs'])); w.u32(len(s['ps'])); w.u32(len(s['pairs']))
    w.u32(s['flags'])
    for c in s['vs']: write_shader_core(w, c); w.w(c['sig'])
    for c in s['ps']:
        write_shader_core(w, c); w.u32(len(c['tex']))
        for x in c['tex']: w.w(x)
    for a, b, c in s['pairs']: w.u32(a); w.u32(b); w.u32(c)

def read_material(r):
    m = {}
    want(r.u32(), 0x19E, 'material ver')
    m['hash'] = r.take(16)
    m['deps'], m['deppad'] = read_deps(r)
    m['f8'], m['f47c'], m['f10'], m['fc'], m['f79c'] = (r.take(4) for _ in range(5))
    m['layers'] = []
    for _ in range(r.u32()):
        m['layers'].append({'f98': r.take(16), 'name': r.take(0x40),
                            'f8c': r.take(4), 'f94': r.take(4), 'f90': r.take(4),
                            'fa8': r.take(4), 'fac': r.take(4)})
    m['t2d'] = [read_c2dtransform(r) for _ in range(r.u32())]
    m['blend'] = []
    for _ in range(r.u32()):
        e = {'f8': r.take(16)}
        n = r.u32()
        e['f48'] = r.take(4)
        e['vals'] = [r.take(4) for _ in range(n)]
        m['blend'].append(e)
    m['shaders'] = read_shader_set(r)
    return m

def write_material(w, m):
    w.u32(0x19E); w.w(m['hash'])
    for name, h in m['deps']:
        w.cstr(name); w.w(h)
    w.w(b'\0'); w.w(m['deppad'])
    for k in ('f8', 'f47c', 'f10', 'fc', 'f79c'): w.w(m[k])
    w.u32(len(m['layers']))
    for l in m['layers']:
        w.w(l['f98']); w.w(l['name'])
        for k in ('f8c', 'f94', 'f90', 'fa8', 'fac'): w.w(l[k])
    w.u32(len(m['t2d']))
    for t in m['t2d']: write_c2dtransform(w, t)
    w.u32(len(m['blend']))
    for e in m['blend']:
        w.w(e['f8']); w.u32(len(e['vals'])); w.w(e['f48'])
        for v in e['vals']: w.w(v)
    write_shader_set(w, m['shaders'])


# --- collision (CCollisionMesh + CBoundingVolumeTree) -------------------------
def read_bvt_node(r):
    n = {'bbox': r.take(24)}
    if r.u32() == 0:
        nv, nt = r.u16(), r.u16()
        n['verts'] = r.take(nv * 12)
        n['tris'] = r.take(nt * 4)
    else:
        n['kids'] = [read_bvt_node(r), read_bvt_node(r)]
    return n

def write_bvt_node(w, n):
    w.w(n['bbox'])
    if 'kids' in n:
        w.u32(1)
        write_bvt_node(w, n['kids'][0]); write_bvt_node(w, n['kids'][1])
    else:
        w.u32(0); w.u16(len(n['verts']) // 12); w.u16(len(n['tris']) // 4)
        w.w(n['verts']); w.w(n['tris'])

def bvt_arena(n):
    """Engine arena bytes: internal node = 0x38+0x38 headroom, leaf = 0x18."""
    if 'kids' in n:
        return 0x70 + bvt_arena(n['kids'][0]) + bvt_arena(n['kids'][1])
    return 0x18

def read_collision(r):
    c = {'name': r.take(0x30)}
    want(r.u32(), 6, 'collision ver')
    c['flag'] = r.u32()
    nv, nt = r.u32(), r.u32()
    c['verts'] = r.take(nv * 12)
    c['tris'] = r.take(nt * 6)
    c['triflags'] = r.take(nt)
    c['bvt'] = None
    if nv and nt:
        want(r.u32(), 3, 'bvt ver')
        arena = r.u32()
        c['bvt'] = read_bvt_node(r) if arena else None
        c['bvt_arena'] = arena
    return c

def write_collision(w, c):
    w.w(c['name']); w.u32(6); w.u32(c['flag'])
    nv, nt = len(c['verts']) // 12, len(c['tris']) // 6
    w.u32(nv); w.u32(nt)
    w.w(c['verts']); w.w(c['tris']); w.w(c['triflags'])
    if nv and nt:
        w.u32(3); w.u32(c['bvt_arena'])
        if c['bvt'] is not None: write_bvt_node(w, c['bvt'])


# --- parts (CMesh + CRenderPacket) -------------------------------------------
def decl_stride(decl):
    st = 0
    for off, t, _ in decl:
        if off == 0xFFFFFFFF: continue
        st = max(st, off + TYPE_SIZE[t])
    return st

def read_part(r):
    p = {'name': r.take(0x30), 'mat_idx': r.u32()}
    want(r.u32(), 2, 'mesh ver')
    p['bbox'] = r.take(24)
    want(r.u32(), 9, 'renderpacket ver')
    p['datasize'] = r.u32()   # part's data-section byte size (vdata+idata)
    p['decl'] = [(r.u32(), r.u32(), r.u32()) for _ in range(22)]
    p['nverts'], p['nprims'], p['f18'] = r.u32(), r.u32(), r.u32()
    p['morphs'] = []
    for _ in range(r.u32()):
        p['morphs'].append({'idx': r.u16(), 'chan': r.u32(), 'count': r.u32()})
    return p

def write_part(w, p):
    w.w(p['name']); w.u32(p['mat_idx']); w.u32(2); w.w(p['bbox'])
    w.u32(9); w.u32(p['datasize'])
    for a, b, c in p['decl']: w.u32(a); w.u32(b); w.u32(c)
    w.u32(p['nverts']); w.u32(p['nprims']); w.u32(p['f18']); w.u32(len(p['morphs']))
    for m in p['morphs']:
        w.u16(m['idx']); w.u32(m['chan']); w.u32(m['count'])


# --- whole model (CModel header, then data) ----------------------------------
def parse(d):
    r = R(d)
    if struct.unpack_from('<5I', d, 0) != VERS:
        raise ValueError('bad version block %s' % (struct.unpack_from('<5I', d, 0),))
    r.o = 20
    m = {'hash': r.take(16)}
    m['deps'], m['deppad'] = read_deps(r)
    nparts, ncoll, naux, nmat, m['nposes'] = (r.u32() for _ in range(5))
    m['lod'] = r.f32()
    m['materials'] = []
    for _ in range(nmat):
        name, pad = read_matref_name(r)
        e = {'ref': name, 'refpad': pad, 'embedded': None}
        if not name:
            e['embedded'] = read_material(r)
        m['materials'].append(e)
    m['collisions'] = [read_collision(r) for _ in range(ncoll)]
    m['auxnames'] = [r.take(0x30) for _ in range(naux)]
    m['bbox'] = r.take(24)
    m['parts'] = [read_part(r) for _ in range(nparts)]
    m['parents'] = m['poses'] = None
    if m['nposes'] > 1 or naux > 0:
        total = nparts + ncoll + naux
        m['parents'] = [r.u16() for _ in range(total)]
        m['poses'] = [[(r.take(16), r.take(12)) for _ in range(total)]
                      for _ in range(m['nposes'])]
    m['hdrpad'] = r.align(16)
    for p in m['parts']:
        p['vdata'] = r.take(p['nverts'] * decl_stride(p['decl']))
        p['idata'] = r.take(p['nprims'] * 6)
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
    w.u32(len(m['parts'])); w.u32(len(m['collisions'])); w.u32(len(m['auxnames']))
    w.u32(len(m['materials'])); w.u32(m['nposes']); w.f32(m['lod'])
    for e in m['materials']:
        w.cstr(e['ref']); w.w(e['refpad'])
        if e['embedded'] is not None:
            write_material(w, e['embedded'])
    for c in m['collisions']: write_collision(w, c)
    for a in m['auxnames']: w.w(a)
    w.w(m['bbox'])
    for p in m['parts']: write_part(w, p)
    if m['parents'] is not None:
        for x in m['parents']: w.u16(x)
        for pose in m['poses']:
            for q, v in pose: w.w(q); w.w(v)
    w.w(m['hdrpad'])
    for p in m['parts']:
        w.w(p['vdata']); w.w(p['idata'])
        for md in p['mdata']: w.w(md)
    return w.data()


def nm(b):
    return b.split(b'\0')[0].decode('latin1')

def cmd_info(a):
    d = _read(a.infile)
    m = parse(d)
    print('%s: %d bytes, %d part(s), %d collision, %d aux, %d material(s), %d pose(s), lod %.1f'
          % (a.infile, len(d), len(m['parts']), len(m['collisions']), len(m['auxnames']),
             len(m['materials']), m['nposes'], m['lod']))
    for e in m['materials']:
        if e['embedded'] is not None:
            mat = e['embedded']
            texs = ', '.join(nm(l['name']) for l in mat['layers'])
            print('  material <embedded>: %d layer(s) [%s], %d VS + %d PS'
                  % (len(mat['layers']), texs, len(mat['shaders']['vs']), len(mat['shaders']['ps'])))
        else:
            print('  material ref: %s' % e['ref'].decode('latin1'))
    for p in m['parts']:
        print('  part %-32s mat=%d verts=%-6d prims=%-6d stride=%d morphs=%d'
              % (nm(p['name']), p['mat_idx'], p['nverts'], p['nprims'],
                 decl_stride(p['decl']), len(p['morphs'])))
    for c in m['collisions']:
        print('  coll %-32s verts=%-4d tris=%-4d flag=%d'
              % (nm(c['name']), len(c['verts'])//12, len(c['tris'])//6, c['flag']))
    for x in m['auxnames']:
        print('  aux  %s' % nm(x))

def cmd_verify(a):
    bad = 0
    for fn in a.infiles:
        d = _read(fn)
        try:
            out = build(parse(d))
            ok = out == d
        except Exception as e:
            print('%s: PARSE FAIL %s' % (fn, e)); bad += 1; continue
        if not ok:
            i = next(i for i, (x, y) in enumerate(zip(out, d)) if x != y) if len(out) == len(d) else -1
            print('%s: MISMATCH len %d->%d first diff @%#x' % (fn, len(d), len(out), i))
            bad += 1
    print('%d/%d byte-identical' % (len(a.infiles) - bad, len(a.infiles)))
    return 1 if bad else 0

def cmd_roundtrip_test(a):
    files = sorted(glob.glob(os.path.join(a.dir, '**', '*.smb'), recursive=True))
    ok = 0; fails = []
    for fn in files:
        d = _read(fn)
        try:
            if build(parse(d)) == d: ok += 1
            else: fails.append((fn, 'mismatch'))
        except Exception as e:
            fails.append((fn, str(e)))
    print('%d/%d byte-identical round-trip' % (ok, len(files)))
    for fn, e in fails[:15]:
        print('  FAIL %s: %s' % (fn, e))
    return 1 if fails else 0




# ============================ OBJ export / import ==============================
import base64, math

DECL60 = tuple([(0, 3, 3), (12, 3, 3), (24, 5, 5)] + [(0xFFFFFFFF, 0, 0)] * 2 +
               [(28, 3, 3)] + [(0xFFFFFFFF, 0, 0)] * 2 + [(40, 3, 3)] +
               [(0xFFFFFFFF, 0, 0)] * 2 + [(56, 9, 9), (52, 5, 5)] +
               [(0xFFFFFFFF, 0, 0)] * 9)   # the most common shipped declaration

def half_to_f32(h):
    s, e, f = (h >> 15) & 1, (h >> 10) & 0x1F, h & 0x3FF
    if e == 0:
        v = f * 2.0 ** -24
    elif e == 31:
        v = float('inf') if f == 0 else float('nan')
    else:
        v = (1 + f / 1024.0) * 2.0 ** (e - 15)
    return -v if s else v

def f32_to_half(x):
    """Truncating f32->f16 (matches the game's own data; sign-preserving)."""
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    s, e, f = (bits >> 31) << 15, (bits >> 23) & 0xFF, bits & 0x7FFFFF
    if e == 0: return s
    if e == 255: return s | 0x7C00 | (f >> 13)
    he = e - 127 + 15
    if he >= 31: return s | 0x7C00
    if he <= 0:
        if he < -10: return s
        return s | ((f | 0x800000) >> (1 - he) >> 13)
    return s | (he << 10) | (f >> 13)

def ff(x):
    return repr(x)

def geo_offsets(decl):
    """pos/normal/uv offsets (slots 0/1/2) - identical in all shipped decls."""
    pos, nrm, uv = decl[0], decl[1], decl[2]
    if not (pos[0] == 0 and pos[1] == 3 and nrm == (12, 3, 3) and uv[:2] == (24, 5)):
        raise ValueError('unexpected geometry slots in declaration: %r' % (decl[:3],))
    return 0, 12, 24

def san(name):
    return ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in name) or '_'

def b64(b): return base64.b64encode(b).decode()
def unb64(s): return base64.b64decode(s)

def jsonify(x):
    if isinstance(x, bytes): return {'b64': b64(x)}
    if isinstance(x, dict): return {k: jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonify(v) for v in x]
    return x

def unjsonify(x):
    if isinstance(x, dict):
        if set(x) == {'b64'}: return unb64(x['b64'])
        return {k: unjsonify(v) for k, v in x.items()}
    if isinstance(x, list): return [unjsonify(v) for v in x]
    return x


def cmd_to_obj(a):
    m = parse(_read(a.infile))
    base_noext = os.path.splitext(a.outfile)[0]
    mtl = os.path.basename(base_noext) + '.mtl'
    obj = ['# exported by smb.py from %s' % os.path.basename(a.infile),
           'mtllib %s' % mtl]
    mtln = []
    for i, e in enumerate(m['materials']):
        if e['embedded'] is not None:
            name = 'embedded_%d' % i
            layers = [nm(l['name']) for l in e['embedded']['layers']]
            mtln.append('newmtl %s' % name)
            for l in layers: mtln.append('# layer: %s' % l)
            if layers: mtln.append('map_Kd %s' % layers[0].replace('\\', '/'))
        else:
            name = san(e['ref'].decode('latin1'))
            mtln.append('newmtl %s' % name)
            mtln.append('# game material ref: %s' % e['ref'].decode('latin1'))
        mtln.append('')
    vbase = 1
    for pi, p in enumerate(m['parts']):
        po, no, uo = geo_offsets(p['decl'])
        st = decl_stride(p['decl'])
        obj.append('g part_%d_%s' % (pi, san(nm(p['name']))))
        e = m['materials'][p['mat_idx']]
        obj.append('usemtl %s' % ('embedded_%d' % p['mat_idx'] if e['embedded'] is not None
                                  else san(e['ref'].decode('latin1'))))
        vd = p['vdata']
        for v in range(p['nverts']):
            b = vd[v*st:(v+1)*st]
            x, y, z = struct.unpack_from('<3f', b, po)
            nx, ny, nz = struct.unpack_from('<3f', b, no)
            hu, hv = struct.unpack_from('<2H', b, uo)
            obj.append('v %s %s %s' % (ff(x), ff(y), ff(z)))
            obj.append('vn %s %s %s' % (ff(nx), ff(ny), ff(nz)))
            obj.append('vt %s %s' % (ff(half_to_f32(hu)), ff(half_to_f32(hv))))
        for t in range(p['nprims']):
            i0, i1, i2 = struct.unpack_from('<3H', p['idata'], t*6)
            obj.append('f %d/%d/%d %d/%d/%d %d/%d/%d' %
                       tuple(x for i in (i0, i1, i2) for x in (vbase+i,)*3))
        vbase += p['nverts']
    for ci, c in enumerate(m['collisions']):
        obj.append('g coll_%d_%s' % (ci, san(nm(c['name']))))
        for x, y, z in struct.iter_unpack('<3f', c['verts']):
            obj.append('v %s %s %s' % (ff(x), ff(y), ff(z)))
        for i0, i1, i2 in struct.iter_unpack('<3H', c['tris']):
            obj.append('f %d %d %d' % (vbase+i0, vbase+i1, vbase+i2))
        vbase += len(c['verts']) // 12
    with open(a.outfile, 'w') as fh:
        fh.write('\n'.join(obj) + '\n')
    with open(os.path.join(os.path.dirname(a.outfile) or '.', mtl), 'w') as fh:
        fh.write('\n'.join(mtln) + '\n')
    # sidecar: everything the OBJ cannot carry, for byte-exact reconstruction
    side = dict(m)
    side['parts'] = []
    for p in m['parts']:
        st = decl_stride(p['decl'])
        sp = dict(p)
        sp['extra'] = b''.join(p['vdata'][v*st+28:(v+1)*st] for v in range(p['nverts']))
        gf = [x for v in range(p['nverts'])
              for x in struct.unpack_from('<7f', p['vdata'], v*st)]
        if not all(math.isfinite(x) for x in gf):
            sp['georaw'] = b''.join(p['vdata'][v*st:v*st+28] for v in range(p['nverts']))
        del sp['vdata'], sp['idata']
        side['parts'].append(sp)
    with open(base_noext + '.smb.json', 'w') as fh:
        json.dump(jsonify(side), fh)
    print('wrote %s + %s + %s.smb.json (%d parts, %d collision, %d materials)' %
          (a.outfile, mtl, base_noext, len(m['parts']), len(m['collisions']), len(m['materials'])))


def load_obj(path):
    vs, vns, vts = [], [], []
    groups = []          # (name, mtl, [((vi,ti,ni) x3), ...])
    cur = None; curmtl = None
    def face_idx(tok):
        p = (tok.split('/') + ['', ''])[:3]
        return tuple(int(x) - 1 if x else None for x in p)
    with open(path) as fh:
        lines = fh.readlines()
    for line in lines:
        t = line.split()
        if not t: continue
        if t[0] == 'v': vs.append(tuple(float(x) for x in t[1:4]))
        elif t[0] == 'vn': vns.append(tuple(float(x) for x in t[1:4]))
        elif t[0] == 'vt': vts.append(tuple(float(x) for x in t[1:3]))
        elif t[0] == 'g' or t[0] == 'o':
            cur = (t[1] if len(t) > 1 else 'default', [curmtl])
            groups.append((cur[0], cur[1], []))
        elif t[0] == 'usemtl':
            curmtl = t[1]
            if groups: groups[-1][1][0] = curmtl
        elif t[0] == 'f':
            if not groups: groups.append(('default', [curmtl], []))
            idx = [face_idx(x) for x in t[1:]]
            for k in range(1, len(idx) - 1):        # fan-triangulate
                groups[-1][2].append((idx[0], idx[k], idx[k+1]))
    return vs, vns, vts, [(g, mtl[0], f) for g, mtl, f in groups]


def pack_geo(pos, norm, uv):
    return struct.pack('<6f2H', *pos, *norm, f32_to_half(uv[0]), f32_to_half(uv[1]))

def rebuild_exact(a):
    with open(a.sidecar) as fh:
        side = unjsonify(json.load(fh))
    vs, vns, vts, groups = load_obj(a.infile)
    parts_faces = {}
    coll_by_idx = {}
    for g, _, faces in groups:
        if g.startswith('part_'): parts_faces[int(g.split('_')[1])] = faces
        elif g.startswith('coll_'): coll_by_idx[int(g.split('_')[1])] = faces
    m = side
    vbase = 0
    for pi, p in enumerate(m['parts']):
        st = decl_stride(p['decl'])
        ex = p.pop('extra'); georaw = p.pop('georaw', None)
        exst = st - 28
        vd = bytearray()
        for v in range(p['nverts']):
            if georaw is not None:
                vd += georaw[v*28:(v+1)*28]
            else:
                vd += pack_geo(vs[vbase+v], vns[vbase+v], vts[vbase+v])
            vd += ex[v*exst:(v+1)*exst]
        p['vdata'] = bytes(vd)
        idata = bytearray()
        for (a0, _, _), (b0, _, _), (c0, _, _) in parts_faces.get(pi, []):
            idata += struct.pack('<3H', a0 - vbase, b0 - vbase, c0 - vbase)
        p['idata'] = bytes(idata)
        if len(p['idata']) != p['nprims'] * 6:
            raise ValueError('face count changed for part %d' % pi)
        vbase += p['nverts']
    return m

# --- new-mesh construction ------------------------------------------------------
def vsub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def norml(v):
    l = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
    return (v[0]/l, v[1]/l, v[2]/l) if l > 1e-12 else (1.0, 0.0, 0.0)

def compute_tangents(verts, faces):
    """verts: [(pos, norm, uv)]; returns per-vertex (tangent, binormal)."""
    tan = [(0.0, 0.0, 0.0)] * len(verts); bin_ = list(tan)
    for i0, i1, i2 in faces:
        p0, p1, p2 = verts[i0][0], verts[i1][0], verts[i2][0]
        u0, u1, u2 = verts[i0][2], verts[i1][2], verts[i2][2]
        e1, e2 = vsub(p1, p0), vsub(p2, p0)
        du1, dv1 = u1[0]-u0[0], u1[1]-u0[1]
        du2, dv2 = u2[0]-u0[0], u2[1]-u0[1]
        d = du1*dv2 - du2*dv1
        if abs(d) < 1e-12: continue
        r = 1.0 / d
        t = tuple((e1[k]*dv2 - e2[k]*dv1) * r for k in range(3))
        b = tuple((e2[k]*du1 - e1[k]*du2) * r for k in range(3))
        for i in (i0, i1, i2):
            tan[i] = tuple(tan[i][k]+t[k] for k in range(3))
            bin_[i] = tuple(bin_[i][k]+b[k] for k in range(3))
    out = []
    for i, (pos, n, uv) in enumerate(verts):
        t = norml(tuple(tan[i][k] - n[k]*sum(n[j]*tan[i][j] for j in range(3)) for k in range(3)))
        b = norml(cross(n, t)) if bin_[i] == (0.0, 0.0, 0.0) else norml(bin_[i])
        out.append((t, b))
    return out

def bbox_of(points):
    xs, ys, zs = zip(*points)
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))

def pack_bbox(bb): return struct.pack('<6f', *bb)

def build_bvt(verts, tris, flags):
    """verts: [(x,y,z)], tris: [(a,b,c)], flags: per-tri byte. Real recursive
    median-split tree; leaf <=16 tris and <=31 local verts (5-bit indices)."""
    def centroid(ti):
        return tuple(sum(verts[i][k] for i in tris[ti]) / 3.0 for k in range(3))
    def make(idxs):
        pts = [verts[i] for t in idxs for i in tris[t]]
        bb = bbox_of(pts)
        lverts = sorted(set(i for t in idxs for i in tris[t]))
        if len(idxs) <= 16 and len(lverts) <= 31:
            lmap = {g: l for l, g in enumerate(lverts)}
            vblob = b''.join(struct.pack('<3f', *verts[g]) for g in lverts)
            keyed = []
            for t in idxs:
                a, b, c = (lmap[i] for i in tris[t])
                keyed.append(flags[t] << 15 | c << 10 | b << 5 | a)
            tblob = b''.join(struct.pack('<I', k) for k in keyed)
            return {'bbox': pack_bbox(bb), 'verts': vblob, 'tris': tblob}
        ax = max(range(3), key=lambda k: bb[k+3] - bb[k])
        order = sorted(idxs, key=lambda t: centroid(t)[ax])
        h = len(order) // 2
        return {'bbox': pack_bbox(bb), 'kids': [make(order[:h]), make(order[h:])]}
    def arena(n):
        if 'kids' in n: return 0x70 + arena(n['kids'][0]) + arena(n['kids'][1])
        return 0x18 + len(n['verts']) + len(n['tris'])
    root = make(list(range(len(tris))))
    return root, 0x38 + arena(root)

def name48(s):
    b = s.encode('latin1')[:47] + b'\0'
    return b + b'\xcd' * (48 - len(b))

def is_coll_group(g):
    gl = g.lower()
    return 'collision' in gl or gl.startswith(('coll_', 'ucx_'))

def check_material_masks(matnames):
    """A material whose geometry mask lacks bit 0x1 loads as NULL and takes the
    whole level prepare down with it. Silent unless an .mtb corpus is present."""
    root = os.environ.get('SMB_MTB_DIR', os.path.join('out', 'mtb', 'materials'))
    for name in matnames:
        fn = os.path.join(root, name.replace('\\', '/') + '.mtb')
        if not os.path.exists(fn): continue
        with open(fn, 'rb') as fh:
            mat = read_material(R(fh.read()))
        fc = struct.unpack('<I', mat['fc'])[0]
        if not (fc & 1):
            raise ValueError(
                'material %r: geometry mask %#x lacks bit 0x1 (model geometry) - '
                'the game will fail the level with materialPal==NULL. Pick a '
                'material a shipped model references.' % (name, fc))

def build_new(a):
    vs, vns, vts, groups = load_obj(a.infile)
    mesh_groups = [(g, mtl, f) for g, mtl, f in groups
                   if f and not is_coll_group(g)]
    coll_groups = [(g, f) for g, mtl, f in groups if f and is_coll_group(g)]
    if not mesh_groups: raise ValueError('no visual geometry groups in OBJ')
    matnames = []
    for _, mtl, _ in mesh_groups:
        name = a.material or (mtl.replace('/', '\\') if mtl else None)
        if not name: raise ValueError('no usemtl in OBJ and no --material given')
        if name not in matnames: matnames.append(name)
    m = {'hash': b'\0' * 16, 'deps': [], 'deppad': b'\0' * ((-1) % 4 or 3),
         'nposes': 1, 'lod': 30.0, 'materials': [], 'collisions': [],
         'auxnames': [], 'parts': [], 'parents': None, 'poses': None}
    m['deppad'] = b'\0' * 3
    check_material_masks(matnames)
    for name in matnames:
        m['materials'].append({'ref': name.encode('latin1'),
                               'refpad': b'\0' * ((-(len(name) + 1)) % 4), 'embedded': None})
    allpts = []
    for gi, (g, mtl, faces) in enumerate(mesh_groups):
        pool = {}; verts = []; tris = []
        for tri in faces:
            out = []
            for vi, ti, ni in tri:
                key = (vi, ti, ni)
                if key not in pool:
                    pos = vs[vi]
                    uv = vts[ti] if ti is not None and ti < len(vts) else (0.0, 0.0)
                    nrm = vns[ni] if ni is not None and ni < len(vns) else None
                    pool[key] = len(verts); verts.append([pos, nrm, uv])
                out.append(pool[key])
            tris.append(tuple(out))
        for v in verts:                    # fill missing normals
            if v[1] is None: v[1] = (0.0, 1.0, 0.0)
        if len(verts) > 65535: raise ValueError('part %s: >65535 vertices' % g)
        verts = [tuple(v) for v in verts]
        tb = compute_tangents(verts, tris)
        vd = bytearray()
        for i, (pos, nrm, uv) in enumerate(verts):
            t, b = tb[i]
            vd += pack_geo(pos, nrm, uv)
            vd += struct.pack('<3f3f2HI', *t, *b,
                              f32_to_half(uv[0]), f32_to_half(uv[1]), 0xFFFFFFFF)
        idata = b''.join(struct.pack('<3H', *t) for t in tris)
        name = a.material or (mtl.replace('/', '\\') if mtl else '')
        bb = bbox_of([v[0] for v in verts]); allpts += [bb[:3], bb[3:]]
        m['parts'].append({'name': name48(san(g)), 'mat_idx': matnames.index(name),
                           'bbox': pack_bbox(bb), 'datasize': len(vd) + len(idata),
                           'decl': list(DECL60),
                           'nverts': len(verts), 'nprims': len(tris), 'f18': 0,
                           'morphs': [], 'vdata': bytes(vd), 'idata': idata, 'mdata': []})
    if not coll_groups and a.auto_collision:
        bb = bbox_of(allpts)
        cv = [(bb[x*3], bb[y*3+1], bb[z*3+2]) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
        ct = [(0,1,3),(3,2,0),(4,6,7),(7,5,4),(0,4,5),(5,1,0),(2,3,7),(7,6,2),(0,2,6),(6,4,0),(1,5,7),(7,3,1)]
        coll_groups = [('Intact.Collision_Box01', None), ]
        auto = (cv, ct)
    else:
        auto = None
    for gi, (g, faces) in enumerate(coll_groups):
        if auto is not None:
            cverts, ctris = auto
        else:
            pool = {}; cverts = []; ctris = []
            for tri in faces:
                out = []
                for vi, _, _ in tri:
                    if vi not in pool:
                        pool[vi] = len(cverts); cverts.append(vs[vi])
                    out.append(pool[vi])
                ctris.append(tuple(out))
        if len(cverts) > 65535: raise ValueError('collision %s: >65535 verts' % g)
        flags = [0x3B] * len(ctris)
        root, arena = build_bvt(cverts, ctris, flags)
        cname = san(g)
        for pre in ('collision_', 'ucx_'):
            if cname.lower().startswith(pre): cname = cname[len(pre):]
        if '.' not in cname: cname = 'Intact.' + cname
        m['collisions'].append({
            'name': name48(cname), 'flag': 0,
            'verts': b''.join(struct.pack('<3f', *v) for v in cverts),
            'tris': b''.join(struct.pack('<3H', *t) for t in ctris),
            'triflags': bytes(flags), 'bvt': root, 'bvt_arena': arena})
        bb = bbox_of(cverts); allpts += [bb[:3], bb[3:]]
    m['bbox'] = pack_bbox(bbox_of(allpts))
    # header pad: build once to find header size, then set pad
    m['hdrpad'] = b''
    probe = build(m)
    datalen = sum(len(p['vdata']) + len(p['idata']) for p in m['parts'])
    m['hdrpad'] = b'\0' * ((-(len(probe) - datalen)) % 16)
    return m

def cmd_from_obj(a):
    try:
        m = rebuild_exact(a) if a.sidecar else build_new(a)
    except ValueError as e:
        sys.exit(str(e))
    out = build(m)
    parse(out)                      # self-check
    with open(a.outfile, 'wb') as fh:
        fh.write(out)
    print('wrote %s (%d bytes, %d parts, %d collision, %d materials)' %
          (a.outfile, len(out), len(m['parts']), len(m['collisions']), len(m['materials'])))


def cmd_mtb_info(a):
    d = _read(a.infile)
    r = R(d)
    m = read_material(r)
    if r.o != len(d):
        raise ValueError('trailing bytes: %#x != %#x' % (r.o, len(d)))
    import struct as _s
    print('%s: %d bytes, %d layer(s), %d 2d-transform(s), %d blend rec(s), %d VS + %d PS, flags %#x'
          % (a.infile, len(d), len(m['layers']), len(m['t2d']), len(m['blend']),
             len(m['shaders']['vs']), len(m['shaders']['ps']), _s.unpack('<I', m['f10'])[0]))
    for dep, h in m['deps']:
        print('  dep: %s' % dep.decode('latin1'))
    types = {0: 'diffuse', 1: 'bump', 6: 'specular', 10: 'glow'}
    for l in m['layers']:
        t = _s.unpack('<I', l['fa8'])[0]
        print('  layer type %-2d %-10s %s' % (t, types.get(t, '?'), nm(l['name'])))

def cmd_mtb_roundtrip_test(a):
    files = sorted(glob.glob(os.path.join(a.dir, '**', '*.mtb'), recursive=True))
    ok = 0; fails = []
    for fn in files:
        d = _read(fn)
        try:
            r = R(d)
            m = read_material(r)
            if r.o != len(d):
                raise ValueError('trailing bytes')
            w = W(); write_material(w, m)
            if w.data() == d: ok += 1
            else: fails.append((fn, 'mismatch'))
        except Exception as e:
            fails.append((fn, str(e)[:80]))
    print('%d/%d .mtb byte-identical round-trip' % (ok, len(files)))
    for fn, e in fails[:10]: print('  FAIL %s: %s' % (fn, e))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('info'); p.add_argument('infile'); p.set_defaults(fn=cmd_info)

    p = sub.add_parser('verify'); p.add_argument('infiles', nargs='+'); p.set_defaults(fn=cmd_verify)

    p = sub.add_parser('roundtrip-test'); p.add_argument('dir'); p.set_defaults(fn=cmd_roundtrip_test)

    p = sub.add_parser('mtb-info'); p.add_argument('infile'); p.set_defaults(fn=cmd_mtb_info)

    p = sub.add_parser('mtb-roundtrip-test'); p.add_argument('dir'); p.set_defaults(fn=cmd_mtb_roundtrip_test)

    p = sub.add_parser('to-obj'); p.add_argument('infile'); p.add_argument('outfile')
    p.set_defaults(fn=cmd_to_obj)

    p = sub.add_parser('from-obj'); p.add_argument('infile'); p.add_argument('outfile')
    p.add_argument('--sidecar', help='.smb.json from to-obj: exact rebuild (geometry from OBJ)')
    p.add_argument('--material', help='game material ref (materials\\... path) for all parts')
    p.add_argument('--auto-collision', action='store_true',
                   help='no collision groups in OBJ: emit a bounding-box collision mesh')
    p.set_defaults(fn=cmd_from_obj)

    a = ap.parse_args()
    rc = a.fn(a)
    sys.exit(rc or 0)


if __name__ == '__main__':
    main()

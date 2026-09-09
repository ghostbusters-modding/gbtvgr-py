"""`.mtb` material tables -- the records .smb models and .bst sets refer into."""
import os
import struct

from ..wire import R, read_deps, want


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

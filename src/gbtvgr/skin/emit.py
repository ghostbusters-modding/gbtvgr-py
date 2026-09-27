"""Write the re-posed skin into a copy of the shipped mesh, and the material and textures it uses."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import io, struct
import numpy as np

from .. import tex
from ..mesh import bfm
from ..mesh.mtb import read_material, write_material
from ..wire import R, W, f32_to_half, half_to_f32

LAYER_KIND = {0: 'diff', 1: 'bump', 6: 'spec', 12: 'aocc'}


def weights4(gbj, w):
    acc = {}
    for j, x in zip(gbj, w):
        if x > 0: acc[j] = acc.get(j, 0) + x
    top = sorted(acc.items(), key=lambda kv: -kv[1])[:4]
    tot = sum(x for _, x in top)
    top = [(j, x / tot) for j, x in top]
    hs = [half_to_f32(f32_to_half(x)) for _, x in top[:-1]]
    hs.append(1.0 - sum(hs))   # the shipped weights sum to exactly one
    js = [j for j, _ in top]
    while len(js) < 4: js.append(js[0]); hs.append(0.0)
    return js, hs


def derived_parts(rig):
    """bone name -> body part, from which bones dominate each shipped part_* packet."""
    m = rig.m; counts = {}
    for p in m['packets'][0]:
        part = bfm.nm(m['parts'][p['parts'][0]]['name'])
        if not part.startswith('part_'): continue
        v = bfm.decode_vertices(p); pal = m['lists'][p['list']]
        for bi, w in zip(v['bones'], v['weights']):
            b = rig.bones[pal[bi[int(np.argmax(w))]]]['name']
            counts.setdefault(b, {}).setdefault(part, 0)
            counts[b][part] += 1
    return {b: max(c, key=c.get) for b, c in counts.items()}


def part_finder(rig, table):
    pidx = {bfm.nm(p['name']): i for i, p in enumerate(rig.m['parts'])}
    table = table or derived_parts(rig)
    def part_of(j):
        while rig.bones[j]['name'] not in table: j = rig.bones[j]['parent']
        return table[rig.bones[j]['name']]
    return pidx, part_of


def compact_materials(m):
    """Drop materials no packet uses any more; shipped meshes never go past eleven."""
    used = sorted({p['mat'] for p in m['packets'][0]})
    remap = {old: new for new, old in enumerate(used)}
    m['materials'] = [m['materials'][i] for i in used]
    for p in m['packets'][0]: p['mat'] = remap[p['mat']]


def build_mesh(rig, posed, profile, mats, compact=False):
    """mats: source material -> (material ref, uv function of the primitive's uvs)."""
    m = rig.m
    pidx, part_of = part_finder(rig, profile['parts'])
    V, T, Tpart, Tmat = [], [], [], []
    for pr in posed:
        base = len(V); ref, uvf = mats[pr['mat']]
        uv = uvf(pr['uv'])
        for i in range(len(pr['P'])):
            js, ws = weights4(pr['gbj'][i], pr['weights'][i])
            V.append((pr['P'][i], pr['N'][i], uv[i], js, ws))
        for tri in pr['tris']:
            votes = {}
            for vi in tri:
                pn = part_of(V[base + vi][3][0])
                votes[pn] = votes.get(pn, 0) + 1
            T.append([base + vi for vi in tri]); Tpart.append(max(votes, key=votes.get)); Tmat.append(ref)
    T = np.array(T)
    # shipped triangles are counter-clockwise about their normals
    Pv = np.array([v[0] for v in V]); Nv = np.array([v[1] for v in V])
    fn = np.cross(Pv[T[:, 1]] - Pv[T[:, 0]], Pv[T[:, 2]] - Pv[T[:, 0]])
    flip = ((fn * Nv[T].sum(1)).sum(1) < 0).mean() > 0.5
    if flip: T = T[:, ::-1]

    used = sorted({j for v in V for j, w in zip(v[3], v[4]) if w > 0})
    if len(used) > bfm.MAX_PALETTE:
        raise ValueError('%d bones in use; one palette holds %d' % (len(used), bfm.MAX_PALETTE))
    lp = {j: k for k, j in enumerate(used)}
    list_index = len(m['lists']); m['lists'].append(used)
    mat_index = {}
    for ref in dict.fromkeys(Tmat):
        m['materials'].append({'ref': ref.encode(), 'refpad': b'\0' * ((-(len(ref) + 1)) % 4), 'embedded': None})
        mat_index[ref] = len(m['materials']) - 1

    keep = [p for p in m['packets'][0]
            if bfm.nm(m['parts'][p['parts'][0]]['name']).startswith(profile['keep'])
            and not bfm.nm(m['parts'][p['parts'][0]]['name']).startswith(profile['drop'])]
    new = []
    groups = sorted({(pidx[pt], mt) for pt, mt in zip(Tpart, Tmat)}, key=lambda k: (k[0], mat_index[k[1]]))
    for part_i, ref in groups:
        tris = T[[i for i, (pt, mt) in enumerate(zip(Tpart, Tmat)) if pidx[pt] == part_i and mt == ref]]
        vids, inv = np.unique(tris.ravel(), return_inverse=True)
        nrm = [tuple(map(float, V[i][1])) for i in vids]
        tan, bin_ = [], []
        for n in nrm:
            n = np.array(n); a = np.array([0, 1, 0]) if abs(n[1]) < 0.9 else np.array([1, 0, 0])
            tt = np.cross(a, n); tt /= np.linalg.norm(tt); bb = np.cross(n, tt)
            tan.append(tuple(tt)); bin_.append(tuple(bb))
        verts = {'pos': [tuple(map(float, V[i][0])) for i in vids], 'normal': nrm,
                 'uv0': [tuple(map(float, V[i][2])) for i in vids], 'tangent': tan, 'binormal': bin_,
                 'bones': [tuple(lp[j] for j in V[i][3]) for i in vids],
                 'weights': [tuple(V[i][4]) for i in vids]}
        p = {'mat': mat_index[ref], 'list': list_index, 'parts': [part_i], 'extra': [], 'morphs': [], 'mdata': []}
        bfm.set_geometry(m, p, verts, [tuple(map(int, t)) for t in inv.reshape(-1, 3)], bfm.DECL_A)
        new.append(p)
    m['packets'][0] = new + keep
    if compact: compact_materials(m)

    # bounds of unknown frame: grow them about the bone so the new shape is not culled
    for i in {p['parts'][0] for p in new}:
        m['parts'][i]['bbox'] = (np.frombuffer(m['parts'][i]['bbox'], '<f4') * 1.5).astype('<f4').tobytes()
    B = np.frombuffer(m['bone_b'], '<f4').reshape(-1, 6).copy()
    for j in used: B[j] *= 1.5
    m['bone_b'] = B.astype('<f4').tobytes()
    out = bfm.build(m)
    if bfm.build(bfm.parse(out)) != out:
        raise AssertionError('rebuilt mesh does not round-trip')
    return out, dict(flip=bool(flip), verts=sum(p['nverts'] for p in new), tris=sum(p['ntris'] for p in new),
                     packets=len(new), kept=len(keep), palette=len(used))


# --- materials ------------------------------------------------------------------
def material(template, layer_dir, name):
    r = R(template); mt = read_material(r)
    if r.o != len(template): raise ValueError('template material has trailing bytes')
    for l in mt['layers']:
        kind = LAYER_KIND[struct.unpack('<I', l['fa8'])[0]]
        path = ('%s\\%s_%s.tga' % (layer_dir, name, kind)).encode()
        l['name'] = path + b'\0' + l['name'][len(path) + 1:]
    w = W(); write_material(w, mt)
    return w.data()


def _bc1(rgb):
    r, g, b = rgb
    c = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    return struct.pack('<HHI', c, c, 0)


def swatch_tex(colors, fmt, size=64):
    if fmt == 47:   # RG8 bump: flat
        return tex.build_header(7, b'\0' * 16, 0, 47, size, size, 0, 1, 0, 0) + bytes([128, 128]) * (size * size)
    n = len(colors); blocks = bytearray()
    for by in range(size // 4):
        for bx in range(size // 4):
            c = _bc1(colors[bx * 4 * n // size])
            blocks += (bytes([255, 255, 0, 0, 0, 0, 0, 0]) + c) if fmt == 50 else c
    return tex.build_header(7, b'\0' * 16, 0, fmt, size, size, 0, 1, 0, 0) + bytes(blocks)


def image_tex(data, max_size=1024):
    """Any image Pillow reads -> a DXT5 .tex, resized to power-of-two sides."""
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert('RGBA')
    pw = lambda v: min(max_size, 1 << max(2, (v - 1).bit_length()))
    if im.size != (pw(im.width), pw(im.height)):
        im = im.resize((pw(im.width), pw(im.height)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, 'DDS', pixel_format='DXT5')
    return tex.from_dds(b.getvalue())

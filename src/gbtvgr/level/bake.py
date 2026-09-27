"""Baking light into the set: per-vertex colours (the in-game proven path),
lightmap tiles, and probe cubemaps.

The lighting model is the HarborDocks one: ambient times an occlusion term,
plus a lambert pool per light with a smooth radius falloff, tone-mapped on the
peak channel so hue survives, then desaturated toward luminance.
"""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import math

import numpy as np

from . import textures
from .mathutil import F32, face_normals, transform_normals, transform_points

DEFAULTS = {
    'ambient': 0.25, 'tint': (1.0, 1.0, 1.0), 'tint_strength': 0.34, 'contrast': 2.0,
    'sat': 0.42, 'peak_max': 0.85,
    'ao': True, 'ao_rays': 8, 'ao_distance': 12.0, 'ao_strength': 0.7,
    'texels_per_unit': 2.0, 'tile_size': 256, 'lightmaps': False,
}
PAD = 2                       # texels of chart padding on every side
NEUTRAL = 128                 # the lightmap shader doubles what it samples


def settings_of(doc, overrides=None):
    s = dict(DEFAULTS)
    s.update(doc.settings.get('bake', {}) if doc is not None else {})
    s.update(overrides or {})
    return s


class LightSample:
    __slots__ = ('pos', 'color', 'radius', 'gain')

    def __init__(self, pos, color, radius, gain=1.0):
        self.pos = np.asarray(pos, np.float64)
        self.color = np.asarray(color, np.float64)
        self.radius = float(radius)
        self.gain = float(gain)

    @classmethod
    def from_node(cls, node):
        p = node.props
        color = [float(c) / 255.0 for c in p.get('color', (255, 255, 255))]
        return cls(node.world_pos(), color, p.get('radius', 30.0), p.get('gain', 1.0))


def lights_of(doc):
    return [LightSample.from_node(n) for n in doc.nodes('light')]


# -- the lighting model -----------------------------------------------------------------
def light_points(pos, nrm, lights, settings, ao=None):
    """(K, 3) positions and normals -> (K, 3) linear colour in 0..1."""
    pos = np.asarray(pos, np.float64).reshape(-1, 3)
    nrm = np.asarray(nrm, np.float64).reshape(-1, 3)
    k = len(pos)
    amb = float(settings['ambient']) * (np.ones(k) if ao is None else np.asarray(ao, np.float64))
    tint = np.asarray(settings['tint'], np.float64)
    tint = 1.0 + (tint - 1.0) * float(settings['tint_strength'])
    rgb = amb[:, None] * tint[None, :]
    contrast = float(settings['contrast'])
    for L in lights:
        d = L.pos[None, :] - pos
        d2 = np.einsum('ij,ij->i', d, d)
        inside = d2 < L.radius * L.radius
        if not inside.any():
            continue
        dist = np.sqrt(np.maximum(d2, 1e-8))
        lam = np.einsum('ij,ij->i', d, nrm) / dist
        ok = inside & (lam > 0.0)
        f = 1.0 - dist / L.radius
        atten = lam * f * f * contrast * L.gain
        rgb[ok] += atten[ok, None] * L.color[None, :]
    m = np.maximum(rgb.max(axis=1), 1e-6)
    t = np.minimum(float(settings['peak_max']), np.sqrt(m / (0.85 + m)))
    rgb *= (t / m)[:, None]
    y = 0.30 * rgb[:, 0] + 0.59 * rgb[:, 1] + 0.11 * rgb[:, 2]
    rgb = y[:, None] + (rgb - y[:, None]) * float(settings['sat'])
    return np.clip(rgb, 0.0, 1.0)


def to_rgba8(rgb, scale=1.0):
    out = np.empty((len(rgb), 4), np.uint8)
    out[:, :3] = np.clip(np.round(np.asarray(rgb) * 255.0 * scale), 0, 255).astype(np.uint8)
    out[:, 3] = 255
    return out


# -- ambient occlusion --------------------------------------------------------------------
def hemisphere_dirs(n):
    """n cosine-weighted directions about +Z, stratified on a golden spiral."""
    i = np.arange(n, dtype=np.float64)
    u = (i + 0.5) / n
    phi = 2.0 * math.pi * ((i * 0.6180339887) % 1.0)
    r = np.sqrt(u)
    return np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(np.maximum(0.0, 1.0 - u))], axis=1)


def _frames(nrm):
    n = np.asarray(nrm, np.float64)
    helper = np.where(np.abs(n[:, 1:2]) < 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
    t = np.cross(helper, n)
    ln = np.linalg.norm(t, axis=1, keepdims=True)
    ln[ln < 1e-9] = 1.0
    t /= ln
    b = np.cross(n, t)
    return t, b


class TriGrid3:
    """Triangles bucketed in a 3D grid; a query gets everything within one cell."""

    def __init__(self, verts, tris, cell):
        self.v = np.asarray(verts, np.float64).reshape(-1, 3)
        self.t = np.asarray(tris, np.int64).reshape(-1, 3)
        self.cell = float(cell)
        p = self.v[self.t] if len(self.t) else np.zeros((0, 3, 3))
        self.lo = p.min(axis=1) if len(p) else np.zeros((0, 3))
        self.hi = p.max(axis=1) if len(p) else np.zeros((0, 3))
        self.origin = self.lo.min(axis=0) if len(p) else np.zeros(3)
        self.buckets = {}
        if len(p):
            ilo = np.floor((self.lo - self.origin) / self.cell).astype(np.int64)
            ihi = np.floor((self.hi - self.origin) / self.cell).astype(np.int64)
            for ti in range(len(p)):
                for a in range(ilo[ti, 0], ihi[ti, 0] + 1):
                    for b in range(ilo[ti, 1], ihi[ti, 1] + 1):
                        for c in range(ilo[ti, 2], ihi[ti, 2] + 1):
                            self.buckets.setdefault((a, b, c), []).append(ti)
            self.buckets = {k: np.array(v, np.int64) for k, v in self.buckets.items()}

    def cell_of(self, pts):
        return np.floor((np.asarray(pts, np.float64) - self.origin) / self.cell).astype(np.int64)

    def around(self, key):
        found = []
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    c = self.buckets.get((key[0] + da, key[1] + db, key[2] + dc))
                    if c is not None:
                        found.append(c)
        return np.unique(np.concatenate(found)) if found else np.zeros(0, np.int64)


def any_hit(origins, dirs, v0, v1, v2, tmax, chunk=2000000):
    """Moeller-Trumbore, rays x triangles, True where a ray hits within tmax."""
    r = len(origins)
    c = len(v0)
    out = np.zeros(r, bool)
    if r == 0 or c == 0:
        return out
    e1 = v1 - v0
    e2 = v2 - v0
    step = max(1, chunk // c)
    for s in range(0, r, step):
        o = origins[s:s + step][:, None, :]
        d = dirs[s:s + step][:, None, :]
        p = np.cross(d, e2[None, :, :])
        det = np.einsum('rcj,rcj->rc', np.broadcast_to(e1[None], p.shape), p)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = 1.0 / det
            tv = o - v0[None, :, :]
            u = np.einsum('rcj,rcj->rc', tv, p) * inv
            q = np.cross(tv, e1[None, :, :])
            v = np.einsum('rcj,rcj->rc', np.broadcast_to(d, q.shape), q) * inv
            t = np.einsum('rcj,rcj->rc', np.broadcast_to(e2[None], q.shape), q) * inv
            ok = (np.abs(det) > 1e-9) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) \
                & (t > 1e-4) & (t <= tmax)
        out[s:s + step] = ok.any(axis=1)
    return out


def ambient_occlusion(pos, nrm, occ_verts, occ_tris, rays=8, distance=12.0, strength=0.7):
    """(K,) occlusion factor in [1-strength, 1]: 1 is open sky."""
    pos = np.asarray(pos, np.float64).reshape(-1, 3)
    nrm = np.asarray(nrm, np.float64).reshape(-1, 3)
    k = len(pos)
    if k == 0 or len(occ_tris) == 0:
        return np.ones(k)
    dirs = hemisphere_dirs(int(rays))
    t, b = _frames(nrm)
    d = (t[:, None, :] * dirs[None, :, 0, None] + b[:, None, :] * dirs[None, :, 1, None]
         + nrm[:, None, :] * dirs[None, :, 2, None])
    origins = pos + nrm * 0.05
    grid = TriGrid3(occ_verts, occ_tris, distance)
    v0 = grid.v[grid.t[:, 0]]
    v1 = grid.v[grid.t[:, 1]]
    v2 = grid.v[grid.t[:, 2]]
    occl = np.zeros(k)
    keys = grid.cell_of(origins)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    start = 0
    n = int(rays)
    while start < k:
        key = tuple(keys[order[start]])
        end = start
        while end < k and tuple(keys[order[end]]) == key:
            end += 1
        idx = order[start:end]
        cand = grid.around(key)
        if len(cand):
            o = np.repeat(origins[idx], n, axis=0)
            dd = d[idx].reshape(-1, 3)
            hit = any_hit(o, dd, v0[cand], v1[cand], v2[cand], distance).reshape(-1, n)
            occl[idx] = hit.mean(axis=1)
        start = end
    return 1.0 - occl * float(strength)


# -- per-vertex bake ---------------------------------------------------------------------------
def _node_geometry(node):
    if node.kind == 'mesh':
        return node.mesh
    return node.terrain.mesh() if node.terrain is not None else None


def bake_vertex_colors(doc, settings=None, occluder_tris=None, progress=None):
    """Light every mesh and terrain node; colours land on the geometry as (N, 4)
    uint8 aligned with its vertices. Returns {node id: colours}."""
    s = settings_of(doc, settings)
    lights = lights_of(doc)
    if s.get('ao'):
        if occluder_tris is None:
            ov, ot, _ = doc.world_triangles(collide_only=True)
        else:
            ov, ot = occluder_tris
    nodes = doc.mesh_nodes()
    out = {}
    for k, node in enumerate(nodes):
        geo = _node_geometry(node)
        if geo is None or len(geo.faces) == 0:
            continue
        pos, nrm, _uv, _idx, cmap = geo.split_corners()[:5]
        m = node.world_matrix()
        wpos = transform_points(m, pos).astype(np.float64)
        wnrm = transform_normals(m, nrm).astype(np.float64)
        ao = None
        if s.get('ao'):
            ao = ambient_occlusion(wpos, wnrm, ov, ot, s['ao_rays'], s['ao_distance'],
                                   s['ao_strength'])
        rgb = light_points(wpos, wnrm, lights, s, ao)
        sums = np.zeros((geo.nverts, 3))
        counts = np.zeros(geo.nverts)
        np.add.at(sums, cmap, rgb)
        np.add.at(counts, cmap, 1.0)
        counts[counts == 0] = 1.0
        colors = to_rgba8(sums / counts[:, None])
        geo.colors = colors
        if node.kind == 'terrain':
            node.terrain.colors = colors
        out[node.id] = colors
        if progress:
            progress(k + 1, len(nodes))
    return out


# -- lightmaps -----------------------------------------------------------------------------------
_AXES = {0: (2, 1), 1: (0, 2), 2: (0, 1)}      # projection axis -> (u axis, v axis)


def _section_nodes(section):
    return [n for n in section.iter_descendants() if n.kind in ('mesh', 'terrain')]


def _charts_for(node):
    """Faces grouped by (dominant axis, sign) with their 2D world-unit coords."""
    geo = _node_geometry(node)
    if geo is None or len(geo.faces) == 0:
        return []
    m = node.world_matrix()
    wv = transform_points(m, geo.verts).astype(np.float64)
    corners = wv[geo.faces]                                   # (M, 3, 3)
    fn = transform_normals(m, geo.face_normals()).astype(np.float64)
    axis = np.argmax(np.abs(fn), axis=1)
    sign = np.sign(fn[np.arange(len(fn)), axis])
    charts = []
    for ax in (0, 1, 2):
        for sg in (-1.0, 1.0):
            faces = np.where((axis == ax) & (sign == sg))[0]
            if not len(faces):
                continue
            ua, va = _AXES[ax]
            uv = np.stack([corners[faces][:, :, ua], corners[faces][:, :, va]], axis=2)
            umin, vmin = uv[:, :, 0].min(), uv[:, :, 1].min()
            charts.append({'node': node, 'faces': faces, 'uv': uv, 'umin': float(umin),
                           'vmin': float(vmin), 'eu': float(uv[:, :, 0].max() - umin),
                           'ev': float(uv[:, :, 1].max() - vmin),
                           'corners': corners[faces], 'normals': fn[faces]})
    return charts


def _shelf_pack(charts, density, size):
    """Place charts on a size x size tile; None when they do not fit."""
    items = []
    for c in charts:
        w = int(math.ceil(c['eu'] * density)) + 2 * PAD + 1
        h = int(math.ceil(c['ev'] * density)) + 2 * PAD + 1
        items.append((h, w, c))
    items.sort(key=lambda it: (-it[0], -it[1]))
    x = y = shelf_h = 0
    placed = []
    for h, w, c in items:
        if w > size or h > size:
            return None
        if x + w > size:
            x = 0
            y += shelf_h
            shelf_h = 0
        if y + h > size:
            return None
        placed.append((c, x, y, w, h))
        x += w
        shelf_h = max(shelf_h, h)
    return placed


def plan_lightmaps(doc, section, tile_size=None, settings=None):
    """Charts and per-corner lightmap uvs for one section's meshes."""
    s = settings_of(doc, settings)
    max_size = int(tile_size or s['tile_size'])
    charts = []
    for n in _section_nodes(section):
        charts.extend(_charts_for(n))
    plan = {'size': (64, 64), 'charts': [], 'lmuv': {}, 'density': 0.0}
    if not charts:
        return plan
    density = float(s['texels_per_unit'])
    placed, size = None, 64
    for _attempt in range(40):
        need = sum((math.ceil(c['eu'] * density) + 2 * PAD + 1) * (math.ceil(c['ev'] * density) + 2 * PAD + 1)
                   for c in charts)
        size = 64
        while size * size < need * 1.25 and size < max_size:
            size *= 2
        placed = _shelf_pack(charts, density, size)
        if placed is not None:
            break
        density *= 0.85
    if placed is None:
        raise ValueError('lightmap charts do not fit a %d tile' % max_size)
    plan['size'] = (size, size)
    plan['density'] = density
    lmuv = {}
    for c, x, y, w, h in placed:
        node = c['node']
        geo = _node_geometry(node)
        arr = lmuv.setdefault(node.id, np.zeros((len(geo.faces), 3, 2), F32))
        u = (x + PAD + (c['uv'][:, :, 0] - c['umin']) * density) / size
        v = (y + PAD + (c['uv'][:, :, 1] - c['vmin']) * density) / size
        arr[c['faces'], :, 0] = u
        arr[c['faces'], :, 1] = v
        plan['charts'].append({'node': node.id, 'faces': c['faces'], 'x': x, 'y': y,
                               'w': w, 'h': h, 'corners': c['corners'], 'normals': c['normals'],
                               'texel': np.stack([u * size, v * size], axis=2)})
    plan['lmuv'] = lmuv
    return plan


def _rasterise(plan):
    """Every covered texel's world position and normal. (H, W) maps."""
    w, h = plan['size']
    pos = np.zeros((h, w, 3))
    nrm = np.zeros((h, w, 3))
    covered = np.zeros((h, w), bool)
    for c in plan['charts']:
        tex = c['texel']                           # (k, 3, 2)
        corners = c['corners']
        normals = c['normals']
        for f in range(len(tex)):
            t = tex[f]
            x0 = max(0, int(math.floor(t[:, 0].min())))
            x1 = min(w - 1, int(math.ceil(t[:, 0].max())))
            y0 = max(0, int(math.floor(t[:, 1].min())))
            y1 = min(h - 1, int(math.ceil(t[:, 1].max())))
            if x1 < x0 or y1 < y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            ax, ay = t[0]
            bx, by = t[1]
            cx, cy = t[2]
            d = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
            if abs(d) < 1e-12:
                continue
            w0 = ((bx - gx) * (cy - gy) - (cx - gx) * (by - gy)) / d
            w1 = ((cx - gx) * (ay - gy) - (ax - gx) * (cy - gy)) / d
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -0.02) & (w1 >= -0.02) & (w2 >= -0.02)
            if not inside.any():
                continue
            ys, xs = np.where(inside)
            p = (w0[inside, None] * corners[f, 0][None, :] + w1[inside, None] * corners[f, 1][None, :]
                 + w2[inside, None] * corners[f, 2][None, :])
            pos[ys + y0, xs + x0] = p
            nrm[ys + y0, xs + x0] = normals[f]
            covered[ys + y0, xs + x0] = True
    return pos, nrm, covered


def _dilate(img, covered, passes=2):
    out = img.copy()
    cov = covered.copy()
    for _ in range(passes):
        acc = np.zeros_like(out)
        cnt = np.zeros(cov.shape)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                sh = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
                sc = np.roll(np.roll(cov, dy, axis=0), dx, axis=1)
                acc += sh * sc[:, :, None]
                cnt += sc
        fill = (~cov) & (cnt > 0)
        out[fill] = acc[fill] / cnt[fill][:, None]
        cov = cov | fill
    return out, cov


def bake_lightmaps(doc, section, plan, settings=None, template_tile=None, progress=None):
    """Three fmt-3 tiles for the section. The three slots carry the same image:
    the directional three-basis split is not attempted here."""
    s = settings_of(doc, settings)
    w, h = plan['size']
    pos, nrm, covered = _rasterise(plan)
    if progress:
        progress(1, 3)
    img = np.full((h, w, 3), NEUTRAL / 255.0)
    if covered.any():
        lights = lights_of(doc)
        p = pos[covered]
        n = nrm[covered]
        ao = None
        if s.get('ao'):
            ov, ot, _ = doc.world_triangles(collide_only=True)
            ao = ambient_occlusion(p, n, ov, ot, s['ao_rays'], s['ao_distance'], s['ao_strength'])
        rgb = light_points(p, n, lights, s, ao)
        lit = np.zeros((h, w, 3))
        lit[covered] = rgb
        lit, cov2 = _dilate(lit, covered)
        img[cov2] = lit[cov2] * 0.5
    if progress:
        progress(2, 3)
    rgba = np.empty((h, w, 4), np.uint8)
    rgba[:, :, :3] = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    rgba[:, :, 3] = 255
    tile = textures.encode_raw(rgba, template=template_tile, fmt=textures.FMT_BGRA8)
    if progress:
        progress(3, 3)
    return {'tiles': [tile, tile, tile], 'size': (w, h), 'lmuv': plan['lmuv'], 'image': rgba}


# -- probes -------------------------------------------------------------------------------------
def _box_blur(face, radius):
    f = face.astype(np.float64)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        p = np.pad(f, pad, mode='edge')
        c = np.cumsum(p, axis=axis)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
        n = f.shape[axis]
        hi = np.take(c, np.arange(2 * radius + 1, 2 * radius + 1 + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        f = (hi - lo) / (2 * radius + 1)
    return f


def ambient_from_env(faces, size=16):
    """Each 64x64 face blurred wide and box-downsampled to the ambient cube."""
    out = []
    for face in faces:
        f = face.astype(np.float64)
        for _ in range(3):
            f = _box_blur(f, 8)
        n = f.shape[0] // size
        small = f.reshape(size, n, size, n, 4).mean(axis=(1, 3))
        small[:, :, 3] = 255
        out.append(np.clip(np.round(small), 0, 255).astype(np.uint8))
    return out


def bake_probes(probes, cubemap_fn, template_ambient=None, template_env=None, size=64):
    """[(ambient .tex bytes, env .tex bytes)] per probe node."""
    out = []
    for p in probes:
        pos = p.world_pos() if hasattr(p, 'world_pos') else np.asarray(p, np.float64)
        faces = [np.asarray(f, np.uint8) for f in cubemap_fn(pos, size)]
        env = textures.encode_raw(faces, template=template_env, fmt=textures.FMT_CUBE)
        amb = textures.encode_raw(ambient_from_env(faces), template=template_ambient,
                                  fmt=textures.FMT_CUBE)
        out.append((amb, env))
    return out


def donor_templates(library):
    """Header templates out of the archives: the 16-byte hash is unresolved, so a
    shipped tile's is copied the way make_lightmap.py does."""
    out = {'tile': None, 'ambient': None, 'env': None}
    for n in library.names('lightmaps'):
        blob = library.read(n)
        if blob and textures.header_desc(blob)[0] == textures.FMT_BGRA8:
            out['tile'] = blob
            break
    for n in library.names('lightprobes'):
        base = n.split('\\')[-1].lower()
        if out['ambient'] is None and base.startswith('ambient_'):
            out['ambient'] = library.read(n)
        elif out['env'] is None and base.startswith('env_'):
            out['env'] = library.read(n)
        if out['ambient'] is not None and out['env'] is not None:
            break
    return out

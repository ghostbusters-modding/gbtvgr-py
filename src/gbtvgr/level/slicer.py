"""Keep only what a section shows from a view box: a facing and ray verdict per triangle,
the kept triangles re-encoded through geom, the BVT rebuilt over them."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import argparse
import collections
import copy
import os
import struct
import sys

import numpy as np

from ..mesh.bvt import bvt_build, bvt_collect, union_bbox
from ..sets import bst, geom
from . import host
from . import transplant as tp

VERDICTS = ('seen', 'occluded', 'backface-only', 'unseen')
SEEN, OCCLUDED, BACKFACE, UNSEEN = range(4)
EYE_HEIGHT = 5.5
EYES, SAMPLES = 4, 4      # a coarser grid or centroid-only rays call visible triangles occluded
INLINE = 0x20000          # mesh flag: no render packet of its own, nothing to slice
MAX_VERTS = 65535
# Extra ray targets inside a triangle, barycentric: the centroid, then a point near each corner.
SAMPLE_BARY = ((1 / 3.0, 1 / 3.0, 1 / 3.0), (2 / 3.0, 1 / 6.0, 1 / 6.0),
               (1 / 6.0, 2 / 3.0, 1 / 6.0), (1 / 6.0, 1 / 6.0, 2 / 3.0))


class SliceError(Exception):
    pass


# --- section lookup and the view box ---------------------------------------------------
def section_index(m, section):
    """An index or a section name -> the index. A repeated name is refused."""
    if isinstance(section, (int, np.integer)):
        if not 0 <= section < len(m['sections']):
            raise SliceError('section %d outside %d sections' % (section, len(m['sections'])))
        return int(section)
    hits = [i for i, s in enumerate(m['sections']) if geom.name_str(s['name']) == section]
    if len(hits) != 1:
        raise SliceError('section %r: %d matches' % (section, len(hits)))
    return hits[0]


def view_box(box):
    b = [float(v) for v in box]
    if len(b) != 6:
        raise SliceError('a view box is x0 y0 z0 x1 y1 z1')
    lo = tuple(min(b[k], b[k + 3]) for k in range(3))
    return lo + tuple(max(b[k], b[k + 3]) for k in range(3))


def eye_points(box, eyes=EYES, height=EYE_HEIGHT):
    """eyes x eyes grid over the box floor at cell centres, each `height` up (one or several),
    never above the box top. A list of points passes through as the eyes."""
    if not isinstance(eyes, int):
        return np.asarray(eyes, np.float64).reshape(-1, 3)
    if eyes < 1:
        raise SliceError('eyes must be >= 1')
    box = view_box(box)
    heights = (float(height),) if np.isscalar(height) else tuple(height)
    ys = sorted(set(min(box[1] + h, box[4]) for h in heights))
    fr = (np.arange(eyes) + 0.5) / eyes
    xs = box[0] + fr * (box[3] - box[0])
    zs = box[2] + fr * (box[5] - box[2])
    return np.array([(x, y, z) for y in ys for x in xs for z in zs], np.float64)


# --- triangle soups --------------------------------------------------------------------
def mesh_arrays(me):
    """(positions (n, 3), triangles (t, 3)) of one inline mesh, or None."""
    if me['flags'] & INLINE or not me['pkt']['nverts'] or not me['pkt']['nprims']:
        return None
    p = me['pkt']
    st = len(p['vdata']) // p['nverts']
    pos = np.frombuffer(p['vdata'], np.uint8).reshape(p['nverts'], st)[:, :12]
    pos = np.ascontiguousarray(pos).view('<f4').astype(np.float64)
    tris = np.frombuffer(p['idata'], '<u2').reshape(-1, 3).astype(np.int64)
    return pos, tris


def section_soup(m, section):
    """Every inline triangle of a section as one soup: (V, T, mesh index per tri)."""
    idx = section_index(m, section)
    vs, ts, owner = [], [], []
    base = 0
    for mi, me in enumerate(m['sections'][idx]['meshes']):
        arr = mesh_arrays(me)
        if arr is None:
            continue
        pos, tris = arr
        vs.append(pos)
        ts.append(tris + base)
        owner.append(np.full(len(tris), mi, np.int64))
        base += len(pos)
    if not vs:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
    return np.concatenate(vs), np.concatenate(ts), np.concatenate(owner)


def _cross(u, v):
    """np.cross spends most of its time on dispatch at these shapes."""
    return np.stack((u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1], u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2],
                     u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]), axis=1)


def _tri_frames(V, T):
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = _cross(b - a, c - a)
    return a, b, c, n, np.linalg.norm(n, axis=1)


# --- ray occlusion over a uniform grid --------------------------------------------------
class Occluders:
    """A triangle soup bucketed on a uniform grid, walked per ray with a 3D DDA.
    Front faces block by default, as the renderer culls the backs."""

    def __init__(self, V, T, cap=128, density=0.25):
        self.V = np.asarray(V, np.float64)
        self.T = np.asarray(T, np.int64).reshape(-1, 3)
        self.a, b, c, self.n, ln = _tri_frames(self.V, self.T)
        self.e1, self.e2 = b - self.a, c - self.a
        k = len(self.T)
        if k == 0:
            self.res = np.ones(3, np.int64)
            self.gmin, self.gmax = np.zeros(3), np.ones(3)
            self.csize = np.ones(3)
            self.cell_start = np.zeros(2, np.int64)
            self.cell_tris = np.zeros(0, np.int64)
            return
        lo = np.minimum(np.minimum(self.a, b), c)
        hi = np.maximum(np.maximum(self.a, b), c)
        self.gmin, self.gmax = lo.min(axis=0) - 1e-3, hi.max(axis=0) + 1e-3
        ext = np.maximum(self.gmax - self.gmin, 1e-6)
        base = (k / density) ** (1 / 3.0)
        scaled = np.round(base * ext / np.exp(np.log(ext).mean()))
        self.res = np.clip(scaled, 1, cap).astype(np.int64)
        self.csize = ext / self.res
        ilo = np.clip(np.floor((lo - self.gmin) / self.csize), 0, self.res - 1).astype(np.int64)
        ihi = np.clip(np.floor((hi - self.gmin) / self.csize), 0, self.res - 1).astype(np.int64)
        span = ihi - ilo + 1
        cnt = span.prod(axis=1)
        rep = np.repeat(np.arange(k), cnt)
        offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        sx, sy = span[rep, 0], span[rep, 1]
        ix = ilo[rep, 0] + offs % sx
        iy = ilo[rep, 1] + (offs // sx) % sy
        iz = ilo[rep, 2] + offs // (sx * sy)
        cid = self._cid(ix, iy, iz)
        order = np.argsort(cid, kind='stable')
        self.cell_tris = rep[order]
        ncell = int(self.res.prod())
        self.cell_start = np.searchsorted(cid[order], np.arange(ncell + 1))

    def _cid(self, ix, iy, iz):
        return (ix * self.res[1] + iy) * self.res[2] + iz

    def _hits(self, O, D, tri, backs_block):
        a, e1, e2 = self.a[tri], self.e1[tri], self.e2[tri]
        p = _cross(D, e2)
        det = np.einsum('ij,ij->i', e1, p)
        ok = np.abs(det) > 1e-12 if backs_block else det > 1e-12
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = 1.0 / det
            tv = O - a
            u = np.einsum('ij,ij->i', tv, p) * inv
            q = _cross(tv, e1)
            v = np.einsum('ij,ij->i', D, q) * inv
            s = np.einsum('ij,ij->i', e2, q) * inv
            ok &= (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
        return ok, s

    def blocked(self, O, D, exclude=None, skin=0.1, backs_block=False, chunk=16384):
        """Per segment O + s*D, s in (0, 1): does a triangle block it? A hit within `skin`
        of either end does not count, so a decal on the target is not an occluder."""
        O = np.asarray(O, np.float64).reshape(-1, 3)
        D = np.asarray(D, np.float64).reshape(-1, 3)
        r = len(O)
        out = np.zeros(r, bool)
        if r == 0 or len(self.T) == 0:
            return out
        exclude = np.full(r, -1, np.int64) if exclude is None else np.asarray(exclude, np.int64)
        for i in range(0, r, chunk):
            j = min(r, i + chunk)
            out[i:j] = self._blocked(O[i:j], D[i:j], exclude[i:j], skin, backs_block)
        return out

    def _blocked(self, O, D, exclude, skin, backs_block):
        r = len(O)
        out = np.zeros(r, bool)
        L = np.linalg.norm(D, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = 1.0 / D
            t0 = (self.gmin - O) * inv
            t1 = (self.gmax - O) * inv
        smin = np.nanmax(np.minimum(t0, t1), axis=1)
        smax = np.nanmin(np.maximum(t0, t1), axis=1)
        s0 = np.maximum(smin, 0.0)
        s1 = np.minimum(smax, 1.0)
        active = np.where((L > 1e-9) & (s0 <= s1))[0]
        if not len(active):
            return out
        P = O[active] + s0[active, None] * D[active]
        cell = np.clip(np.floor((P - self.gmin) / self.csize), 0, self.res - 1).astype(np.int64)
        Da = D[active]
        step = np.sign(Da).astype(np.int64)
        with np.errstate(divide='ignore', invalid='ignore'):
            nextb = self.gmin + (cell + (step > 0)) * self.csize
            tmax = np.where(Da != 0, (nextb - O[active]) / Da, np.inf)
            tdelta = np.where(Da != 0, self.csize / np.abs(Da), np.inf)
        send = s1[active]
        lo_s, hi_s = skin / L[active], 1.0 - skin / L[active]
        while len(active):
            cid = self._cid(cell[:, 0], cell[:, 1], cell[:, 2])
            start = self.cell_start[cid]
            cnt = self.cell_start[cid + 1] - start
            tot = int(cnt.sum())
            hit_rays = np.zeros(len(active), bool)
            if tot:
                rep = np.repeat(np.arange(len(active)), cnt)
                offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                tri = self.cell_tris[np.repeat(start, cnt) + offs]
                ok, s = self._hits(O[active][rep], Da[rep], tri, backs_block)
                ok &= (tri != exclude[active][rep]) & (s > lo_s[rep]) & (s < hi_s[rep])
                hit_rays[np.unique(rep[ok])] = True
            out[active[hit_rays]] = True
            ax = np.argmin(tmax, axis=1)
            rows = np.arange(len(active))
            done = hit_rays | (tmax[rows, ax] >= send)
            cell[rows, ax] += step[rows, ax]
            done |= (cell[rows, ax] < 0) | (cell[rows, ax] >= self.res[ax])
            tmax[rows, ax] += tdelta[rows, ax]
            keep = ~done
            active, cell, Da, step, tmax, tdelta = (active[keep], cell[keep], Da[keep], step[keep],
                                                    tmax[keep], tdelta[keep])
            send, lo_s, hi_s = send[keep], lo_s[keep], hi_s[keep]
        return out


# --- the report ---------------------------------------------------------------------------
def material_label(m, me):
    e = m['materials'][tp._h70(me)]
    ref = geom.name_str(e['ref'][0])
    return ref.replace('/', '\\') if ref else 'embedded'


def _occluder_soup(m, section, others, region=None):
    """The section's own triangles first (so `exclude` indexes them), then the others' cut to
    `region` (lo, hi); an other is a section of m, or (set, section) from another set."""
    V, T, _own = section_soup(m, section)
    vs, ts = [V], [T]
    base = len(V)
    for o in others:
        om, osec = o if isinstance(o, tuple) else (m, o)
        ov, ot, _x = section_soup(om, osec)
        if region is not None and len(ot):
            p = ov[ot]
            inside = np.all(p.max(axis=1) >= region[0], axis=1)
            ot = ot[inside & np.all(p.min(axis=1) <= region[1], axis=1)]
        vs.append(ov)
        ts.append(ot + base)
        base += len(ov)
    return np.concatenate(vs), np.concatenate(ts)


def tri_verdicts(V, T, box, eyes, occ, samples=1, skin=0.1, backs_block=False):
    """(verdict code, faces the box) per triangle of (V, T): rays from every eye to `samples`
    points of each front-facing triangle, blocked by `occ` (whose first len(T) are T)."""
    box = view_box(box)
    a, b, c, n, ln = _tri_frames(V, T)
    k = len(T)
    degenerate = ln < 1e-12
    tol = 1e-6 * np.maximum(ln, 1e-12)
    cen = (a + b + c) / 3.0
    sup = np.where(n > 0, np.asarray(box[3:]), np.asarray(box[:3]))
    box_front = np.einsum('ij,ij->i', n, sup - cen) > tol
    seen = np.zeros(k, bool)
    any_front = np.zeros(k, bool)
    own = np.arange(k)
    E = np.asarray(eyes, np.float64).reshape(-1, 3)
    fronts = [((np.einsum('ij,j->i', n, e) - np.einsum('ij,ij->i', n, cen)) > tol)
              & box_front & ~degenerate for e in E]
    for f in fronts:
        any_front |= f
    # every eye at the centroid first; the extra samples only chase what is still unseen
    for w in SAMPLE_BARY[:max(1, min(samples, len(SAMPLE_BARY)))]:
        for e, front in zip(E, fronts):
            tgt = np.where(front & ~seen)[0]
            if not len(tgt):
                continue
            S = w[0] * a[tgt] + w[1] * b[tgt] + w[2] * c[tgt]
            blk = occ.blocked(np.broadcast_to(e, S.shape), S - e, exclude=own[tgt], skin=skin,
                              backs_block=backs_block)
            seen[tgt[~blk]] = True
    out = np.full(k, UNSEEN, np.int64)
    out[~box_front] = BACKFACE
    out[any_front & ~seen] = OCCLUDED
    out[seen] = SEEN
    out[degenerate] = UNSEEN
    return out, box_front & ~degenerate


def _mesh_verdict(counts):
    """One seen triangle makes the mesh seen; otherwise the majority, occluded on a tie."""
    if counts[SEEN]:
        return SEEN
    rest = [(counts[v], -v) for v in (OCCLUDED, BACKFACE, UNSEEN) if counts[v]]
    return -max(rest)[1] if rest else UNSEEN


def facing_report(m, section, box, others=(), eyes=EYES, samples=SAMPLES, skin=0.1,
                  backs_block=False, height=EYE_HEIGHT):
    """Per mesh: triangle count, the front-facing and ray-visible shares, the material,
    a verdict; the per-triangle codes ride along for slice_section."""
    idx = section_index(m, section)
    box = view_box(box)
    E = eye_points(box, eyes, height)
    own_V, own_T, owner = section_soup(m, idx)
    pts = np.concatenate([E, own_V]) if len(own_V) else E
    region = (pts.min(axis=0) - skin, pts.max(axis=0) + skin)
    V, T = _occluder_soup(m, idx, others, region)
    codes, front = tri_verdicts(own_V, own_T, box, E, Occluders(V, T), samples, skin, backs_block)
    meshes = []
    totals = collections.Counter()
    for mi, me in enumerate(m['sections'][idx]['meshes']):
        sel = owner == mi
        mc = codes[sel]
        counts = {v: int((mc == v).sum()) for v in range(4)}
        nt = int(sel.sum())
        vd = _mesh_verdict(counts) if nt else UNSEEN
        backs = own_T[sel][mc == BACKFACE]
        back_bbox = None
        if len(backs):
            p = own_V[backs.ravel()]
            back_bbox = tuple(p.min(axis=0)) + tuple(p.max(axis=0))
        meshes.append({'index': mi, 'name': geom.name_str(me['name']), 'tris': nt,
                       'back_bbox': back_bbox,
                       'front': int(front[sel].sum()), 'seen': counts[SEEN],
                       'front_share': (float(front[sel].sum()) / nt) if nt else 0.0,
                       'seen_share': (counts[SEEN] / float(nt)) if nt else 0.0,
                       'counts': {VERDICTS[v]: counts[v] for v in range(4)},
                       'verdict': VERDICTS[vd], 'material': material_label(m, me),
                       'inline': not (me['flags'] & INLINE), 'tri_verdict': mc})
        totals.update({VERDICTS[v]: counts[v] for v in range(4)})
    return {'section': idx, 'name': geom.name_str(m['sections'][idx]['name']), 'box': box,
            'eyes': E, 'samples': samples, 'tris': int(len(own_T)), 'meshes': meshes,
            'totals': {v: totals.get(v, 0) for v in VERDICTS},
            'mesh_verdicts': {v: sum(1 for r in meshes if r['verdict'] == v) for v in VERDICTS}}


def format_report(rep, every=False):
    lines = ['%s (section %d): %d meshes, %d triangles, box %s, %d eyes x %d samples'
             % (rep['name'], rep['section'], len(rep['meshes']), rep['tris'],
                ' '.join('%g' % v for v in rep['box']), len(rep['eyes']), rep['samples'])]
    t = rep['totals']
    lines.append('triangles: ' + ', '.join('%s %d' % (v, t[v]) for v in VERDICTS))
    mv = rep['mesh_verdicts']
    lines.append('meshes:    ' + ', '.join('%s %d' % (v, mv[v]) for v in VERDICTS))
    lines.append('%4s %-28s %6s %6s %6s  %-14s %s'
                 % ('#', 'mesh', 'tris', 'front', 'seen', 'verdict', 'material'))
    for r in rep['meshes']:
        if not every and r['verdict'] == 'seen' and r['seen_share'] >= 0.999:
            continue
        lines.append('%4d %-28s %6d %5.0f%% %5.0f%%  %-14s %s'
                     % (r['index'], r['name'][:28], r['tris'], 100 * r['front_share'],
                        100 * r['seen_share'], r['verdict'], r['material']))
        nb = r['counts']['backface-only']
        if nb and nb >= 0.1 * r['tris']:
            lines.append('     %d backs at %s' % (nb, ' '.join('%.1f' % v for v in r['back_bbox'])))
    return '\n'.join(lines)


# --- slicing --------------------------------------------------------------------------------
def _slice_geo(geo, keep):
    tris = [t for t, k in zip(geo['tris'], keep) if k]
    used = sorted(set(i for t in tris for i in t))
    remap = {old: new for new, old in enumerate(used)}
    out = dict(geo)
    out['nverts'] = len(used)
    out['tris'] = [(remap[a], remap[b], remap[c]) for a, b, c in tris]
    for key in ('vec', 'uv', 'color', 'raw'):
        out[key] = {s: [v[i] for i in used] for s, v in geo[key].items()}
    for key in ('vecraw', 'uvraw'):
        out[key] = {s: {remap[i]: b for i, b in d.items() if i in remap}
                    for s, d in geo.get(key, {}).items()}
    out['pad'] = [geo['pad'][i] for i in used] if geo.get('pad') else None
    return out


def _capture(m):
    if 'origdatasize' not in m:
        geom.capture(m)


def _kept_soup(rec):
    vs, ts = [], []
    base = 0
    for me in rec['meshes']:
        arr = mesh_arrays(me)
        if arr is None:
            continue
        vs.append(arr[0])
        ts.append(arr[1] + base)
        base += len(arr[0])
    if not vs:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return np.concatenate(vs), np.concatenate(ts)


def _vkey(p):
    return struct.pack('<3f', *p)


def rebuild_bvt(rec, inflate=0.01):
    """The BVT over the record's render triangles, both windings, padded as retail pads. A
    triangle matching a shipped collision triangle by position keeps its surface and flags."""
    V, T = _kept_soup(rec)
    if not len(T):
        rec['bvt'], rec['bvtflag'] = None, 0
        return 0
    shipped = {}
    surf_c, flag_c = collections.Counter(), collections.Counter()
    if rec['bvt'] is not None:
        cv, ct, cs, cf = bvt_collect(rec['bvt'])
        for t, s, f in zip(ct, cs, cf):
            shipped[tuple(sorted(_vkey(cv[i]) for i in t))] = (s, f)
            surf_c[s] += 1
            flag_c[f] += 1
    dsurf = surf_c.most_common(1)[0][0] if surf_c else 0
    dflag = flag_c.most_common(1)[0][0] if flag_c else 0
    pool, verts = {}, []
    tris, surf, flags = [], [], []
    a, b, c, n, ln = _tri_frames(V, T)
    for ti in np.where(ln > 1e-12)[0]:
        ids = []
        for i in T[ti]:
            key = _vkey(V[i])
            gi = pool.get(key)
            if gi is None:
                gi = pool[key] = len(verts)
                verts.append(tuple(float(x) for x in V[i]))
            ids.append(gi)
        if len(set(ids)) < 3:
            continue
        s, f = shipped.get(tuple(sorted(_vkey(verts[i]) for i in ids)), (dsurf, dflag))
        for tri in ((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[1])):
            tris.append(tri)
            surf.append(s)
            flags.append(f)
    if not tris:
        rec['bvt'], rec['bvtflag'] = None, 0
        return 0
    if max(surf) >= max(1, len(rec['lrefs'])):
        raise SliceError('surface index %d outside %d lrefs' % (max(surf), len(rec['lrefs'])))
    rec['bvt'], rec['bvtflag'] = bvt_build(verts, tris, surf, flags, inflate=inflate)
    return len(tris)


def _finish_record(rec, bbox):
    """'tight' bounds the kept meshes and the BVT; 'keep' unions the shipped box in, which
    can carry editor padding (Room64 ships 9 ft of it under the road)."""
    bb = geom.section_bbox(rec)
    if bb is None:
        raise SliceError('%s: nothing left to bound' % geom.name_str(rec['name']))
    if bbox == 'keep':
        bb = union_bbox([bb, rec['bbox']])
    rec['bbox'] = struct.pack('<6f', *bb)
    # the blob as relayout lays it, so a capture() on a fresh parse finds the ballast in place
    parts = []
    for me in rec['meshes']:
        if not (me['flags'] & INLINE):
            parts += [me['pkt']['vdata'], me['pkt']['idata']] + list(me['pkt'].get('mdata') or [])
    body = b''.join(parts)
    body += b'\0' * ((-len(body)) % geom.BLOB_ALIGN)
    rec['blob'] = body + rec.get('ballast', b'')
    rec['fa20'] = len(rec['blob'])
    return rec


def kept_indices(report, keep=('seen',), tris=True):
    """Source mesh indices a slice keeps, in order: the sliced record's meshes map onto them."""
    codes = set(VERDICTS.index(k) for k in keep)
    out = []
    for r in report['meshes']:
        if not r['inline']:
            out.append(r['index'])
        elif tris and np.isin(r['tri_verdict'], list(codes)).any():
            out.append(r['index'])
        elif not tris and r['tris'] and VERDICTS.index(r['verdict']) in codes:
            out.append(r['index'])
    return out


def slice_section(m, section, box=None, keep=('seen',), tris=True, others=(), eyes=EYES,
                  report=None, bvt='rebuild', samples=SAMPLES, skin=0.1, backs_block=False,
                  height=EYE_HEIGHT, bbox='tight'):
    """A copy of the section holding only the triangles (or, with tris=False, the meshes)
    whose verdict is in `keep`. Returns (record, report)."""
    idx = section_index(m, section)
    _capture(m)
    if report is None:
        if box is None:
            raise SliceError('slice_section needs a view box or a report')
        report = facing_report(m, idx, box, others, eyes, samples, skin, backs_block, height)
    keep_codes = set(VERDICTS.index(k) for k in keep)
    if bvt not in ('rebuild', 'keep', 'drop'):
        raise SliceError("bvt must be 'rebuild', 'keep' or 'drop'")
    if bbox not in ('tight', 'keep'):
        raise SliceError("bbox must be 'tight' or 'keep'")
    src = m['sections'][idx]
    rec = copy.deepcopy(src)
    rec['meshes'] = []
    for mi in kept_indices(report, keep, tris):
        me, r = copy.deepcopy(src['meshes'][mi]), report['meshes'][mi]
        if tris and not (me['flags'] & INLINE):
            mask = np.isin(r['tri_verdict'], list(keep_codes))
            if not mask.all():
                if me['pkt']['morphs']:
                    raise SliceError('mesh %r carries morphs; slice it whole (tris=False)'
                                     % r['name'])
                me['pkt'] = geom.encode_mesh(_slice_geo(geom.decode_mesh(me['pkt']), mask))
        rec['meshes'].append(me)
    if not any(mesh_arrays(me) for me in rec['meshes']):
        raise SliceError('%s: no mesh keeps a triangle from that box' % report['name'])
    if bvt == 'rebuild' and src['bvt'] is not None:
        rebuild_bvt(rec)
    elif bvt == 'drop':
        rec['bvt'], rec['bvtflag'] = None, 0
    report['bvt'] = bvt if src['bvt'] is not None else 'none'
    return _finish_record(rec, bbox), report


def backdrop(m, section, box, keep=('seen',), others=(), **kw):
    """Whole meshes only: scenery that is drawn and never walked."""
    return slice_section(m, section, box, keep=keep, tris=False, others=others, **kw)


def two_sided(m, section, meshes):
    """A copy of the section whose named (or indexed) meshes carry every triangle twice,
    the second with flipped winding, normal and binormal."""
    idx = section_index(m, section)
    _capture(m)
    rec = copy.deepcopy(m['sections'][idx])
    want = set()
    for sel in meshes:
        if isinstance(sel, (int, np.integer)):
            want.add(int(sel))
        else:
            want.update(i for i, me in enumerate(rec['meshes']) if geom.name_str(me['name']) == sel)
    if not want:
        raise SliceError('no mesh matches %r' % (meshes,))
    for mi in sorted(want):
        me = rec['meshes'][mi]
        if me['flags'] & INLINE:
            raise SliceError('mesh %d has no render packet' % mi)
        g = geom.decode_mesh(me['pkt'])
        n = g['nverts']
        if 2 * n > MAX_VERTS:
            raise SliceError('mesh %d: %d vertices doubled exceed the 16-bit index cap' % (mi, n))
        for slot, vals in list(g['vec'].items()):
            flip = slot in (geom.S_NORMAL, geom.S_BINORMAL)
            g['vec'][slot] = vals + [(-x, -y, -z) if flip else (x, y, z) for x, y, z in vals]
        for key in ('uv', 'color', 'raw'):
            g[key] = {s: v + v for s, v in g[key].items()}
        for key in ('vecraw', 'uvraw'):
            g[key] = {s: dict(d, **{i + n: b for i, b in d.items()})
                      for s, d in g.get(key, {}).items()}
        if g.get('pad'):
            g['pad'] = g['pad'] + g['pad']
        g['tris'] = g['tris'] + [(a + n, c + n, b + n) for a, b, c in g['tris']]
        g['nverts'] = 2 * n
        me['pkt'] = geom.encode_mesh(g)
    return rec


def slice_set(m, section, box, lightmap_dir, **kw):
    """host.new_set_from_section around the section, then the slice in its place.
    Returns (set, staged, report); the set is laid out and ready for bst.build."""
    idx = section_index(m, section)
    _capture(m)
    seed, staged = host.new_set_from_section(m, idx, lightmap_dir)
    rec, rep = slice_section(seed, 0, box, **kw)
    seed['sections'][0] = rec
    seed['bsp'] = tp._pack_bsp((0.0, 0.0, 0.0, 0.0), 0, -1, -1, struct.unpack('<6f', rec['bbox']))
    geom.relayout(seed)
    return seed, staged, rep


# --- CLI --------------------------------------------------------------------------------------
def _load(path):
    with open(path, 'rb') as fh:
        return bst.parse(fh.read())


def _section_arg(s):
    return int(s) if s.lstrip('-').isdigit() else s


def _common(p):
    p.add_argument('bst')
    p.add_argument('section', type=_section_arg, help='index or name')
    p.add_argument('--box', type=float, nargs=6, required=True,
                   metavar=('X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1'),
                   help='the view box, set coordinates')
    p.add_argument('--others', type=_section_arg, nargs='*', default=[],
                   help='sections of the same set that also occlude')
    p.add_argument('--eyes', type=int, default=EYES, help='eye grid per axis (default %d)' % EYES)
    p.add_argument('--samples', type=int, default=SAMPLES,
                   help='ray targets per triangle, 1..4 (default %d)' % SAMPLES)
    p.add_argument('--height', type=float, nargs='+', default=[EYE_HEIGHT],
                   help='eye heights above the box floor (default %g)' % EYE_HEIGHT)
    p.add_argument('--backs-block', action='store_true', help='backs occlude too')


def _out(p):
    p.add_argument('-o', '--out', required=True)
    p.add_argument('--keep', default='seen', help='verdicts to keep, comma separated')
    p.add_argument('--bvt', choices=('rebuild', 'keep', 'drop'), default='rebuild')
    p.add_argument('--bbox', choices=('tight', 'keep'), default='tight',
                   help="'keep' unions the shipped section box in (editor padding survives)")
    p.add_argument('--dir', help='directory component for the tile and probe names')


def cmd_report(a):
    m = _load(a.bst)
    rep = facing_report(m, a.section, a.box, a.others, a.eyes, a.samples, backs_block=a.backs_block,
                        height=a.height)
    print(format_report(rep, every=a.all))
    return 0


def _cmd_slice(a, tris):
    m = _load(a.bst)
    keep = tuple(k.strip() for k in a.keep.split(',') if k.strip())
    bad = [k for k in keep if k not in VERDICTS]
    if bad:
        raise SliceError('unknown verdict %r' % bad[0])
    dirname = a.dir or os.path.splitext(os.path.basename(a.out))[0]
    seed, staged, rep = slice_set(m, a.section, a.box, dirname, keep=keep, tris=tris,
                                  others=a.others, eyes=a.eyes, bvt=a.bvt, samples=a.samples,
                                  backs_block=a.backs_block, height=a.height, bbox=a.bbox)
    data = bst.build(seed)
    with open(a.out, 'wb') as fh:
        fh.write(data)
    with open(a.out, 'rb') as fh:
        if fh.read() != data:
            raise SliceError('%s did not read back as written' % a.out)
    print(format_report(rep))
    sec = seed['sections'][0]
    st = geom.section_stats(sec)
    print('wrote %s (%d bytes): %d of %d meshes, %d of %d triangles, bvt %s (%d tris), '
          '%d files to stage' % (a.out, len(data), st['meshes'], len(rep['meshes']), st['tris'],
                                 rep['tris'], rep['bvt'], st['colltris'], len(staged)))
    for src, dst in staged:
        print('    %s -> %s' % (src, dst))
    bad = tp.check_refs(bst.parse(data))
    for b in bad:
        print('    ' + b)
    return int(bool(bad))


def cmd_slice(a):
    return _cmd_slice(a, True)


def cmd_backdrop(a):
    return _cmd_slice(a, False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('report', help='facing and ray verdicts per mesh from a view box')
    _common(p)
    p.add_argument('--all', action='store_true', help='list fully seen meshes too')
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser('slice', help='a one-section set holding only the kept triangles')
    _common(p)
    _out(p)
    p.set_defaults(fn=cmd_slice)
    p = sub.add_parser('backdrop', help='the same by whole meshes: scenery')
    _common(p)
    _out(p)
    p.set_defaults(fn=cmd_backdrop)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except SliceError as exc:
        sys.stderr.write('slicer: %s\n' % exc)
        return 2


if __name__ == '__main__':
    sys.exit(main())

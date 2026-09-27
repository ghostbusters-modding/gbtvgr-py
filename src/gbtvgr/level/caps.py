"""Mouths: the rim a loft corridor ends on, the caps that close a box corridor's end
down to the room's real mouth, and the plug for an opening that leads nowhere."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import argparse
import math
import struct
import sys

import numpy as np

from ..mesh.bvt import bvt_build, bvt_collect, union_bbox
from ..mesh.smb import decl_stride
from ..sets import bst, geom
from . import lighting
from . import link
from . import sections as gsec
from . import textures
from . import transplant as tp
from .mathutil import F32

AX = {'x': 0, 'z': 2}
CROSS = {'x': 2, 'z': 0}
RIM_N = 96           # rim points per mouth
RIM_LADDER = (0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)   # fan depths into the room, first hit wins
RIM_REACH = 1.3      # no hit within this times the rect's half-diagonal: an open-sided doorway
RIM_SLACK = 0.5      # a hit this far past the rect is the room's far floor, not the mouth
PUSH = 0.5           # the axis ray starts this far outside the room's face
DEPTH = 4.0          # a hit no deeper than this is the room's own wall at the mouth
REACH = 8.0          # nothing this far along the axis, nor around: empty space
INSET = 1.0          # enclosure rays start this far inside the room
EDGE = 0.05          # lattice points pull this far off the rectangle's edges
WALL_COS = 0.5       # a hit surface within 60 degrees of facing the corridor is a wall
FACE_TOL = 0.5       # a mask plane must sit this close to the corridor's end face
BAND = 0.25          # traced cap: height of one row band
OVERLAP = 0.05       # traced cap: how far past the bisected rim edge the cap reaches
BISECT = 0.02        # traced cap: edge precision along the cross axis and the floor
MIN_GAP = 0.2        # traced cap: an opening narrower than this is closed over
METHODS = ('trace', 'cells')
PLUG_INSET = 0.05    # a plug sits this far inside the room's bbox face
PLUG_INFLATE = 0.02  # thinnest shipped BVT boxes are 0.03; never ship a flat one
MIN_EXTENT = 0.5     # a doorway thinner than this on its face is a line, not a hole
SKEW_TOL = 0.5       # the two halves of an aligned quad share a plane this closely
THICK_MAX = 4.0      # two copies of a quad sit up to 3 ft apart; more is a volume
MAX_VERTS = 65535


class Mask:
    """One mouth's cell grid. grid[j, i] is True where the room is open, j along y
    from y_lo, i along the cross axis from c_lo; solid marks the room's own wall."""

    def __init__(self, axis, sign, plane, c_lo, c_hi, y_lo, y_hi, grid, solid, riser=None):
        self.axis, self.sign, self.plane = axis, int(sign), float(plane)
        self.c_lo, self.c_hi = float(c_lo), float(c_hi)
        self.y_lo, self.y_hi = float(y_lo), float(y_hi)
        self.grid = np.asarray(grid, bool)
        self.solid = np.asarray(solid, bool)
        # closed cells a walkable floor crosses: capped from the cell bottom to this height
        self.riser = (np.full(self.grid.shape, np.nan) if riser is None
                      else np.asarray(riser, np.float64))
        # traced outline: [(y0, y1, [(left, right), ...])] per band, None until trace()
        self.bands = None
        # the (ny+1, nc+1) lattice the cells were judged from, for check_caps
        self.pt_open = np.zeros((self.ny + 1, self.nc + 1), bool)
        self.pt_solid = np.zeros((self.ny + 1, self.nc + 1), bool)
        self.pt_empty = np.ones((self.ny + 1, self.nc + 1), bool)

    @property
    def ny(self):
        return self.grid.shape[0]

    @property
    def nc(self):
        return self.grid.shape[1]

    @property
    def dc(self):
        return (self.c_hi - self.c_lo) / self.nc

    @property
    def dy(self):
        return (self.y_hi - self.y_lo) / self.ny

    @property
    def n_open(self):
        return int(self.grid.sum())

    @property
    def void(self):
        return ~self.grid & ~self.solid

    def cell(self, i, j):
        """(c0, c1, y0, y1) of cell column i, row j."""
        return (self.c_lo + i * self.dc, self.c_lo + (i + 1) * self.dc,
                self.y_lo + j * self.dy, self.y_lo + (j + 1) * self.dy)

    @property
    def open_rect(self):
        """Bounding (c0, y0, c1, y1) of the open cells, or None."""
        js, is_ = np.nonzero(self.grid)
        if not len(js):
            return None
        return (self.c_lo + is_.min() * self.dc, self.y_lo + js.min() * self.dy,
                self.c_lo + (is_.max() + 1) * self.dc, self.y_lo + (js.max() + 1) * self.dy)

    def point(self, c, y):
        """World point on the plane at cross coordinate c, height y."""
        p = [0.0, y, 0.0]
        p[AX[self.axis]] = self.plane
        p[CROSS[self.axis]] = c
        return tuple(p)

    def rows(self):
        """ASCII rows, top down: o open, # solid, . void, _ a floor riser."""
        out = []
        for j in range(self.ny - 1, -1, -1):
            out.append(''.join('o' if self.grid[j, i] else '_' if np.isfinite(self.riser[j, i])
                               else '#' if self.solid[j, i] else '.' for i in range(self.nc)))
        return out


# --- rays --------------------------------------------------------------------------------
EMPTY = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))


def _tris(bvt):
    """The collision triangles; a section without a tree gets an empty soup."""
    if bvt is None:
        return EMPTY
    v, t, _s, _f = bvt_collect(bvt)
    if not t:
        return EMPTY
    V = np.asarray(v, np.float64)
    T = np.asarray(t, np.int64)
    return V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]


def _slab(tri, ax, lo, hi):
    """Triangles whose extent on axis ax overlaps [lo, hi]."""
    v0, v1, v2 = tri
    a = np.stack([v0[:, ax], v1[:, ax], v2[:, ax]], axis=1)
    keep = (a.min(axis=1) <= hi) & (a.max(axis=1) >= lo)
    return v0[keep], v1[keep], v2[keep]


def _nearest(origins, direction, tri, chunk=128):
    """(distance, triangle index) of the first triangle each ray meets, inf and -1
    for a miss. Both windings ship in every BVT, so no culling."""
    out = np.full(len(origins), np.inf)
    hit = np.full(len(origins), -1)
    if tri is None or not len(tri[0]) or not len(origins):
        return out, hit
    v0, v1, v2 = tri
    d = np.asarray(direction, np.float64)
    e1, e2 = v1 - v0, v2 - v0
    p = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, p)
    ok_t = np.abs(det) > 1e-9
    inv = np.where(ok_t, 1.0 / np.where(ok_t, det, 1.0), 0.0)
    for s in range(0, len(origins), chunk):
        o = np.asarray(origins[s:s + chunk], np.float64)
        tv = o[:, None, :] - v0[None, :, :]
        u = np.einsum('rtj,tj->rt', tv, p) * inv
        q = np.cross(tv, e1[None, :, :])
        v = np.einsum('j,rtj->rt', d, q) * inv
        t = np.einsum('tj,rtj->rt', e2, q) * inv
        ok = ok_t[None, :] & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-6)
        t = np.where(ok, t, np.inf)
        best = t.argmin(axis=1)
        out[s:s + chunk] = t[np.arange(len(o)), best]
        hit[s:s + chunk] = np.where(np.isfinite(out[s:s + chunk]), best, -1)
    return out, hit


def _nearest_dirs(origins, dirs, tri, chunk=16):
    """_nearest with a direction per ray."""
    out = np.full(len(origins), np.inf)
    if tri is None or not len(tri[0]) or not len(origins):
        return out
    v0, v1, v2 = tri
    e1, e2 = v1 - v0, v2 - v0
    for s in range(0, len(origins), chunk):
        o = np.asarray(origins[s:s + chunk], np.float64)
        d = np.asarray(dirs[s:s + chunk], np.float64)
        p = np.cross(d[:, None, :], e2[None, :, :])
        det = np.einsum('tj,rtj->rt', e1, p)
        ok_t = np.abs(det) > 1e-9
        inv = np.where(ok_t, 1.0 / np.where(ok_t, det, 1.0), 0.0)
        tv = o[:, None, :] - v0[None, :, :]
        u = np.einsum('rtj,rtj->rt', tv, p) * inv
        q = np.cross(tv, e1[None, :, :])
        v = np.einsum('rj,rtj->rt', d, q) * inv
        t = np.einsum('tj,rtj->rt', e2, q) * inv
        ok = ok_t & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-6)
        out[s:s + chunk] = np.where(ok, t, np.inf).min(axis=1)
    return out


def _facing(tri, ax):
    """|unit normal| along ax per triangle: 1 faces the axis squarely, 0 runs along it."""
    v0, v1, v2 = tri
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1)
    return np.abs(n[:, ax]) / np.where(ln > 1e-12, ln, 1.0)


def _fronts(tri, direction):
    """True per triangle when its CCW normal points against the ray: the side the
    artist made visible."""
    v0, v1, v2 = tri
    n = np.cross(v1 - v0, v2 - v0)
    return n @ np.asarray(direction, np.float64) < 0.0


def _render_tris(sec):
    """The section's single-sided render triangles, read straight out of the vertex
    buffers (position is always float3 at offset 0)."""
    parts = []
    for me in sec['meshes']:
        if me['flags'] & 0x20000:
            continue
        pkt = me['pkt']
        n, st = pkt['nverts'], decl_stride(pkt['decl'])
        if not n or not pkt['nprims']:
            continue
        vd = np.frombuffer(pkt['vdata'], np.uint8)[:n * st].reshape(n, st)
        pos = np.nan_to_num(vd[:, :12].copy().view('<f4').reshape(n, 3).astype(np.float64))
        idx = np.frombuffer(pkt['idata'], '<u2')[:pkt['nprims'] * 3].reshape(-1, 3)
        parts.append(pos[idx])
    if not parts:
        return None
    P = np.concatenate(parts)
    return P[:, 0], P[:, 1], P[:, 2]


def _cells(pts):
    """Cells whose four lattice corners all hold."""
    return pts[:-1, :-1] & pts[:-1, 1:] & pts[1:, :-1] & pts[1:, 1:]


def _lattice(c_lo, c_hi, y_lo, y_hi, nc, ny):
    cs = np.linspace(c_lo, c_hi, nc + 1)
    ys = np.linspace(y_lo, y_hi, ny + 1)
    cs = np.clip(cs, c_lo + EDGE, c_hi - EDGE) if c_hi - c_lo > 2 * EDGE else cs
    ys = np.clip(ys, y_lo + EDGE, y_hi - EDGE) if y_hi - y_lo > 2 * EDGE else ys
    gc, gy = np.meshgrid(cs, ys, indexing='xy')
    return gc.ravel(), gy.ravel()


def _classify(tri, rtri, axis, sign, plane, cs, ys, depth, outdoor=True):
    """(solid, open, empty, inner) per lattice point: geometry within depth along the
    axis; a front-facing render floor below and no wall; nothing around; the inset slab.
    Indoors, open also needs a ceiling: a cave wall's outer slivers sit over floors too."""
    a, c = AX[axis], CROSS[axis]
    n = len(cs)
    o = np.zeros((n, 3))
    o[:, 1], o[:, c] = ys, cs
    o[:, a] = plane + sign * PUSH
    d = [0.0, 0.0, 0.0]
    d[a] = -sign
    lo, hi = sorted((plane + sign * PUSH, plane - sign * depth))
    # render counts too: a wall nobody can reach ships without collision
    slab = _slab(tri, a, lo, hi)
    if rtri is not None:
        rs = _slab(rtri, a, lo, hi)
        slab = tuple(np.concatenate([x, y]) for x, y in zip(slab, rs))
    t, hit = _nearest(o, d, slab)
    solid = t <= PUSH + depth
    # a skim along a sloped floor or a tube's rim is solid, but it is not a wall
    wall = solid & (_facing(slab, a)[np.maximum(hit, 0)] >= WALL_COS) if len(slab[0]) else solid
    p = plane - sign * INSET
    o[:, a] = p
    inner = _slab(tri, a, p - 1e-3, p + 1e-3)
    down = (0.0, -1.0, 0.0)
    if rtri is None:
        floor = np.isfinite(_nearest(o, down, inner)[0])
        both = inner
    else:
        # the render mesh is single-sided: over a floor it faces us, over a roof it does not
        rinner = _slab(rtri, a, p - 1e-3, p + 1e-3)
        t, hit = _nearest(o, down, rinner)
        floor = np.isfinite(t)
        if len(rinner[0]):
            floor &= _fronts(rinner, down)[np.maximum(hit, 0)]
            if not outdoor:
                up = (0.0, 1.0, 0.0)
                t, hit = _nearest(o, up, rinner)
                floor &= np.isfinite(t) & _fronts(rinner, up)[np.maximum(hit, 0)]
        both = tuple(np.concatenate([x, y]) for x, y in zip(inner, rinner))
    lo, hi = sorted((p, p - sign * REACH))
    reach = _slab(tri, a, lo, hi)
    if rtri is not None:
        reach = tuple(np.concatenate([x, y]) for x, y in zip(reach, _slab(rtri, a, lo, hi)))
    empty = _nearest(o, d, reach)[0] > REACH
    for dd in (down, (0.0, 1.0, 0.0)):
        empty &= np.isinf(_nearest(o, dd, both)[0])
    for sd in (-1.0, 1.0):
        dd = [0.0, 0.0, 0.0]
        dd[c] = sd
        empty &= np.isinf(_nearest(o, dd, both)[0])
    return solid, ~wall & floor, empty, inner


def _risers(inner, axis, sign, plane, c_lo, dc, nc, y_lo):
    """Per column, the room's floor height at the mouth where it stands above y_lo,
    else nan: the cap below a walkable floor ends at the floor, not at a cell edge."""
    a, c = AX[axis], CROSS[axis]
    cs = np.concatenate([c_lo + np.arange(nc + 1) * dc, c_lo + (np.arange(nc) + 0.5) * dc])
    cs = np.clip(cs, c_lo + EDGE, c_lo + nc * dc - EDGE)
    o = np.zeros((len(cs), 3))
    o[:, 1], o[:, c], o[:, a] = y_lo + EDGE, cs, plane - sign * INSET
    down = _nearest(o, (0.0, -1.0, 0.0), inner)[0]
    up = _nearest(o, (0.0, 1.0, 0.0), inner)[0]
    h = np.where(np.isinf(down) & np.isfinite(up), y_lo + EDGE + up, np.nan)
    edge, mid = h[:nc + 1], h[nc + 1:]
    stack = np.stack([edge[:-1], edge[1:], mid], axis=1)
    best = np.where(np.isnan(stack), -np.inf, stack).max(axis=1)
    return np.where(np.isinf(best), np.nan, best)


def _geometry(m, room_index, cache=None):
    """(collision, render) triangle arrays of a section, memoised in cache when given."""
    if cache is not None and room_index in cache:
        return cache[room_index]
    sec = m['sections'][room_index]
    out = (_tris(sec['bvt']), _render_tris(sec))
    if cache is not None:
        cache[room_index] = out
    return out


def mouth_mask(m, room_index, plane, axis, sign, lo, hi, cell=0.5, depth=DEPTH, cache=None):
    """Grid over the rectangle lo..hi (3-vectors, the axis component ignored) on the
    room's face plane. sign +1: the corridor lies beyond the room's hi face."""
    if axis not in AX:
        raise ValueError("axis must be 'x' or 'z', not %r" % (axis,))
    c = CROSS[axis]
    c_lo, c_hi, y_lo, y_hi = float(lo[c]), float(hi[c]), float(lo[1]), float(hi[1])
    if c_hi <= c_lo or y_hi <= y_lo:
        raise ValueError('mouth rectangle is empty: %s..%s' % (lo, hi))
    nc = max(1, int(math.ceil((c_hi - c_lo) / cell - 1e-6)))
    ny = max(1, int(math.ceil((y_hi - y_lo) / cell - 1e-6)))
    tri, rtri = _geometry(m, room_index, cache)
    cs, ys = _lattice(c_lo, c_hi, y_lo, y_hi, nc, ny)
    riser = np.full((ny, nc), np.nan)
    outdoor = bool(m['sections'][room_index]['b2'])
    solid, open_, empty, inner = _classify(tri, rtri, axis, sign, plane, cs, ys, depth, outdoor)
    solid = solid.reshape(ny + 1, nc + 1)
    open_ = open_.reshape(ny + 1, nc + 1)
    # a cell is open only when all four corners are: the cap overlaps the rim, never the hole
    cell_open = _largest(_cells(open_))
    cell_solid = ~cell_open & (solid[:-1, :-1] | solid[:-1, 1:] | solid[1:, :-1] | solid[1:, 1:])
    if len(inner[0]):
        dy = (y_hi - y_lo) / ny
        h = _risers(inner, axis, sign, plane, c_lo, (c_hi - c_lo) / nc, nc, y_lo)
        for i in np.nonzero(np.isfinite(h))[0]:
            for j in range(ny):
                if cell_open[j, i]:
                    break
                y0, y1 = y_lo + j * dy, y_lo + (j + 1) * dy
                if y1 <= h[i]:
                    continue
                if h[i] > y0 and open_[j + 1, i] and open_[j + 1, i + 1]:
                    riser[j, i] = h[i]
                break
    mk = Mask(axis, sign, plane, c_lo, c_hi, y_lo, y_hi, cell_open, cell_solid, riser)
    mk.pt_open, mk.pt_solid = open_, solid
    mk.pt_empty = empty.reshape(ny + 1, nc + 1)
    return mk


def _largest(cells):
    """The biggest 4-connected block of True cells: the passage, without the stray
    cells that happen to sit over some visible surface elsewhere."""
    seen = np.zeros(cells.shape, bool)
    best = np.zeros(cells.shape, bool)
    ny, nc = cells.shape
    for j0, i0 in zip(*np.nonzero(cells)):
        if seen[j0, i0]:
            continue
        comp, stack = [], [(j0, i0)]
        seen[j0, i0] = True
        while stack:
            j, i = stack.pop()
            comp.append((j, i))
            for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1)):
                if 0 <= jj < ny and 0 <= ii < nc and cells[jj, ii] and not seen[jj, ii]:
                    seen[jj, ii] = True
                    stack.append((jj, ii))
        if len(comp) > int(best.sum()):
            best[:] = False
            for j, i in comp:
                best[j, i] = True
    return best


def open_union(masks):
    """Bounding (c0, y0, c1, y1) of every open cell of every mask, or None."""
    rects = [mk.open_rect for mk in masks]
    rects = [r for r in rects if r is not None]
    if not rects:
        return None
    return (min(r[0] for r in rects), min(r[1] for r in rects),
            max(r[2] for r in rects), max(r[3] for r in rects))


# --- the rim ---------------------------------------------------------------------------------
class Rim:
    """A mouth's outline: n points on the plane at equal angles about the centre, angle 0
    along +cross. depth[i] is how far into the room point i was probed."""

    def __init__(self, axis, sign, plane, centre, pts, depth, clamped, rect=None):
        self.axis, self.sign, self.plane = axis, int(sign), float(plane)
        self.centre = (float(centre[0]), float(centre[1]))
        self.pts = np.asarray(pts, np.float64).reshape(-1, 2)
        self.depth = np.asarray(depth, np.float64)
        self.clamped = np.asarray(clamped, bool)
        self.rect = rect

    @property
    def n(self):
        return len(self.pts)

    @property
    def lowest(self):
        """(c, y) of the lowest rim point."""
        i = int(np.argmin(self.pts[:, 1]))
        return float(self.pts[i, 0]), float(self.pts[i, 1])

    @property
    def arcs(self):
        """Cumulative distance around the outline, n + 1 values, the last the perimeter."""
        d = np.linalg.norm(np.roll(self.pts, -1, axis=0) - self.pts, axis=1)
        return np.concatenate([[0.0], np.cumsum(d)])

    def points3(self, inset=0.0, push=0.0):
        """(n, 3) world points at plane - sign * inset, pushed radially from the centre."""
        p = self.pts.copy()
        if np.any(push):
            r = p - np.asarray(self.centre)[None, :]
            ln = np.linalg.norm(r, axis=1, keepdims=True)
            p += np.reshape(push, (-1, 1)) * r / np.where(ln > 1e-9, ln, 1.0)
        out = np.zeros((self.n, 3))
        out[:, CROSS[self.axis]] = p[:, 0]
        out[:, 1] = p[:, 1]
        out[:, AX[self.axis]] = self.plane - self.sign * np.asarray(inset, np.float64)
        return out

    def chord(self, y):
        """(c0, c1) of the outline's interior at height y around the lowest point, or None."""
        p, q = self.pts, np.roll(self.pts, -1, axis=0)
        dy = q[:, 1] - p[:, 1]
        cross = ((p[:, 1] <= y) & (q[:, 1] > y)) | ((q[:, 1] <= y) & (p[:, 1] > y))
        if not cross.any():
            return None
        f = (y - p[cross, 1]) / dy[cross]
        cs = np.sort(p[cross, 0] + f * (q[cross, 0] - p[cross, 0]))
        c_low = self.lowest[0]
        for k in range(0, len(cs) - 1, 2):
            if cs[k] <= c_low <= cs[k + 1]:
                return float(cs[k]), float(cs[k + 1])
        return float(cs[0]), float(cs[-1])

    def is_rect(self, tol=0.05):
        """True when every point sits on the outline's own bounding rectangle."""
        lo, hi = self.pts.min(axis=0), self.pts.max(axis=0)
        on = np.min(np.abs(np.stack([self.pts - lo[None, :], hi[None, :] - self.pts])), axis=(0, 2))
        return bool(np.all(on <= tol))


def _rect_exit(cc, cy, dirs, rect_c, rect_y):
    """Distance along each in-plane ray from (cc, cy) to the rectangle's boundary."""
    t = np.full(len(dirs), np.inf)
    for k, (lo, hi, at) in enumerate(((rect_c[0], rect_c[1], cc), (rect_y[0], rect_y[1], cy))):
        d = dirs[:, k]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.minimum(t, np.where(d > 1e-12, (hi - at) / d, np.inf))
            t = np.minimum(t, np.where(d < -1e-12, (lo - at) / d, np.inf))
    return t


def mouth_rim(m, room, plane, centre, axis, n=RIM_N, rect=None, sign=None, insets=RIM_LADDER,
              reach=RIM_REACH, slack=RIM_SLACK, cache=None):
    """The room's real outline around an opening on `plane`: a fan of n rays from `centre`
    cast in the plane against the render mesh. rect = (lo, hi) clamps open-sided rays."""
    if axis not in AX:
        raise ValueError("axis must be 'x' or 'z', not %r" % (axis,))
    a, c = AX[axis], CROSS[axis]
    bb = struct.unpack('<6f', m['sections'][room]['bbox'])
    if sign is None:
        sign = 1 if plane >= (bb[a] + bb[a + 3]) / 2.0 else -1
    cc, cy = float(centre[c]), float(centre[1])
    th = 2.0 * np.pi * np.arange(n) / n
    d2 = np.stack([np.cos(th), np.sin(th)], axis=1)
    dirs = np.zeros((n, 3))
    dirs[:, c], dirs[:, 1] = d2[:, 0], d2[:, 1]
    limit, exit_t = np.inf, None
    if rect is not None:
        lo, hi = rect
        rc, ry = (float(lo[c]), float(hi[c])), (float(lo[1]), float(hi[1]))
        limit = reach * math.hypot((rc[1] - rc[0]) / 2.0, (ry[1] - ry[0]) / 2.0)
        exit_t = _rect_exit(cc, cy, d2, rc, ry)
    _tri, rtri = _geometry(m, room, cache)
    lo_a, hi_a = sorted((plane, plane - sign * (max(insets) + 1e-3)))
    slab = _slab(rtri, a, lo_a, hi_a) if rtri is not None else None
    dist = np.full(n, np.inf)
    depth = np.full(n, float(insets[0]))
    for s in insets:
        todo = np.isinf(dist)
        if not todo.any():
            break
        o = np.zeros((int(todo.sum()), 3))
        o[:, a], o[:, c], o[:, 1] = plane - sign * s, cc, cy
        t = _nearest_dirs(o, dirs[todo], slab)
        t = np.where(t <= limit, t, np.inf)
        hit = np.isfinite(t)
        idx = np.nonzero(todo)[0][hit]
        dist[idx], depth[idx] = t[hit], s
    clamped = np.isinf(dist)
    if exit_t is not None:
        clamped |= dist > exit_t + slack
        dist = np.where(clamped, exit_t, dist)
        depth = np.where(clamped, float(insets[0]), depth)
    if np.isinf(dist).any():
        raise ValueError('ray %d of the mouth fan on section %d finds no geometry and no rect'
                         % (int(np.nonzero(np.isinf(dist))[0][0]), room))
    if np.median(dist) < 0.5:
        raise ValueError('the fan centre %s sits inside section %d\'s rock' % ((cc, cy), room))
    pts = np.array([cc, cy])[None, :] + dist[:, None] * d2
    return Rim(axis, sign, plane, (cc, cy), pts, depth, clamped, rect)


# --- rectangles ----------------------------------------------------------------------------
def _runs(row):
    out, start = [], None
    for i, closed in enumerate(list(row) + [False]):
        if closed and start is None:
            start = i
        elif not closed and start is not None:
            out.append((start, i))
            start = None
    return out


def closed_rects(mask):
    """Every closed cell as (c0, c1, y0, y1): runs along a row merged into one strip,
    identical neighbouring rows into one band, and riser cells cut at their floor."""
    partial = np.isfinite(mask.riser)
    closed = ~mask.grid & ~partial
    rows = [tuple(_runs(closed[j])) for j in range(mask.ny)]
    out, j = [], 0
    while j < mask.ny:
        k = j
        while k + 1 < mask.ny and rows[k + 1] == rows[j]:
            k += 1
        for i0, i1 in rows[j]:
            out.append((mask.c_lo + i0 * mask.dc, mask.c_lo + i1 * mask.dc,
                        mask.y_lo + j * mask.dy, mask.y_lo + (k + 1) * mask.dy))
        j = k + 1
    for j, i in zip(*np.nonzero(partial)):
        c0, c1, y0, _y1 = mask.cell(i, j)
        out.append((c0, c1, y0, float(mask.riser[j, i])))
    return out


def _clip(rect, R, eps=1e-4):
    c0, c1, y0, y1 = (max(rect[0], R[0]), min(rect[1], R[1]),
                      max(rect[2], R[2]), min(rect[3], R[3]))
    if c1 - c0 <= eps or y1 - y0 <= eps:
        return None
    return (c0, c1, y0, y1)


def _frame(M, R, eps=1e-4):
    """R minus M as up to four rectangles: the corridor face outside the mask."""
    out = []
    if M[2] - R[2] > eps:
        out.append((R[0], R[1], R[2], min(M[2], R[3])))
    if R[3] - M[3] > eps:
        out.append((R[0], R[1], max(M[3], R[2]), R[3]))
    y0, y1 = max(M[2], R[2]), min(M[3], R[3])
    if y1 - y0 > eps:
        if M[0] - R[0] > eps:
            out.append((R[0], min(M[0], R[1]), y0, y1))
        if R[1] - M[1] > eps:
            out.append((max(M[1], R[0]), R[1], y0, y1))
    return out


def cap_rects(mask, R):
    """Closed cells clipped to the corridor face R = (c0, c1, y0, y1), plus R's frame."""
    out = [r for r in (_clip(x, R) for x in closed_rects(mask)) if r is not None]
    out += _frame((mask.c_lo, mask.c_hi, mask.y_lo, mask.y_hi), R)
    return out


# --- the traced outline --------------------------------------------------------------------
def _edges(is_open, c_closed, c_open, ys, step):
    """Per edge, the innermost closed point between a closed and an open sample, scanned
    at `step`: rock is not monotone, so a bisection can stop at an outer flip."""
    c_closed, c_open = np.array(c_closed, float), np.array(c_open, float)
    n = max(2, int(math.ceil(np.abs(c_open - c_closed).max() / step)))
    fr = np.arange(1, n) / float(n)
    cs = c_closed[:, None] + (c_open - c_closed)[:, None] * fr[None, :]
    nh = ys.shape[1]
    pc = np.repeat(cs.ravel(), nh)
    py = np.tile(ys, (1, n - 1)).ravel()
    closed = ~is_open(pc, py).reshape(len(cs), n - 1, nh).all(axis=2)
    last = np.where(closed.any(axis=1), (n - 2) - np.argmax(closed[:, ::-1], axis=1), -1)
    return c_closed + (c_open - c_closed) * (last + 1.5) / n


def open_points(m, room_index, mask, cs, ys, cache=None, depth=DEPTH):
    """The mask's own point test at arbitrary (cross, height) pairs on its plane."""
    tri, rtri = _geometry(m, room_index, cache)
    return _classify(tri, rtri, mask.axis, mask.sign, mask.plane, np.asarray(cs, float),
                     np.asarray(ys, float), depth, bool(m['sections'][room_index]['b2']))[1]


def trace(m, room_index, mask, band=BAND, cache=None, depth=DEPTH):
    """Fill mask.bands: the open interval(s) of every row band with edges scanned on the
    point test, the bottom band starting at the hole's floor. Full bands carry []."""
    def is_open(cs, ys):
        return open_points(m, room_index, mask, cs, ys, cache, depth)
    c_lo, c_hi, y_lo, y_hi = mask.c_lo, mask.c_hi, mask.y_lo, mask.y_hi
    rows = np.nonzero(mask.grid.any(axis=1))[0]
    if not len(rows):
        mask.bands = [(y_lo, y_hi, [])]
        return mask
    # the floor: bisect along y under the lowest open cells' middle column
    j0 = rows[0]
    cols = np.nonzero(mask.grid[j0])[0]
    c_mid = c_lo + int(round((cols.min() + cols.max() + 1) / 2.0)) * mask.dc
    c_mid = min(max(c_mid, c_lo + EDGE), c_hi - EDGE)
    lo_y, hi_y = y_lo + EDGE, max(y_lo + EDGE, y_lo + j0 * mask.dy)
    if is_open([c_mid], [lo_y])[0]:
        h0 = y_lo
    else:
        for _ in range(int(math.ceil(math.log2(max((hi_y - lo_y) / BISECT, 2.0))))):
            mid = (lo_y + hi_y) / 2.0
            if is_open([c_mid], [mid])[0]:
                hi_y = mid
            else:
                lo_y = mid
        h0 = (lo_y + hi_y) / 2.0
    edges = [y_lo] if h0 <= y_lo + 1e-3 else [y_lo, h0]
    y = edges[-1]
    while y_hi - y > band * 1.5:
        y += band
        edges.append(y)
    edges.append(y_hi)
    bands = [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)]
    if h0 > y_lo + 1e-3:
        bands[0] = (y_lo, h0, 'full')
    # sample every band at three heights across the lattice columns
    cs = np.clip(np.linspace(c_lo, c_hi, mask.nc + 1), c_lo + EDGE, c_hi - EDGE)
    live = [(b[0], b[1]) for b in bands if len(b) == 2]
    hs = np.array([(y0 + 0.01, (y0 + y1) / 2.0, y1 - 0.01) for y0, y1 in live])
    n_b, n_c = len(live), len(cs)
    if n_b:
        pc = np.tile(cs, n_b * 3)
        py = np.repeat(hs.ravel(), n_c)
        ok = is_open(pc, py).reshape(n_b, 3, n_c).all(axis=1)
    out = []
    queue = []                 # (band, side, c_closed, c_open)
    runs = []
    for k in range(n_b):
        r = _runs(ok[k])
        runs.append(r)
        for i0, i1 in r:
            if i0 > 0:
                queue.append((k, 0, cs[i0 - 1], cs[i0]))
            if i1 < n_c:
                queue.append((k, 1, cs[i1], cs[i1 - 1]))
    found = {}
    if queue:
        ks = np.array([q[0] for q in queue])
        e = _edges(is_open, [q[2] for q in queue], [q[3] for q in queue], hs[ks], BISECT)
        for q, v in zip(queue, e):
            found[(q[0], q[1], q[2])] = float(v)
    # the hole's cross extent per grid row, so a pocket beside it stays capped
    extent = {}
    for j in rows:
        cols = np.nonzero(mask.grid[j])[0]
        extent[j] = (c_lo + cols.min() * mask.dc, c_lo + (cols.max() + 1) * mask.dc)
    k = 0
    for b in bands:
        if len(b) == 3:
            out.append((b[0], b[1], []))
            continue
        y0, y1 = b
        jc = int((((y0 + y1) / 2.0) - y_lo) / mask.dy)
        lo_c, hi_c = extent[min(rows, key=lambda j: abs(j - jc))]
        ivals = []
        for i0, i1 in runs[k]:
            left = c_lo if i0 == 0 else found[(k, 0, cs[i0 - 1])] + OVERLAP
            right = c_hi if i1 == n_c else found[(k, 1, cs[i1])] - OVERLAP
            if right - left >= MIN_GAP and right > lo_c and left < hi_c:
                ivals.append((left, right))
        out.append((y0, y1, ivals))
        k += 1
    mask.bands = out
    return mask


def _segments(bands, R, eps=1e-4):
    """Per band clipped to R: (y0, y1, [(c0, c1), ...]) of the capped stretches."""
    out = []
    for y0, y1, ivals in bands:
        y0, y1 = max(y0, R[2]), min(y1, R[3])
        if y1 - y0 <= eps:
            continue
        segs, c = [], R[0]
        for left, right in sorted(ivals):
            if left - c > eps:
                segs.append((c, min(left, R[1])))
            c = max(c, right)
        if R[1] - c > eps:
            segs.append((c, R[1]))
        segs = [(a, b) for a, b in segs if b - a > eps and a < R[1] and b > R[0]]
        segs = [(max(a, R[0]), min(b, R[1])) for a, b in segs]
        out.append((y0, y1, segs))
    return out


def _strip(bottom, top, y0, y1):
    """Triangles between two horizontal chains (ascending c lists on y0 and y1)."""
    tris, ib, it = [], 0, 0
    while ib + 1 < len(bottom) or it + 1 < len(top):
        b, t = (bottom[ib], y0), (top[it], y1)
        if it + 1 >= len(top) or (ib + 1 < len(bottom) and bottom[ib + 1] <= top[it + 1]):
            tris.append((b, (bottom[ib + 1], y0), t))
            ib += 1
        else:
            tris.append((b, (top[it + 1], y1), t))
            it += 1
    return tris


def band_triangles(bands, R):
    """2D (c, y) triangles of the traced cap inside R. Each capped stretch is one strip
    whose chains carry the neighbouring bands' stretch ends, so no edge has a T-junction."""
    segs = _segments(bands, R)
    tris = []
    for k, (y0, y1, ss) in enumerate(segs):
        below = [c for a, b in segs[k - 1][2] for c in (a, b)] if k > 0 else []
        above = [c for a, b in segs[k + 1][2] for c in (a, b)] if k + 1 < len(segs) else []
        for a, b in ss:
            bot = [a] + sorted(c for c in set(below) if a + 1e-4 < c < b - 1e-4) + [b]
            top = [a] + sorted(c for c in set(above) if a + 1e-4 < c < b - 1e-4) + [b]
            tris += _strip(bot, top, y0, y1)
    return tris


def rect_triangles(rects):
    return [t for c0, c1, y0, y1 in rects
            for t in (((c0, y0), (c1, y0), (c1, y1)), ((c0, y0), (c1, y1), (c0, y1)))]


# --- geometry ------------------------------------------------------------------------------
def _quad(rect, axis, plane, normal_sign):
    """Four corners CCW as seen from the normal's side."""
    a, c = AX[axis], CROSS[axis]
    c0, c1, y0, y1 = rect
    pts = []
    for cv, yv in ((c0, y0), (c1, y0), (c1, y1), (c0, y1)):
        p = [0.0, yv, 0.0]
        p[a], p[c] = plane, cv
        pts.append(p)
    n = np.cross(np.subtract(pts[1], pts[0]), np.subtract(pts[2], pts[0]))
    if n[a] * normal_sign < 0:
        pts.reverse()
    return pts


def _wall_mesh(rects, axis, plane, normal_sign, uv_scale):
    """(pos, nrm, uv, idx) for the quads, uvs planar in world units over uv_scale."""
    return _plane_mesh(rect_triangles(rects), axis, plane, normal_sign, uv_scale)


def _plane_mesh(tris2d, axis, plane, normal_sign, uv_scale):
    """(pos, nrm, uv, idx) for (c, y) triangles on the plane, wound to face normal_sign;
    vertices shared within a triangle strip by exact position."""
    a, c = AX[axis], CROSS[axis]
    pos, idx, index = [], [], {}
    for tri in tris2d:
        pts = []
        for cv, yv in tri:
            p = [0.0, float(yv), 0.0]
            p[a], p[c] = plane, float(cv)
            pts.append(tuple(p))
        n = np.cross(np.subtract(pts[1], pts[0]), np.subtract(pts[2], pts[0]))
        if abs(n[a]) < 1e-12:
            continue
        if n[a] * normal_sign < 0:
            pts.reverse()
        ids = []
        for p in pts:
            if p not in index:
                index[p] = len(pos)
                pos.append(p)
            ids.append(index[p])
        idx.append(tuple(ids))
    pos = np.asarray(pos, F32).reshape(-1, 3)
    nrm = np.zeros_like(pos)
    nrm[:, a] = normal_sign
    uv = np.stack([pos[:, c], pos[:, 1]], axis=1) / uv_scale
    return pos, nrm, uv.astype(F32), np.asarray(idx, np.int64).reshape(-1, 3)


def _inline_meshes(sec):
    return [me for me in sec['meshes'] if not (me['flags'] & 0x20000)]


def _append_to_mesh(me, pos, nrm, uv, idx, source=None, lmuv=None):
    """Grow one mesh record by the wall. Geometric slots are computed; every other
    slot (colours, lightmap uv, unknowns) copies the nearest existing vertex's, or
    `source` (a point) 's nearest vertex for all of them; lmuv overrides slot 12."""
    g = geom.decode_mesh(me['pkt'])
    n0, k = g['nverts'], len(pos)
    if n0 + k > MAX_VERTS:
        raise ValueError('mesh %r would carry %d vertices > %d'
                         % (geom.name_str(me['name']), n0 + k, MAX_VERTS))
    old = np.asarray(g['vec'][geom.S_POS], np.float64)
    if source is None:
        near = np.argmin(((pos[:, None, :].astype(np.float64) - old[None, :, :]) ** 2).sum(axis=2),
                         axis=1)
    else:
        near = np.full(k, int(np.argmin(((old - np.asarray(source, np.float64)) ** 2).sum(axis=1))))
    tan, bin_ = gsec.tangents(pos, nrm, uv, idx)
    computed = {geom.S_POS: pos, geom.S_NORMAL: nrm, geom.S_TANGENT: tan, geom.S_BINORMAL: bin_}
    for slot, vals in g['vec'].items():
        src = computed.get(slot)
        vals += ([tuple(float(x) for x in p) for p in src] if src is not None
                 else [vals[i] for i in near])
    for slot, vals in g['uv'].items():
        vals += ([tuple(float(x) for x in p) for p in uv] if slot in (2, 3)
                 else [tuple(lmuv)] * k if slot == 12 and lmuv is not None
                 else [vals[i] for i in near])
    for table in ('color', 'raw'):
        for slot, vals in g[table].items():
            vals += [vals[i] for i in near]
    if g['pad']:
        g['pad'] += [g['pad'][i] for i in near]
    g['tris'] += [tuple(int(x) + n0 for x in t) for t in idx]
    g['nverts'] = n0 + k
    me['pkt'] = geom.encode_mesh(g)
    return k


def _grow_bvt(sec, pos, idx, surf=None, flags=0, rebuild=True, inflate=1.0):
    """Both windings of the wall into the section's BVT: rebuilt whole for a generated
    section, or grafted as a sibling subtree so a shipped tree keeps its layout."""
    verts, tris, surfs, tflags = bvt_collect(sec['bvt'])
    if surf is None:
        surf = max(set(surfs), key=surfs.count) if surfs else 0
    base = len(verts)
    new_v = [tuple(float(x) for x in p) for p in pos]
    new_t = [tuple(int(x) + base for x in t) for t in idx]
    new_t += [(t[0], t[2], t[1]) for t in new_t]
    if rebuild or sec['bvt'] is None:
        root, arena = bvt_build(verts + new_v, tris + new_t, surfs + [surf] * len(new_t),
                                tflags + [flags] * len(new_t), inflate=inflate)
    else:
        sub, _ = bvt_build(new_v, [tuple(i - base for i in t) for t in new_t],
                           [surf] * len(new_t), [flags] * len(new_t), inflate=PLUG_INFLATE)
        old = sec['bvt']
        root = {'bbox': struct.pack('<6f', *union_bbox([old['bbox'], sub['bbox']])),
                'kids': [old, sub]}
        arena = 0x38 + _arena(root)
    sec['bvt'], sec['bvtflag'] = root, arena
    return len(new_t)


def _arena(n):
    if 'kids' in n:
        return 0x70 + _arena(n['kids'][0]) + _arena(n['kids'][1])
    return 0x18 + len(n['verts']) + len(n['tris'])


def _relayout(m):
    if 'origdatasize' not in m:
        geom.capture(m)
    geom.relayout(m)


# --- the cap ---------------------------------------------------------------------------------
def corridor_face(m, connector_index, mask):
    """(end coordinate, R) of the corridor's end face that the mask's plane sits on."""
    mb = link._mesh_bbox(m['sections'][connector_index])
    a, c = AX[mask.axis], CROSS[mask.axis]
    end = mb[a] if mask.sign > 0 else mb[a + 3]
    if abs(end - mask.plane) > FACE_TOL:
        raise ValueError('mask plane %.2f is %.2f from the corridor end face at %.2f'
                         % (mask.plane, abs(end - mask.plane), end))
    return end, (mb[c], mb[c + 3], mb[1], mb[4])


def cap_mouth(m, connector_index, room_index, mask, uv_scale=8.0, method='trace',
              band=BAND):
    """A wall on the corridor's end face around the mouth, in the corridor's own mesh
    and BVT: 'trace' follows the bisected outline in row bands, 'cells' caps whole cells."""
    if method not in METHODS:
        raise ValueError('method must be one of %s, not %r' % (METHODS, method))
    sec = m['sections'][connector_index]
    meshes = _inline_meshes(sec)
    if not meshes:
        raise ValueError('section %d has no inline mesh to cap' % connector_index)
    end, R = corridor_face(m, connector_index, mask)
    if method == 'trace':
        if mask.bands is None:
            trace(m, room_index, mask, band)
        tris = band_triangles(mask.bands, R)
        tris += rect_triangles(_frame((mask.c_lo, mask.c_hi, mask.y_lo, mask.y_hi), R))
        pieces = sum(len(ss) for _y0, _y1, ss in _segments(mask.bands, R))
    else:
        rects = cap_rects(mask, R)
        tris, pieces = rect_triangles(rects), len(rects)
    report = {'method': method, 'rects': pieces, 'verts': 0, 'tris': 0,
              'open_cells': mask.n_open, 'closed_cells': int((~mask.grid).sum()),
              'face': R, 'plane': end}
    if not tris:
        return report
    pos, nrm, uv, idx = _plane_mesh(tris, mask.axis, end, mask.sign, uv_scale)
    report['verts'] = _append_to_mesh(meshes[0], pos, nrm, uv, idx)
    report['tris'] = _grow_bvt(sec, pos, idx, rebuild=True)
    _relayout(m)
    return report


def mouths(m, connector_index, rooms, cell=0.5):
    """A mask per room over the corridor's end face on that room's bbox face."""
    mb = link._mesh_bbox(m['sections'][connector_index])
    out = []
    for room in rooms:
        bb = struct.unpack('<6f', m['sections'][room]['bbox'])
        found = None
        for axis, a in AX.items():
            for end, face, sign in ((mb[a], bb[a + 3], 1), (mb[a + 3], bb[a], -1)):
                if abs(end - face) <= FACE_TOL and (found is None or
                                                    abs(end - face) < found[0]):
                    found = (abs(end - face), axis, end, sign)
        if found is None:
            raise ValueError('connector %d has no end face on a bbox face of section %d'
                             % (connector_index, room))
        _d, axis, plane, sign = found
        out.append(mouth_mask(m, room, plane, axis, sign, mb[:3], mb[3:], cell))
    return out


def cap_connector(m, connector_index, a_index, b_index, cell=0.5, uv_scale=8.0,
                  method='trace'):
    """Both mouths of an existing box corridor. Returns (masks, reports)."""
    masks = mouths(m, connector_index, (a_index, b_index), cell)
    reports = [cap_mouth(m, connector_index, r, mk, uv_scale, method)
               for r, mk in zip((a_index, b_index), masks)]
    return masks, reports


# --- the plug ----------------------------------------------------------------------------------
def _nearest_mesh(sec, point):
    best = None
    for me in _inline_meshes(sec):
        bb = struct.unpack('<6f', me['pkt']['bbox'])
        d = sum(max(bb[k] - point[k], 0.0, point[k] - bb[k + 3]) ** 2 for k in range(3))
        if best is None or d < best[0]:
            best = (d, me)
    return None if best is None else best[1]


def _nearest_surf(sec, point):
    verts, tris, surfs, _f = bvt_collect(sec['bvt'])
    if not tris:
        return 0
    V = np.asarray(verts, np.float64)
    cen = V[np.asarray(tris)].mean(axis=1)
    return surfs[int(np.argmin(((cen - np.asarray(point)) ** 2).sum(axis=1)))]


def ambient_texel(m, room_index, library, margin=1):
    """(u, v) of the centre of a lit texel of the room's own tiles, its colour nearest
    room_ambient and its neighbours lit too, or None without readable tiles."""
    s = m['sections'][room_index]
    imgs = []
    for k in range(3):
        name = tp.slot_name(s['names3'][k * tp.SLOT:(k + 1) * tp.SLOT])
        blob = library.read(tp.pod_path(name)) if library is not None and name else None
        if blob:
            imgs.append(textures.decode(blob)[2][:, :, :3].astype(np.float64))
    if not imgs or len({im.shape for im in imgs}) != 1:
        return None
    mean = np.mean(imgs, axis=0)
    lit = np.all([im.max(axis=2) > 0 for im in imgs], axis=0)
    h, w = lit.shape
    ok = lit.copy()
    for dy in range(-margin, margin + 1):
        for dx in range(-margin, margin + 1):
            ok &= np.roll(np.roll(lit, dy, axis=0), dx, axis=1)
    ok[:margin, :] = ok[-margin:, :] = ok[:, :margin] = ok[:, -margin:] = False
    if not ok.any():
        ok = lit
    if not ok.any():
        return None
    target = np.asarray(lighting.room_ambient(m, room_index, library), np.float64)
    d = np.where(ok, ((mean - target[None, None, :]) ** 2).sum(axis=2), np.inf)
    y, x = np.unravel_index(int(np.argmin(d)), d.shape)
    return ((x + 0.5) / w, (y + 0.5) / h)


def plug_opening(m, room_index, quad, uv_scale=8.0, library=None):
    """A wall across an opening that leads nowhere, facing into the room, in the mesh
    nearest the opening and the room's BVT, one flat colour. Returns what was added."""
    sec = m['sections'][room_index]
    f = link.opening_face(m, room_index, quad)
    axis, sign = f['face']
    if axis not in AX:
        raise NotImplementedError('a %s-axis opening is a hatch, not a doorway' % axis)
    a, c = AX[axis], CROSS[axis]
    bb = struct.unpack('<6f', sec['bbox'])
    plane = min(max(f['centre'][a], bb[a] + PLUG_INSET), bb[a + 3] - PLUG_INSET)
    # a hand-authored quad can overhang the room's box; nothing lives out there
    box = (bb[c] + PLUG_INSET, bb[c + 3] - PLUG_INSET, bb[1] + PLUG_INSET, bb[4] - PLUG_INSET)
    rect = _clip((f['lo'][c], f['hi'][c], f['lo'][1], f['hi'][1]), box)
    if rect is None:
        raise ValueError('opening at %s lies outside section %d' % (f['centre'], room_index))
    me = _nearest_mesh(sec, f['centre'])
    if me is None:
        raise ValueError('section %d has no inline mesh to plug into' % room_index)
    pos, nrm, uv, idx = _wall_mesh([rect], axis, plane, -sign, uv_scale)
    # every plug vertex shares one lightmap texel and one colour: a streak-free wall
    texel = ambient_texel(m, room_index, library)
    report = {'plane': plane, 'rect': rect, 'mesh': geom.name_str(me['name']), 'texel': texel,
              'verts': _append_to_mesh(me, pos, nrm, uv, idx, source=f['centre'], lmuv=texel)}
    surf = _nearest_surf(sec, f['centre']) if sec['bvt'] is not None else 0
    report['tris'] = _grow_bvt(sec, pos, idx, surf=surf, rebuild=sec['bvt'] is None)
    _relayout(m)
    return report


# --- the check -----------------------------------------------------------------------------
def _doorway_geometry(m, rec):
    """(lo, hi, axis, plane) of a record; axis None for a thick one (two unrelated quads)
    or a skewed one (a 45 degree wall), where a grid on one plane says nothing."""
    corners = link._corners(rec['verts'])
    lo = [min(p[k] for p in corners) for k in range(3)]
    hi = [max(p[k] for p in corners) for k in range(3)]
    ext = [hi[k] - lo[k] for k in range(3)]
    ax = min(range(3), key=lambda k: ext[k])
    axis = 'xyz'[ax]
    if ext[ax] > THICK_MAX:
        axis = None
    elif axis in CROSS:
        c = CROSS[axis]
        mid = (lo[c] + hi[c]) / 2.0
        near = [p[ax] for p in corners if p[c] < mid]
        far = [p[ax] for p in corners if p[c] >= mid]
        if near and far and abs(sum(near) / len(near) - sum(far) / len(far)) > SKEW_TOL:
            axis = None
    return lo, hi, axis, (lo[ax] + hi[ax]) / 2.0


def doorway_masks(m, k, cell=1.0, cache=None):
    """The two masks of doorway k, each seen from its own section. None for a hatch,
    a skewed quad, or a record that is a line rather than a rectangle."""
    rec = link.read_doorways(m)[k]
    lo, hi, axis, plane = _doorway_geometry(m, rec)
    if axis not in AX:
        return None
    a = AX[axis]
    if hi[1] - lo[1] < MIN_EXTENT or hi[CROSS[axis]] - lo[CROSS[axis]] < MIN_EXTENT:
        return None
    # a record's two quad copies can sit a wall's thickness apart: each section is
    # read from the copy on its own side, or its probe lands inside the other room
    planes = sorted(set(round(p[a], 1) for p in link._corners(rec['verts'])))
    out = []
    for s in (rec['a'], rec['b']):
        sign = _side(m, s, axis, plane, lo, hi, cache)
        own = planes[0] if sign > 0 else planes[-1]
        out.append(mouth_mask(m, s, own, axis, sign, lo, hi, cell, cache=cache))
    return out


def _side(m, s, axis, plane, lo, hi, cache=None, reach=20.0):
    """The mask sign of section s (+1: it sits below the plane), read from its own
    geometry near the rectangle: two sections often share a box, so centres mislead.
    Area, not triangle count: a loft's finely cut end collar would outvote its body."""
    a, c = AX[axis], CROSS[axis]
    n_lo = n_hi = 0.0
    for tri in _geometry(m, s, cache):
        if tri is None:
            continue
        cen = (tri[0] + tri[1] + tri[2]) / 3.0
        area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]), axis=1)
        near = ((cen[:, c] >= lo[c] - DEPTH) & (cen[:, c] <= hi[c] + DEPTH)
                & (cen[:, 1] >= lo[1] - DEPTH) & (cen[:, 1] <= hi[1] + DEPTH)
                & (np.abs(cen[:, a] - plane) <= reach))
        n_lo += float(area[near & (cen[:, a] < plane)].sum())
        n_hi += float(area[near & (cen[:, a] > plane)].sum())
    if n_lo == n_hi:
        bb = struct.unpack('<6f', m['sections'][s]['bbox'])
        return 1 if plane >= (bb[a] + bb[a + 3]) / 2.0 else -1
    return 1 if n_lo > n_hi else -1


def check_caps(m, cell=1.0):
    """Doorways whose rectangle opens into one section where the other, indoor, has
    nothing around at all: a hole to the void. An enclosed pocket passes. [] = clean."""
    bad = []
    cache = {}
    for k, rec in enumerate(link.read_doorways(m)):
        masks = doorway_masks(m, k, cell, cache)
        if masks is None:
            continue
        ma, mb = masks
        # an empty outdoor side (b2, the outdoor flag) shows sky, not the void
        out_a, out_b = (bool(m['sections'][s]['b2']) for s in (rec['a'], rec['b']))
        holes = np.zeros(ma.grid.shape, bool)
        if not out_b:
            holes |= _cells(ma.pt_open & mb.pt_empty)
        if not out_a:
            holes |= _cells(mb.pt_open & ma.pt_empty)
        n = int(holes.sum())
        if not n:
            continue
        j, i = [int(v) for v in np.argwhere(holes)[0]]
        c0, c1, y0, y1 = ma.cell(i, j)
        p = ma.point((c0 + c1) / 2.0, (y0 + y1) / 2.0)
        names = [geom.name_str(m['sections'][s]['name']) for s in (rec['a'], rec['b'])]
        bad.append('doorway %d (%s | %s): %d of %d cells open into the void, e.g. '
                   '(%.2f, %.2f, %.2f)' % (k, names[0], names[1], n, ma.grid.size, *p))
    return bad


# --- CLI ----------------------------------------------------------------------------------------
def _load(path):
    with open(path, 'rb') as fh:
        return bst.parse(fh.read())


def cmd_check(a):
    rc = 0
    for f in a.files:
        bad = check_caps(_load(f), a.cell)
        print('%s %s: %d problems' % ('OK  ' if not bad else 'FAIL', f, len(bad)))
        for b in bad:
            print('    ' + b)
        rc |= bool(bad)
    return rc


def cmd_mask(a):
    m = _load(a.file)
    masks = doorway_masks(m, a.doorway, a.cell)
    if masks is None:
        print('doorway %d is not a plane doorway (a hatch, a line, skewed or thick)' % a.doorway)
        return 1
    rec = link.read_doorways(m)[a.doorway]
    for s, mk in zip((rec['a'], rec['b']), masks):
        print('section %d %s: %s plane %.2f sign %+d, %d of %d cells open'
              % (s, geom.name_str(m['sections'][s]['name']), mk.axis, mk.plane, mk.sign,
                 mk.n_open, mk.grid.size))
        for row in mk.rows():
            print('    ' + row)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('check', help='doorways that open into the void')
    p.add_argument('files', nargs='+')
    p.add_argument('--cell', type=float, default=1.0)
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser('mask', help="one doorway's two mouth masks as ASCII")
    p.add_argument('file')
    p.add_argument('doorway', type=int)
    p.add_argument('--cell', type=float, default=0.5)
    p.set_defaults(fn=cmd_mask)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())

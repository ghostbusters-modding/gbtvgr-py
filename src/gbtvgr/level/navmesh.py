"""Navmesh generation over marked-walkable geometry, and the checks a generated
mesh has to pass before it ships.

A cell is a square of the grid whose four corners all land on walkable floor,
with nothing solid in the head room above it. Links follow the HarborDocks
rule: neighbours agree on both shared corner heights or there is no link.
"""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import itertools
import math
import struct

import numpy as np

from gbtvgr.sets import geom as ggeom

from .mathutil import F32, transform_points

EDGE_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
EDGE_CORNERS = {(-1, 0): (0, 1), (0, 1): (1, 2), (1, 0): (3, 2), (0, -1): (0, 3)}
CORNER_OFFSETS = ((0, 0), (0, 1), (1, 1), (1, 0))
GROUP_TAG = 0x48415554
HEAD_CLEAR = 0.25            # a hit this close above a corner is the floor itself


class VerticalCaster:
    """Vertical ray casts against a triangle soup, bucketed on the XZ plane.
    Steep triangles never register: a wall has no height to give."""

    def __init__(self, verts, tris, walkable=None, bucket=None):
        self.v = np.asarray(verts, np.float64).reshape(-1, 3)
        self.t = np.asarray(tris, np.int64).reshape(-1, 3)
        self.walkable = (np.ones(len(self.t), bool) if walkable is None
                         else np.asarray(walkable, bool).reshape(-1))
        self.p = self.v[self.t] if len(self.t) else np.zeros((0, 3, 3))
        self.lo = self.p.min(axis=1) if len(self.p) else np.zeros((0, 3))
        self.hi = self.p.max(axis=1) if len(self.p) else np.zeros((0, 3))
        if len(self.p):
            ext = self.hi.max(axis=0) - self.lo.min(axis=0)
            self.bucket = bucket or max(4.0, float(max(ext[0], ext[2])) / 64.0)
            self.origin = self.lo.min(axis=0)[[0, 2]]
        else:
            self.bucket = bucket or 4.0
            self.origin = np.zeros(2)
        self.buckets = {}
        if len(self.p):
            ilo = np.floor((self.lo[:, [0, 2]] - self.origin) / self.bucket).astype(np.int64)
            ihi = np.floor((self.hi[:, [0, 2]] - self.origin) / self.bucket).astype(np.int64)
            for ti in range(len(self.p)):
                for bi in range(ilo[ti, 0], ihi[ti, 0] + 1):
                    for bj in range(ilo[ti, 1], ihi[ti, 1] + 1):
                        self.buckets.setdefault((bi, bj), []).append(ti)
            self.buckets = {k: np.array(v, np.int64) for k, v in self.buckets.items()}

    def hits(self, x, z, eps=1e-4):
        """(ys, walkable flags, triangle indices) of every surface over (x, z)."""
        bi = int(math.floor((x - self.origin[0]) / self.bucket))
        bj = int(math.floor((z - self.origin[1]) / self.bucket))
        cand = self.buckets.get((bi, bj))
        empty = (np.zeros(0), np.zeros(0, bool), np.zeros(0, np.int64))
        if cand is None:
            return empty
        p = self.p[cand]
        ax, ay, az = p[:, 0, 0], p[:, 0, 1], p[:, 0, 2]
        bx, by, bz = p[:, 1, 0], p[:, 1, 1], p[:, 1, 2]
        cx, cy, cz = p[:, 2, 0], p[:, 2, 1], p[:, 2, 2]
        d = (bx - ax) * (cz - az) - (cx - ax) * (bz - az)
        flat = np.abs(d) > 1e-9
        if not flat.any():
            return empty
        with np.errstate(divide='ignore', invalid='ignore'):
            w0 = ((bx - x) * (cz - z) - (cx - x) * (bz - z)) / d
            w1 = ((cx - x) * (az - z) - (ax - x) * (cz - z)) / d
            w2 = 1.0 - w0 - w1
            inside = flat & (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps)
            if not inside.any():
                return empty
            y = (w0 * ay + w1 * by + w2 * cy)[inside]
        idx = cand[inside]
        order = np.argsort(-y)
        return y[order], self.walkable[idx[order]], idx[order]


class BoxGrid:
    """Triangle bounding boxes bucketed on XZ, for 'anything solid in this column'."""

    def __init__(self, verts, tris, bucket=8.0):
        v = np.asarray(verts, np.float64).reshape(-1, 3)
        t = np.asarray(tris, np.int64).reshape(-1, 3)
        p = v[t] if len(t) else np.zeros((0, 3, 3))
        self.lo = p.min(axis=1) if len(p) else np.zeros((0, 3))
        self.hi = p.max(axis=1) if len(p) else np.zeros((0, 3))
        self.bucket = bucket
        self.origin = self.lo.min(axis=0)[[0, 2]] if len(p) else np.zeros(2)
        self.buckets = {}
        if len(p):
            ilo = np.floor((self.lo[:, [0, 2]] - self.origin) / bucket).astype(np.int64)
            ihi = np.floor((self.hi[:, [0, 2]] - self.origin) / bucket).astype(np.int64)
            for ti in range(len(p)):
                for bi in range(ilo[ti, 0], ihi[ti, 0] + 1):
                    for bj in range(ilo[ti, 1], ihi[ti, 1] + 1):
                        self.buckets.setdefault((bi, bj), []).append(ti)
            self.buckets = {k: np.array(v, np.int64) for k, v in self.buckets.items()}

    def overlaps(self, x0, z0, x1, z1, y0, y1):
        """True when any box meets the column [x0..x1, z0..z1] between y0 and y1."""
        if not self.buckets:
            return False
        b0 = np.floor((np.array([x0, z0]) - self.origin) / self.bucket).astype(np.int64)
        b1 = np.floor((np.array([x1, z1]) - self.origin) / self.bucket).astype(np.int64)
        cand = []
        for bi in range(b0[0], b1[0] + 1):
            for bj in range(b0[1], b1[1] + 1):
                c = self.buckets.get((bi, bj))
                if c is not None:
                    cand.append(c)
        if not cand:
            return False
        c = np.unique(np.concatenate(cand))
        lo, hi = self.lo[c], self.hi[c]
        ok = (lo[:, 0] < x1) & (hi[:, 0] > x0) & (lo[:, 2] < z1) & (hi[:, 2] > z0) \
            & (lo[:, 1] < y1) & (hi[:, 1] > y0)
        return bool(ok.any())


class WalkableWorld:
    """The triangles a navmesh is built over: walkable floor, solid blockers,
    and the boxes of placed props nothing may path through."""

    def __init__(self, verts, tris, walkable=None, exclusions=(), facing_up=None):
        self.verts = np.asarray(verts, F32).reshape(-1, 3)
        self.tris = np.asarray(tris, np.int64).reshape(-1, 3)
        self.walkable = (np.ones(len(self.tris), bool) if walkable is None
                         else np.asarray(walkable, bool).reshape(-1))
        # a ceiling's back is not a floor, but it is no blocker either: headroom covers it
        self.floor = self.walkable if facing_up is None else self.walkable & np.asarray(facing_up, bool)
        self.exclusions = [(np.asarray(lo, np.float64), np.asarray(hi, np.float64))
                           for lo, hi in exclusions]
        self._caster = None
        self._blockers = None

    @classmethod
    def from_document(cls, doc, library=None):
        vs, ts, walk, ups, base = [], [], [], [], 0
        for n in doc.mesh_nodes():
            geo = n.mesh if n.kind == 'mesh' else (n.terrain.mesh() if n.terrain else None)
            if geo is None or len(geo.faces) == 0:
                continue
            collide = bool(n.props.get('collide', True))
            walkable = collide and bool(n.props.get('walkable', True))
            if not collide:
                continue
            v = transform_points(n.world_matrix(), geo.verts)
            f = np.asarray(geo.faces, np.int64)
            vs.append(v)
            ts.append(f + base)
            walk.append(np.full(len(f), walkable, bool))
            ups.append(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])[:, 1] > 0)
            base += len(v)
        exclusions = []
        if library is not None:
            for n in doc.nodes('component') + doc.nodes('actor'):
                if not n.props.get('solid', False):
                    continue
                ref = None
                if n.kind == 'component':
                    cd = doc.components.get(n.component_id)
                    ref = n.props.get('mesh') or (cd.mesh if cd else None)
                else:
                    ref = n.props.get('model')
                if not ref:
                    continue
                try:
                    model = library.read_model(ref)
                except Exception:  # noqa: BLE001
                    model = None
                if model is None:
                    continue
                lo, hi = model.bbox
                corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                    for z in (lo[2], hi[2])], F32)
                w = transform_points(n.world_matrix(), corners)
                exclusions.append((w.min(axis=0), w.max(axis=0)))
        if not vs:
            return cls(np.zeros((0, 3), F32), np.zeros((0, 3), np.int64), None, exclusions)
        return cls(np.concatenate(vs), np.concatenate(ts), np.concatenate(walk), exclusions,
                   np.concatenate(ups))

    @property
    def caster(self):
        if self._caster is None:
            self._caster = VerticalCaster(self.verts, self.tris, self.floor)
        return self._caster

    @property
    def blockers(self):
        if self._blockers is None:
            solid = ~self.walkable
            self._blockers = BoxGrid(self.verts, self.tris[solid])
        return self._blockers

    def walkable_bounds(self):
        w = self.tris[self.floor]
        if not len(w):
            return None
        p = self.verts[w].reshape(-1, 3)
        return p.min(axis=0), p.max(axis=0)


class NavResult:
    def __init__(self, nav, cells, verts, nodes, islands, params, report=''):
        self.nav = nav
        self.cells = cells
        self.verts = verts
        self.nodes = nodes
        self.islands = islands
        self.params = params
        self.report = report
        self.world = None

    @property
    def empty(self):
        return not self.nodes


def _empty_nav():
    return {'empty': True, 'f': 2, 'nverts': 0, 'nnodes': 0, 'nextra': 0, 'nparts': 0}


def build_navmesh(world, cell=6.0, max_step=2.0, max_slope_deg=40.0, agent_height=8.0,
                  origin_hint=None):
    params = {'cell': cell, 'max_step': max_step, 'max_slope_deg': max_slope_deg,
              'agent_height': agent_height}
    bounds = world.walkable_bounds()
    if bounds is None:
        r = NavResult(_empty_nav(), [], np.zeros((0, 3), F32), [], [], params,
                      'no walkable geometry')
        r.world = world
        return r
    lo, hi = bounds
    if origin_hint is not None:
        ox, oz = float(origin_hint[0]), float(origin_hint[1])
    else:
        ox = math.floor((lo[0] - cell) / cell) * cell
        oz = math.floor((lo[2] - cell) / cell) * cell
    nx = int(math.ceil((hi[0] - ox) / cell)) + 1
    nz = int(math.ceil((hi[2] - oz) / cell)) + 1
    tan_slope = math.tan(math.radians(max_slope_deg))
    # a step across a cell shows up as half its height at the edge midpoint
    cont_tol = max_step * 0.5
    caster = world.caster
    blockers = world.blockers
    sample_cache = {}

    def sample(fi, fj):
        """Hits at fractional grid coordinates, cached on the half grid."""
        k = (int(round(fi * 2)), int(round(fj * 2)))
        if k not in sample_cache:
            sample_cache[k] = caster.hits(ox + fi * cell, oz + fj * cell)
        return sample_cache[k]

    def corner(i, j):
        return sample(i, j)

    # (sample offset, corner indices whose mean the surface should pass through)
    CONTINUITY = (((0.5, 0.0), (0, 3)), ((0.0, 0.5), (0, 1)), ((1.0, 0.5), (3, 2)),
                  ((0.5, 1.0), (1, 2)), ((0.5, 0.5), (0, 1, 2, 3)))

    rejected = {'slope': 0, 'headroom': 0, 'blocked': 0, 'excluded': 0, 'broken': 0}
    cells = {}
    meta = []
    for i in range(nx):
        for j in range(nz):
            hits = [corner(i + di, j + dj) for di, dj in CORNER_OFFSETS]
            walk = [ys[wk] for ys, wk, _t in hits]
            if any(len(w) == 0 for w in walk):
                continue
            seen = set()
            for combo in itertools.islice(itertools.product(*[list(w) for w in walk]), 256):
                heights = [float(h) for h in combo]
                key = tuple(int(round(h * 8)) for h in heights)
                if key in seen:
                    continue
                seen.add(key)
                edges = ((0, 1), (1, 2), (2, 3), (3, 0))
                if max(abs(heights[a] - heights[b]) for a, b in edges) / cell > tan_slope:
                    rejected['slope'] += 1
                    continue
                whole = True
                for (fi, fj), idx in CONTINUITY:
                    est = sum(heights[c] for c in idx) / len(idx)
                    ys, wk, _t = sample(i + fi, j + fj)
                    cand = ys[wk]
                    if not len(cand) or np.abs(cand - est).min() >= cont_tol:
                        whole = False
                        break
                if not whole:
                    rejected['broken'] += 1
                    continue
                clear = True
                for k in range(4):
                    ys, _wk, _t = hits[k]
                    h = heights[k]
                    if np.any((ys > h + HEAD_CLEAR) & (ys <= h + agent_height)):
                        clear = False
                        break
                if not clear:
                    rejected['headroom'] += 1
                    continue
                x0, z0 = ox + i * cell, oz + j * cell
                hmin, hmax = min(heights), max(heights)
                if blockers.overlaps(x0, z0, x0 + cell, z0 + cell,
                                     hmin + HEAD_CLEAR, hmax + agent_height):
                    rejected['blocked'] += 1
                    continue
                excluded = False
                for elo, ehi in world.exclusions:
                    if (elo[0] < x0 + cell and ehi[0] > x0 and elo[2] < z0 + cell
                            and ehi[2] > z0 and elo[1] < hmax + agent_height
                            and ehi[1] > hmin - 0.5):
                        excluded = True
                        break
                if excluded:
                    rejected['excluded'] += 1
                    continue
                ck = (i, j, key[0])
                cells[ck] = len(meta)
                meta.append({'key': ck, 'i': i, 'j': j, 'heights': heights})
    if not meta:
        r = NavResult(_empty_nav(), [], np.zeros((0, 3), F32), [], [], params,
                      'no cell passed: %s' % rejected)
        r.world = world
        return r

    by_ij = {}
    for m in meta:
        by_ij.setdefault((m['i'], m['j']), []).append(m)
    vids, verts = {}, []

    def vid(i, j, h):
        k = (i, j, int(round(h * 8)))
        if k not in vids:
            vids[k] = len(verts)
            verts.append((ox + i * cell, h, oz + j * cell))
        return vids[k]

    nodes = []
    for m in meta:
        i, j, hs = m['i'], m['j'], m['heights']
        ring = [vid(i + di, j + dj, h) for (di, dj), h in zip(CORNER_OFFSETS, hs)]
        nb = []
        for d in EDGE_DIRS:
            a, b = EDGE_CORNERS[d]
            best, best_err = -1, None
            for other in by_ij.get((i + d[0], j + d[1]), []):
                oh = other['heights']
                # the neighbour's corners for the same shared edge, in its own ring
                mine = {CORNER_OFFSETS[a]: hs[a], CORNER_OFFSETS[b]: hs[b]}
                err = 0.0
                ok = True
                for (di, dj), h in mine.items():
                    oi, oj = di - d[0], dj - d[1]
                    ok2 = False
                    for c, (odi, odj) in enumerate(CORNER_OFFSETS):
                        if (odi, odj) == (oi, oj):
                            e = abs(oh[c] - h)
                            ok2 = e <= max_step
                            err += e
                    if not ok2:
                        ok = False
                        break
                if ok and (best_err is None or err < best_err):
                    best, best_err = cells[other['key']], err
            nb.append(best)
        pts = np.array([verts[v] for v in ring])
        c = pts.mean(axis=0)
        hdr = struct.pack('<I3f2f6f', 0, c[0], c[1], c[2], pts[:, 1].min() - 1.0,
                          pts[:, 1].max() + 30.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0)
        nodes.append({'key': m['key'], 'centre': (float(c[0]), float(c[1]), float(c[2])),
                      'nb': nb, 'ring': ring, 'heights': hs, 'hdr': hdr})
    islands = _islands(nodes)
    nav = {'empty': False, 'f': 2, 'nverts': len(verts), 'nnodes': len(nodes),
           'nextra': 1, 'nparts': 0,
           'verts': b''.join(struct.pack('<3f', *v) for v in verts),
           'nodes': [{'hdr': n['hdr'], 'v': struct.pack('<4I', *n['ring']),
                      'nb': struct.pack('<4i', *n['nb']),
                      'd0': struct.pack('<4I', 0, 0, 0, 0), 'f15c': struct.pack('<I', 0)}
                     for n in nodes],
           'extra': [{'a': struct.pack('<I', GROUP_TAG), 'b': struct.pack('<I', 3),
                      'c': struct.pack('<f', 1.0), 'rows': b''}],
           'parts': []}
    links = sum(1 for n in nodes for k in n['nb'] if k >= 0) // 2
    report = ('%d nodes, %d verts, %d links, %d island%s (largest %d); rejected: %s'
              % (len(nodes), len(verts), links, len(islands), '' if len(islands) == 1 else 's',
                 len(islands[0]) if islands else 0, rejected))
    r = NavResult(nav, [m['key'] for m in meta], np.array(verts, F32).reshape(-1, 3),
                  nodes, islands, params, report)
    r.world = world
    return r


def _islands(nodes):
    seen, out = set(), []
    for start in range(len(nodes)):
        if start in seen:
            continue
        comp, q = {start}, [start]
        while q:
            c = q.pop()
            for n in nodes[c]['nb']:
                if n >= 0 and n not in comp:
                    comp.add(n)
                    q.append(n)
        seen |= comp
        out.append(sorted(comp))
    out.sort(key=len, reverse=True)
    return out


def node_at(result, x, z, y=None):
    """Index of the node whose cell holds (x, z), nearest in y when several."""
    cell = result.params['cell']
    if not result.nodes:
        return None
    hits = []
    for k, n in enumerate(result.nodes):
        i, j, _b = n['key']
        v0 = result.verts[n['ring'][0]]
        if v0[0] - 1e-6 <= x <= v0[0] + cell + 1e-6 and v0[2] - 1e-6 <= z <= v0[2] + cell + 1e-6:
            hits.append(k)
    if not hits:
        return None
    if y is not None and len(hits) > 1:
        hits.sort(key=lambda k: abs(result.nodes[k]['centre'][1] - y))
    return hits[0]


def nav_selftest(result, probes=()):
    """The offline gate: every problem listed is a nav bug the arena shipped once."""
    problems = []
    nodes = result.nodes
    if not nodes:
        problems.append('navmesh is empty')
        return problems
    cell = result.params['cell']
    climb = cell * math.tan(math.radians(result.params['max_slope_deg'])) + result.params['max_step']
    for k, n in enumerate(nodes):
        for m in n['nb']:
            if m >= 0 and k not in nodes[m]['nb']:
                problems.append('asymmetric link %d -> %d' % (k, m))
    if len(result.islands) > 1:
        problems.append('nav split into %d islands (sizes %s)'
                        % (len(result.islands), [len(i) for i in result.islands][:8]))
    for k, n in enumerate(nodes):
        cy = n['centre'][1]
        for m in n['nb']:
            if m >= 0 and abs(nodes[m]['centre'][1] - cy) > climb + 1e-6:
                problems.append('cliff link %d -> %d (%.2f)' % (k, m, abs(nodes[m]['centre'][1] - cy)))
    for k, n in enumerate(nodes):
        for a, d in enumerate(EDGE_DIRS):
            if n['nb'][a] < 0:
                continue
            i0, i1 = EDGE_CORNERS[d]
            e0, e1 = result.verts[n['ring'][i0]], result.verts[n['ring'][i1]]
            w = max(abs(e0[0] - e1[0]), abs(e0[2] - e1[2]))
            if w < cell - 1e-4:
                problems.append('link %d edge only %.2f wide' % (k, w))
    if result.world is not None:
        caster = result.world.caster
        for k, n in enumerate(nodes):
            cx, cy, cz = n['centre']
            ys, wk, _t = caster.hits(cx, cz)
            if not np.any(wk & (np.abs(ys - cy) <= 0.5)):
                problems.append('no walkable floor under node %d at (%.1f, %.1f, %.1f)'
                                % (k, cx, cy, cz))
    main = set(result.islands[0]) if result.islands else set()
    for p in probes:
        name = p[0] if isinstance(p[0], str) else None
        coords = p[1:] if name else p
        x, z = coords[0], coords[-1]
        y = coords[1] if len(coords) == 3 else None
        k = node_at(result, x, z, y)
        if k is None:
            problems.append('probe %s (%.1f, %.1f) has no nav cell' % (name or '', x, z))
        elif k not in main:
            problems.append('probe %s (%.1f, %.1f) is off the main island' % (name or '', x, z))
    return problems


def nav_to_polys(result):
    """(verts, rings) for drawing."""
    return result.verts, [n['ring'] for n in result.nodes]


def nav_from_dict(nav):
    """A shipped nav dict -> (verts, rings), the read-only view."""
    verts, polys = ggeom.nav_polys(nav)
    return np.array(verts, F32).reshape(-1, 3), polys

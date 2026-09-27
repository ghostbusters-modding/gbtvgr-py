"""Gates over a composed level: spawns in bounds, paths walkable, no unlinked voids.
Wire-layer only, no numpy; every check returns a problem list, [] = clean."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import argparse
import collections
import math
import struct
import sys

from .. import lvl as _lvl
from ..mesh.bvt import bvt_collect
from ..sets import bst, geom
from . import link as lk
from . import transplant as tp
from . import wire

FLY_PREFIX = 'CFlyer'
SPAWN_HEIGHT = 1.5     # ft above a nav node's polygon a grounded spawn must sit within
FLY_HEIGHT = 12.0      # same, for CFlyer* rows
DOORWAY_MARGIN = 3.0   # no spawn/pool row this close to a doorway plane
BBOX_EPS = 0.1         # slack for "the BSP section holds the point"
STEP_MAX = 1.5         # ft, floor height step a path may take between nav nodes
CLEARANCE_MIN = 7.0    # ft, minimum ceiling clearance along a path
SWEEP_WIDTH = 2.0       # ft, doorway-crossing sweep box: along the edge tangent
SWEEP_DEPTH = 1.0       # ft, straddling the crossing plane
SWEEP_HEIGHT = 6.0      # ft, floor to head clearance
MIN_OPENING_EXTENT = 0.5   # ft; smaller boundary loops are tessellation seams, not holes
MAX_OPENING_FRACTION = 0.6   # of the section's own bbox size; bigger reads as an exposed
                              # edge (open sky, or where an unshipped neighbour picked up)
DOOR_COVER_MARGIN = 1.0    # ft slack when matching a geometric hole to a doorway record
XZ_TOL = 2.0            # ft off-polygon slack, the AI controllers' own search radius (RE_SECTION_LINK.md 4.5)


# ============================================================================
# small geometry helpers
# ============================================================================

def _vec3(s):
    return tuple(float(x) for x in s.split(','))


def _in_poly_xz(pt, ring):
    """Ray-cast point-in-polygon on the xz plane; ring is world-space points."""
    x, z = pt[0], pt[2]
    inside = False
    n = len(ring)
    for i in range(n):
        x1, _, z1 = ring[i]
        x2, _, z2 = ring[(i + 1) % n]
        if (z1 > z) != (z2 > z):
            xin = x1 + (z - z1) * (x2 - x1) / (z2 - z1)
            if x < xin:
                inside = not inside
    return inside


def _dist_to_ring_xz(pt, ring):
    """Nearest distance from pt to the ring's edges, projected to xz."""
    x, z = pt[0], pt[2]
    best = None
    n = len(ring)
    for i in range(n):
        x1, _, z1 = ring[i]
        x2, _, z2 = ring[(i + 1) % n]
        dx, dz = x2 - x1, z2 - z1
        l2 = dx * dx + dz * dz
        t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((x - x1) * dx + (z - z1) * dz) / l2))
        d = math.hypot(x - (x1 + t * dx), z - (z1 + t * dz))
        if best is None or d < best:
            best = d
    return best


def _ray_tri_up(o, a, b, c):
    """Height above o where the vertical ray hits a,b,c, or None if it misses or the
    face is floor-like (catalogue.py's ny > 0: an upward bump is not a ceiling)."""
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    if uz * vx - ux * vz >= 0:
        return None
    denom = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2])
    if abs(denom) < 1e-9:
        return None
    l1 = ((b[2] - c[2]) * (o[0] - c[0]) + (c[0] - b[0]) * (o[2] - c[2])) / denom
    l2 = ((c[2] - a[2]) * (o[0] - c[0]) + (a[0] - c[0]) * (o[2] - c[2])) / denom
    l3 = 1.0 - l1 - l2
    if l1 < -1e-6 or l2 < -1e-6 or l3 < -1e-6:
        return None
    y = l1 * a[1] + l2 * b[1] + l3 * c[1]
    dy = y - o[1]
    return dy if dy > 0 else None


def _ray_up(node, origin, max_dist):
    """Nearest BVT triangle straight above origin, bbox-pruned; None = clear to max_dist."""
    best = [None]

    def walk(n):
        bb = struct.unpack('<6f', n['bbox'])
        if not (bb[0] <= origin[0] <= bb[3] and bb[2] <= origin[2] <= bb[5]):
            return
        if bb[4] < origin[1] or (best[0] is not None and bb[1] - origin[1] > best[0]):
            return
        if 'kids' in n:
            for k in n['kids']:
                walk(k)
            return
        verts = [struct.unpack_from('<3f', n['verts'], i * 12)
                 for i in range(len(n['verts']) // 12)]
        for i in range(len(n['tris']) // 4):
            k = struct.unpack_from('<I', n['tris'], i * 4)[0]
            tri = (verts[k & 31], verts[(k >> 5) & 31], verts[(k >> 10) & 31])
            t = _ray_tri_up(origin, *tri)
            if t is not None and t <= max_dist and (best[0] is None or t < best[0]):
                best[0] = t
    walk(node)
    return best[0]


def _tri_aabb(tri, box):
    """Akenine-Moller triangle/AABB SAT test; box = (cx,cy,cz,ex,ey,ez) half-extents."""
    cx, cy, cz, ex, ey, ez = box
    v = [(p[0] - cx, p[1] - cy, p[2] - cz) for p in tri]
    e = [(v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]),
         (v[2][0] - v[1][0], v[2][1] - v[1][1], v[2][2] - v[1][2]),
         (v[0][0] - v[2][0], v[0][1] - v[2][1], v[0][2] - v[2][2])]
    half = (ex, ey, ez)
    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    for ai in axes:
        for ej in e:
            a = (ai[1] * ej[2] - ai[2] * ej[1], ai[2] * ej[0] - ai[0] * ej[2],
                 ai[0] * ej[1] - ai[1] * ej[0])
            if a == (0.0, 0.0, 0.0):
                continue
            ps = [v[i][0] * a[0] + v[i][1] * a[1] + v[i][2] * a[2] for i in range(3)]
            r = half[0] * abs(a[0]) + half[1] * abs(a[1]) + half[2] * abs(a[2])
            if max(-max(ps), min(ps)) > r:
                return False
    for k in range(3):
        vals = [v[0][k], v[1][k], v[2][k]]
        if max(vals) < -half[k] or min(vals) > half[k]:
            return False
    n = (e[0][1] * e[1][2] - e[0][2] * e[1][1], e[0][2] * e[1][0] - e[0][0] * e[1][2],
         e[0][0] * e[1][1] - e[0][1] * e[1][0])
    d = n[0] * v[0][0] + n[1] * v[0][1] + n[2] * v[0][2]
    r = half[0] * abs(n[0]) + half[1] * abs(n[1]) + half[2] * abs(n[2])
    return abs(d) <= r


def _box_hits_bvt(node, box):
    """First BVT triangle overlapping an axis-aligned box (centre+half-extents), or None."""
    if node is None:
        return None
    cx, cy, cz, ex, ey, ez = box
    blo = (cx - ex, cy - ey, cz - ez)
    bhi = (cx + ex, cy + ey, cz + ez)

    def walk(n):
        bb = struct.unpack('<6f', n['bbox'])
        if bb[3] < blo[0] or bb[0] > bhi[0] or bb[4] < blo[1] or bb[1] > bhi[1] \
           or bb[5] < blo[2] or bb[2] > bhi[2]:
            return None
        if 'kids' in n:
            for k in n['kids']:
                hit = walk(k)
                if hit is not None:
                    return hit
            return None
        verts = [struct.unpack_from('<3f', n['verts'], i * 12)
                 for i in range(len(n['verts']) // 12)]
        for i in range(len(n['tris']) // 4):
            k = struct.unpack_from('<I', n['tris'], i * 4)[0]
            tri = (verts[k & 31], verts[(k >> 5) & 31], verts[(k >> 10) & 31])
            if _tri_aabb(tri, box):
                return tri
        return None
    return walk(node)


def _aabb_of(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, (min(zs) + max(zs)) / 2.0,
            (max(xs) - min(xs)) / 2.0, (max(ys) - min(ys)) / 2.0, (max(zs) - min(zs)) / 2.0)


# ============================================================================
# nav helpers (independent of link.py's private ones, same conventions)
# ============================================================================

def _nav_verts(nav):
    return [struct.unpack_from('<3f', nav['verts'], i * 12) for i in range(nav['nverts'])]


def _nearest_nav_node(m, p):
    nav = m['nav']
    if nav['empty'] or not nav['nodes']:
        return None
    return min(range(len(nav['nodes'])),
              key=lambda i: math.dist(tp.nav_header(nav['nodes'][i])['centroid'], p))


def _node_walkable(nav, i):
    e = struct.unpack('<I', nav['nodes'][i]['f15c'])[0]
    return struct.unpack('<I', nav['extra'][e]['b'])[0] & lk.WALKABLE_MASK != 0


def _bfs_path(nav, s, g):
    """FIFO BFS over edge-indexed neighbours, walkable parts only (second_way.py's rule)."""
    if s == g:
        return [s]
    seen, prev, q = {s}, {}, collections.deque([s])
    while q:
        x = q.popleft()
        for j in tp.nav_neighbours(nav['nodes'][x]):
            if j < 0 or j in seen or not _node_walkable(nav, j):
                continue
            seen.add(j)
            prev[j] = x
            if j == g:
                q.clear()
                break
            q.append(j)
    if g not in seen:
        return None
    path = [g]
    while path[-1] != s:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def _edge_between(m, a, b):
    """The ring edge of node a whose neighbour is b, with midpoint, unit tangent and
    outward xz normal (link.py's _edge convention: clockwise rings, out = perp(tangent))."""
    nav = m['nav']
    verts = _nav_verts(nav)
    da = nav['nodes'][a]
    ring = tp.nav_ring(da)
    n = len(ring)
    for e, nb in enumerate(tp.nav_neighbours(da)):
        if nb != b:
            continue
        va, vb = verts[ring[e]], verts[ring[(e + 1) % n]]
        mid = tuple((va[k] + vb[k]) / 2.0 for k in range(3))
        tx, tz = vb[0] - va[0], vb[2] - va[2]
        tl = math.hypot(tx, tz)
        tangent = (tx / tl, tz / tl) if tl else (1.0, 0.0)
        out = (-tangent[1], tangent[0])
        return {'mid': mid, 'a': va, 'b': vb, 'tangent': tangent, 'out': out}
    return None


def _node_section(m, nav, i):
    leaf = tp.bsp_walk(m, tp.nav_header(nav['nodes'][i])['centroid'])
    return geom.bsp_nodes(m)[leaf][1]


def _ceiling_clearance(m, p, max_dist=300.0):
    try:
        leaf = tp.bsp_walk(m, p)
        section = geom.bsp_nodes(m)[leaf][1]
    except (IndexError, ValueError, ZeroDivisionError):
        return None
    if not 0 <= section < len(m['sections']):
        return None
    bvt = m['sections'][section]['bvt']
    if bvt is None:
        return None
    return _ray_up(bvt, (p[0], p[1] + 0.25, p[2]), max_dist)


def _sweep_blocked(m, sec_a, sec_b, mid, tangent, out):
    """A 2x1x6 ft box straddling a doorway crossing; the first triangle either
    section's BVT puts in it (a step, a lip, leftover wall) is the blockage."""
    corners = []
    for tw in (-SWEEP_WIDTH / 2.0, SWEEP_WIDTH / 2.0):
        for od in (-SWEEP_DEPTH / 2.0, SWEEP_DEPTH / 2.0):
            for dy in (0.0, SWEEP_HEIGHT):
                corners.append((mid[0] + tangent[0] * tw + out[0] * od, mid[1] + dy,
                               mid[2] + tangent[1] * tw + out[1] * od))
    box = _aabb_of(corners)
    for si in (sec_a, sec_b):
        if not 0 <= si < len(m['sections']):
            continue
        bvt = m['sections'][si]['bvt']
        if bvt is None:
            continue
        hit = _box_hits_bvt(bvt, box)
        if hit is not None:
            return 'section %d triangle near (%.2f, %.2f, %.2f)' % (si, hit[0][0], hit[0][1], hit[0][2])
    return None


# ============================================================================
# actor gathering (wire.actors over the .lvl + every .sec, engine order)
# ============================================================================

def _flies(row):
    if row.get('cls', '').startswith(FLY_PREFIX):
        return True
    return (row.get('enemy_class') or '').startswith(FLY_PREFIX)


def _spawns(level):
    maps = wire.actor_classes(level)
    out = []
    for si, actor in wire.actors(level):
        if maps[si].get(actor.tag) != 'CSpawn':
            continue
        f = _lvl.actor_fields(actor)
        if 'pos' not in f:
            continue
        out.append({'kind': 'spawn', 'name': actor.tag, 'cls': 'CSpawn',
                    'enemy_class': f.get('enemyClassName', ''), 'pos': _vec3(f['pos'])})
    return out


def _pool_rows(level):
    """Dormant characters (createStatus 0, a .cit to load): the engine's spawn stock."""
    maps = wire.actor_classes(level)
    out = []
    for si, actor in wire.actors(level):
        f = _lvl.actor_fields(actor)
        if 'charInfoFilename' not in f or f.get('createStatus') != '0' or 'pos' not in f:
            continue
        out.append({'kind': 'pool', 'name': actor.tag, 'cls': maps[si].get(actor.tag, '?'),
                    'pos': _vec3(f['pos'])})
    return out


def _triggers(level):
    maps = wire.actor_classes(level)
    out = []
    for si, actor in wire.actors(level):
        if maps[si].get(actor.tag) != 'CTrigger':
            continue
        f = _lvl.actor_fields(actor)
        if 'pos' not in f:
            continue
        out.append({'kind': 'trigger', 'name': actor.tag, 'cls': 'CTrigger', 'pos': _vec3(f['pos'])})
    return out


# ============================================================================
# 1. spawn_check
# ============================================================================

def _height_above_nav(m, nodes, p, thresh, xz_tol=XZ_TOL, eps=0.1):
    """Node whose polygon holds p in xz (inside, or within the engine's own off-mesh
    edge radius, RE_SECTION_LINK.md 4.5) and whose floor is within thresh below p."""
    nav = m['nav']
    if nav['empty'] or not nodes:
        return None
    verts = _nav_verts(nav)
    for i in nodes:
        ring = [verts[v] for v in tp.nav_ring(nav['nodes'][i])]
        if not (_in_poly_xz(p, ring) or _dist_to_ring_xz(p, ring) <= xz_tol):
            continue
        floor_y = tp.nav_header(nav['nodes'][i])['centroid'][1]
        if -eps <= p[1] - floor_y <= thresh:
            return i
    return None


def _nearest_in(m, nodes, p):
    if not nodes:
        return None
    nav = m['nav']
    best = min(nodes, key=lambda i: math.dist(tp.nav_header(nav['nodes'][i])['centroid'], p))
    return best, math.dist(tp.nav_header(nav['nodes'][best])['centroid'], p)


def _near_doorway(m, section, p, margin):
    for q in lk.openings(m, section):
        if len(q) < 3:
            continue
        face = lk.opening_face(m, section, q)
        axis, sign = face['face']
        ai = 'xyz'.index(axis)
        lo, hi = face['lo'], face['hi']
        if not all(lo[k] - margin <= p[k] <= hi[k] + margin for k in range(3) if k != ai):
            continue
        d = abs(p[ai] - face['centre'][ai])
        if d < margin:
            return {'distance': d, 'axis': axis, 'sign': sign}
    return None


def spawn_check(level, m):
    """Every CSpawn, dormant pool row and CTrigger centre: in its BSP section, near a
    walkable nav polygon (unless it flies), clear of doorway planes. [] = clean."""
    problems = []
    nsec = len(m['sections'])
    for row in _spawns(level) + _pool_rows(level) + _triggers(level):
        p = row['pos']
        tag = '%s %s' % (row['kind'], row['name'])
        loc = '(%.2f, %.2f, %.2f)' % p
        try:
            leaf = tp.bsp_walk(m, p)
            section = geom.bsp_nodes(m)[leaf][1]
        except (IndexError, ValueError, ZeroDivisionError):
            problems.append('%s at %s: BSP walk failed' % (tag, loc))
            continue
        if not 0 <= section < nsec:
            problems.append('%s at %s: BSP leaf has no section (%d)' % (tag, loc, section))
            continue
        bb = struct.unpack('<6f', m['sections'][section]['bbox'])
        if not all(bb[k] - BBOX_EPS <= p[k] <= bb[k + 3] + BBOX_EPS for k in range(3)):
            problems.append('%s at %s: BSP section %d bbox does not hold it' % (tag, loc, section))
            continue
        if row['kind'] not in ('spawn', 'pool'):
            continue
        nodes = lk.section_nav_nodes(m, section)
        thresh = FLY_HEIGHT if _flies(row) else SPAWN_HEIGHT
        if _height_above_nav(m, nodes, p, thresh) is None:
            nearest = _nearest_in(m, nodes, p)
            if nearest is None:
                problems.append('%s at %s: section %d has no nav nodes' % (tag, loc, section))
            else:
                ni, nd = nearest
                problems.append('%s at %s: not within %.1f ft above a nav node polygon in '
                                'section %d; nearest nav node %d, %.2f ft away'
                                % (tag, loc, thresh, section, ni, nd))
        door = _near_doorway(m, section, p, DOORWAY_MARGIN)
        if door is not None:
            problems.append('%s at %s: %.2f ft from the %s%s doorway plane of section %d'
                            % (tag, loc, door['distance'], door['axis'],
                               '+' if door['sign'] > 0 else '-', section))
    return problems


# ============================================================================
# 2/3. path_check, route_check
# ============================================================================

def path_check(m, start, goal):
    """BFS from the nav node under start to the one under goal (engine rules), then each
    step's floor rise, ceiling clearance and doorway crossing. [] or one first-failure."""
    nav = m['nav']
    if nav['empty'] or not nav['nodes']:
        return ['level has no navmesh: no path possible']
    s, g = _nearest_nav_node(m, start), _nearest_nav_node(m, goal)
    gs = tp.nav_header(nav['nodes'][s])['group']
    gg = tp.nav_header(nav['nodes'][g])['group']
    if gs != gg:
        return ['nav node %d (start, component %d) and node %d (goal, component %d): '
                'different components, no path' % (s, gs, g, gg)]
    path = _bfs_path(nav, s, g)
    if path is None:
        return ['no walkable path from nav node %d to node %d' % (s, g)]
    for a, b in zip(path, path[1:]):
        edge = _edge_between(m, a, b)
        if edge is None:
            return ['nav node %d -> %d: no matching ring edge (data inconsistency)' % (a, b)]
        mid = edge['mid']
        loc = '(%.2f, %.2f, %.2f)' % mid
        ha = tp.nav_header(nav['nodes'][a])['centroid'][1]
        hb = tp.nav_header(nav['nodes'][b])['centroid'][1]
        step = abs(ha - hb)
        if step > STEP_MAX:
            return ['floor step %.2f ft > %.1f ft between nav node %d and %d at %s'
                    % (step, STEP_MAX, a, b, loc)]
        clearance = _ceiling_clearance(m, mid)
        if clearance is not None and clearance < CLEARANCE_MIN:
            return ['ceiling clearance %.2f ft < %.1f ft over nav edge %d-%d at %s'
                    % (clearance, CLEARANCE_MIN, a, b, loc)]
        sec_a, sec_b = _node_section(m, nav, a), _node_section(m, nav, b)
        if sec_a != sec_b:
            hit = _sweep_blocked(m, sec_a, sec_b, mid, edge['tangent'], edge['out'])
            if hit is not None:
                return ['doorway crossing %d -> %d (section %d to %d) blocked at %s: %s'
                        % (a, b, sec_a, sec_b, loc, hit)]
    return []


def route_check(level, m, waypoints):
    """path_check chained across waypoints (hero start, trigger centres in beat order,
    level end); every leg's problems, tagged with the leg."""
    problems = []
    for i in range(len(waypoints) - 1):
        for p in path_check(m, waypoints[i], waypoints[i + 1]):
            problems.append('leg %d->%d: %s' % (i, i + 1, p))
    return problems


# ============================================================================
# 4. void_openings
# ============================================================================

def _section_boundary_loops(sec):
    """Connected components of BVT edges owned by exactly one triangle: real holes in
    a watertight collision mesh, plus tessellation seams (filtered by size upstream)."""
    bvt = sec['bvt']
    if bvt is None:
        return []
    verts, tris, _surf, _tflags = bvt_collect(bvt)
    counts = {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            counts[key] = counts.get(key, 0) + 1
    boundary = [e for e, n in counts.items() if n == 1]
    if not boundary:
        return []
    adj = {}
    for u, v in boundary:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    seen, loops = set(), []
    for u, _v in boundary:
        if u in seen:
            continue
        comp, stack = [], [u]
        seen.add(u)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        loops.append([verts[i] for i in comp])
    return loops


def _loop_quad(pts):
    """Bounding rectangle of a boundary loop on its thinnest axis, opening_face's shape."""
    lo = [min(p[k] for p in pts) for k in range(3)]
    hi = [max(p[k] for p in pts) for k in range(3)]
    ext = [hi[k] - lo[k] for k in range(3)]
    ax = min(range(3), key=lambda k: ext[k])
    others = [k for k in range(3) if k != ax]
    plane = (lo[ax] + hi[ax]) / 2.0
    corners = []
    for u in (lo[others[0]], hi[others[0]]):
        for w in (lo[others[1]], hi[others[1]]):
            c = [0.0, 0.0, 0.0]
            c[ax], c[others[0]], c[others[1]] = plane, u, w
            corners.append(tuple(c))
    return {'axis': 'xyz'[ax], 'others': others, 'lo': tuple(lo), 'hi': tuple(hi),
            'ext': ext, 'quad': corners}


def _bbox_covered(lo, hi, door_boxes, margin):
    return any(all(dlo[k] - margin <= lo[k] and hi[k] <= dhi[k] + margin for k in range(3))
              for dlo, dhi in door_boxes)


def void_openings(m, min_extent=MIN_OPENING_EXTENT, max_fraction=MAX_OPENING_FRACTION,
                  margin=DOOR_COVER_MARGIN):
    """Geometric collision holes not covered by a doorway record, doorway-sized only
    (unverified corpus-clean, see the report). [{'section','axis','lo','hi','quad'}]."""
    out = []
    for si, sec in enumerate(m['sections']):
        loops = _section_boundary_loops(sec)
        if not loops:
            continue
        sbb = struct.unpack('<6f', sec['bbox'])
        sext = [sbb[k + 3] - sbb[k] for k in range(3)]
        door_boxes = []
        for q in lk.openings(m, si):
            if len(q) < 3:
                continue
            xs = [p[0] for p in q]; ys = [p[1] for p in q]; zs = [p[2] for p in q]
            door_boxes.append(((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))))
        for pts in loops:
            if len(pts) < 3:
                continue
            info = _loop_quad(pts)
            if any(info['ext'][k] < min_extent for k in info['others']):
                continue
            if any(info['ext'][k] > max_fraction * sext[k] for k in info['others']):
                continue
            if _bbox_covered(info['lo'], info['hi'], door_boxes, margin):
                continue
            out.append({'section': si, 'axis': info['axis'], 'lo': info['lo'], 'hi': info['hi'],
                        'quad': info['quad']})
    return out


# ============================================================================
# CLI
# ============================================================================

def _load_bst(path):
    with open(path, 'rb') as fh:
        return bst.parse(fh.read())


def cmd_spawns(a):
    src = wire.DirSource(root=a.mod_dir)
    level = wire.open_level(src, a.stem, load_set=True)
    bad = spawn_check(level, level.bst)
    print('%s %s %s: %d problems' % ('OK  ' if not bad else 'FAIL', a.mod_dir, a.stem, len(bad)))
    for b in bad:
        print('    ' + b)
    return int(bool(bad))


def cmd_route(a):
    src = wire.DirSource(root=a.mod_dir)
    level = wire.open_level(src, a.stem, load_set=True)
    waypoints = [_vec3(w) for w in a.waypoints]
    bad = route_check(level, level.bst, waypoints)
    print('%s %s %s: %d problems' % ('OK  ' if not bad else 'FAIL', a.mod_dir, a.stem, len(bad)))
    for b in bad:
        print('    ' + b)
    return int(bool(bad))


def cmd_void(a):
    m = _load_bst(a.bst)
    out = void_openings(m)
    print('%s: %d void openings' % (a.bst, len(out)))
    for o in out:
        print('    section %d axis %s lo (%.2f, %.2f, %.2f) hi (%.2f, %.2f, %.2f)'
              % (o['section'], o['axis'], o['lo'][0], o['lo'][1], o['lo'][2],
                 o['hi'][0], o['hi'][1], o['hi'][2]))
    return int(bool(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('spawns', help='spawn/pool/trigger placement gate for one level')
    p.add_argument('mod_dir'); p.add_argument('stem')
    p.set_defaults(fn=cmd_spawns)
    p = sub.add_parser('route', help='path_check chained across waypoints')
    p.add_argument('mod_dir'); p.add_argument('stem')
    p.add_argument('waypoints', nargs='+', metavar='X,Y,Z')
    p.set_defaults(fn=cmd_route)
    p = sub.add_parser('void', help='geometric section holes a doorway record does not cover')
    p.add_argument('bst')
    p.set_defaults(fn=cmd_void)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())

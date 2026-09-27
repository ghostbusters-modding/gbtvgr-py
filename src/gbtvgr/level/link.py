"""Link a grafted room and its connector into the host set: doorway records, PVS
lists, the nav join, and the check that keeps every shipped set clean."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import argparse
import math
import struct
import sys

from ..mesh.bvt import union_bbox
from ..sets import bst, geom
from . import place
from . import transplant as tp

DOORWAY = 0x118
DOORWAY_FILL = b'\xcd'      # every pad byte of every shipped record
DOORWAY_BIT = 0x80000000    # a doorway id inside arr4f4
MAX_DOORWAYS = 24           # "Too many adjacent doorways in room %s", asserted at load
MAX_LINKS = 30              # link-header slots per section, never bound-checked
MAX_EDGES = 12
CORNER_EPS = 0.1            # the two quad copies of an 8-vertex record sit 0.05 apart
FACE_EPS = 0.05
WALKABLE_MASK = 5           # the AI controllers' default terrain mask
PART_TAG = 0x48415554       # the part name hash the authoring layer stamps on its nav
FORMS = ('doorway', 'b2')


def _i32s(b):
    return tp._i32s(b)


def _u32s(b):
    return tp._u32s(b)


# --- doorway records -------------------------------------------------------------------
def read_doorways(m):
    """The fx table as [{'a', 'b', 'verts'}]; the pad bytes are constant and dropped."""
    fx = m['fx']
    out = []
    for k in range(len(fx) // DOORWAY):
        r = fx[k * DOORWAY:(k + 1) * DOORWAY]
        n = struct.unpack_from('<I', r, 0x7C)[0]
        a, b = struct.unpack_from('<ii', r, 0x108)
        out.append({'a': a, 'b': b,
                    'verts': [struct.unpack_from('<3f', r, 0x80 + i * 12) for i in range(n)]})
    return out


def pack_doorway(rec):
    verts = rec['verts']
    if not 1 <= len(verts) <= 8:
        raise ValueError('a doorway carries 1 to 8 vertices, not %d' % len(verts))
    r = struct.pack('<Iii', 2, rec['a'], rec['b'])
    r += DOORWAY_FILL * (0x7C - len(r))
    r += struct.pack('<I', len(verts)) + b''.join(struct.pack('<3f', *v) for v in verts)
    r += DOORWAY_FILL * (0x108 - len(r))
    return r + struct.pack('<iiII', rec['a'], rec['b'], 0, 0)


def write_doorways(m, recs):
    m['fx'] = b''.join(pack_doorway(r) for r in recs)
    return m


def door_id(index):
    """The s32 a PVS list carries for doorway `index`."""
    return (DOORWAY_BIT | index) - 0x100000000


def is_door(entry):
    return entry < 0


def door_index(entry):
    return entry & 0x7FFFFFFF


def doorway_counts(m):
    """Adjacent doorway count per section."""
    counts = [0] * len(m['sections'])
    for r in read_doorways(m):
        counts[r['a']] += 1
        counts[r['b']] += 1
    return counts


def add_doorway(m, a, b, quad):
    """Append one doorway between sections a and b through `quad` (world corners)."""
    ns = len(m['sections'])
    if not (0 <= a < ns and 0 <= b < ns) or a == b:
        raise ValueError('doorway needs two distinct sections, got %d and %d' % (a, b))
    counts = doorway_counts(m)
    for s in (a, b):
        if counts[s] + 1 > MAX_DOORWAYS:
            raise ValueError('section %d would carry %d doorways > %d'
                             % (s, counts[s] + 1, MAX_DOORWAYS))
    recs = read_doorways(m)
    recs.append({'a': a, 'b': b, 'verts': [tuple(float(x) for x in p) for p in quad]})
    write_doorways(m, recs)
    return len(recs) - 1


# --- openings ---------------------------------------------------------------------------
def _corners(verts):
    """Distinct corners of a record: the two near-coincident quad copies collapse to one."""
    groups = []
    for p in verts:
        for g in groups:
            if max(abs(p[k] - g[0][k]) for k in range(3)) <= CORNER_EPS:
                g.append(p)
                break
        else:
            groups.append([p])
    return [tuple(sum(p[k] for p in g) / len(g) for k in range(3)) for g in groups]


def openings(m, section_index):
    """The section's real openings: the corner list of every doorway record naming it."""
    return [_corners(r['verts']) for r in read_doorways(m)
            if section_index in (r['a'], r['b'])]


def translated_openings(donor_m, donor_index, offset):
    return [[tuple(p[k] + offset[k] for k in range(3)) for p in q]
            for q in openings(donor_m, donor_index)]


def opening_face(m, section_index, quad):
    """Which section face an opening sits on, as make_connector's (axis, sign), with its
    centre, extent, cross-axis span and height. The face axis is the quad's thinnest."""
    lo = [min(p[k] for p in quad) for k in range(3)]
    hi = [max(p[k] for p in quad) for k in range(3)]
    ext = [hi[k] - lo[k] for k in range(3)]
    ax = min(range(3), key=lambda k: ext[k])
    centre = tuple((lo[k] + hi[k]) / 2.0 for k in range(3))
    bb = struct.unpack('<6f', m['sections'][section_index]['bbox'])
    sign = 1 if centre[ax] >= (bb[ax] + bb[ax + 3]) / 2.0 else -1
    axis = 'xyz'[ax]
    cross = {'x': 2, 'z': 0}.get(axis)
    return {'face': (axis, sign), 'centre': centre, 'extent': tuple(ext),
            'lo': tuple(lo), 'hi': tuple(hi),
            'span': (lo[cross], hi[cross]) if cross is not None else None,
            'height': (lo[1], hi[1]) if axis != 'y' else None}


def _mesh_bbox(sec):
    boxes = [me['pkt']['bbox'] for me in sec['meshes'] if not (me['flags'] & 0x20000)]
    if not boxes:
        raise ValueError('section %r has no inline mesh' % geom.name_str(sec['name']))
    return union_bbox(boxes)


def _rect(mb, ax, plane):
    c = 2 if ax == 0 else 0
    out = []
    for y, cv in ((mb[1], mb[c]), (mb[1], mb[c + 3]), (mb[4], mb[c + 3]), (mb[4], mb[c])):
        q = [0.0, y, 0.0]
        q[ax], q[c] = plane, cv
        out.append(tuple(q))
    return out


MOUTH_REACH = 2.1   # a loft's collar runs this far past the room's face (caps.RIM_LADDER max)


def connector_mouths(m, connector, host, room):
    """The connector's two open ends as quads: the mesh-bbox face that lies on a
    host (resp. room) bbox face plane."""
    mb = _mesh_bbox(m['sections'][connector])

    def mouth(other):
        bb = struct.unpack('<6f', m['sections'][other]['bbox'])
        for ax in (0, 2):
            for end, o, s in ((mb[ax], 3, 1.0), (mb[ax + 3], 0, -1.0)):
                plane = bb[ax + o]
                # a box ends on the room's face; a loft's collar runs on past it
                if -FACE_EPS <= (plane - end) * s <= MOUTH_REACH:
                    return _rect(mb, ax, plane)
        raise ValueError('connector %d has no open end on a bbox face of section %d; '
                         'pass quads' % (connector, other))
    return mouth(host), mouth(room)


# --- visibility ---------------------------------------------------------------------------
def _pvs(s):
    return _i32s(s['arr4f4'])


def _extend(entries, add, self_index, cap, what, section):
    for v in add:
        if v == self_index or v in entries:
            continue
        entries.append(v)
    if len(entries) > cap:
        raise ValueError('section %d %s %d entries > %d' % (section, what, len(entries), cap))
    return entries


def _set_pvs(m, index, add):
    s = m['sections'][index]
    entries = _extend(_pvs(s), add, index, tp.CAP['arr4f4'], 'arr4f4', index)
    s['arr4f4'] = struct.pack('<%di' % len(entries), *entries)


def _set_neighbours(m, index, add):
    s = m['sections'][index]
    entries = _extend(_u32s(s['arr470']), add, index, tp.CAP['arr470'], 'arr470', index)
    s['arr470'] = tp._pack_u32s(entries)


def link_visibility(m, host, connector, room, form='doorway', quads=None):
    """RE_SECTION_LINK.md 1.7. 'b2' flags the connector and room as outdoor cells and
    lists them plainly; 'doorway' adds two portal records and lists them as doorway ids.
    Returns the doorway indices added."""
    if form not in FORMS:
        raise ValueError('form must be one of %s' % (FORMS,))
    host_pvs = _pvs(m['sections'][host])
    if form == 'b2':
        for i in (connector, room):
            m['sections'][i]['b2'] = 1
        _set_pvs(m, host, [connector, room])
        _set_pvs(m, connector, [host, room] + host_pvs)
        _set_pvs(m, room, [connector, host] + host_pvs)
        _set_neighbours(m, host, [connector])
        _set_neighbours(m, connector, [host, room])
        _set_neighbours(m, room, [connector])
        return []
    q_hc, q_cr = quads if quads is not None else connector_mouths(m, connector, host, room)
    d1 = add_doorway(m, host, connector, q_hc)
    d2 = add_doorway(m, connector, room, q_cr)
    _set_pvs(m, host, [door_id(d1), connector, door_id(d2), room])
    _set_pvs(m, connector, [door_id(d1), host, door_id(d2), room] + host_pvs)
    _set_pvs(m, room, [door_id(d2), connector, door_id(d1), host] + host_pvs)
    return [d1, d2]


def link_graph(m):
    """Per-node link lists as FUN_1402E4F30 builds them at load: doorways are the
    negative ids, direct section links only between two b1 sections."""
    secs = m['sections']
    links = {i: [] for i in range(len(secs))}
    for k, r in enumerate(read_doorways(m)):
        links[door_id(k)] = [r['a'], r['b']]
        for s in (r['a'], r['b']):
            if door_id(k) not in links[s]:
                links[s].append(door_id(k))
    for i, s in enumerate(secs):
        if s['b1']:
            links[i] += [n for n in _u32s(s['arr470']) if secs[n]['b1'] and n not in links[i]]
    return links


def reachable(m, start, within=None):
    """Sections a BFS over the link graph reaches from `start`, stepping only onto
    nodes in `within` when given."""
    links = link_graph(m)
    seen, queue = {start}, [start]
    while queue:
        x = queue.pop()
        for n in links.get(x, ()):
            if n in seen or (within is not None and n not in within):
                continue
            seen.add(n)
            queue.append(n)
    return sorted(s for s in seen if not is_door(s))


def can_see(m, c, r):
    """The static half of the 1.2 rule: r drawn from camera section c, screen rects aside."""
    if c == r:
        return True
    pvs = _pvs(m['sections'][c])
    if r not in pvs:
        return False
    if m['sections'][c]['b2'] or m['sections'][r]['b2']:
        return True
    return r in reachable(m, c, within=set(pvs) | {c})


def _bbox_dist(a, b):
    """0 if the two boxes overlap on every axis, else the gap between their nearest faces."""
    return math.sqrt(sum(max(a[k] - b[k + 3], b[k] - a[k + 3], 0.0) ** 2 for k in range(3)))


def _door_bbox(rec):
    verts = rec['verts']
    return tuple(min(p[k] for p in verts) for k in range(3)) + \
        tuple(max(p[k] for p in verts) for k in range(3))


def pvs_everything(m):
    """Debug PVS: every section sees every other section and every doorway directly,
    capped at 199 nearest by bbox distance past 200. b2 and arr470 untouched."""
    secs = m['sections']
    ns = len(secs)
    doors = read_doorways(m)
    door_bb = [_door_bbox(r) for r in doors]
    notes = []
    for i, s in enumerate(secs):
        entries = list(range(ns))
        entries.remove(i)
        entries += [door_id(k) for k in range(len(doors))]
        if len(entries) > tp.CAP['arr4f4']:
            bb = struct.unpack('<6f', s['bbox'])

            def key(e, bb=bb):
                other = door_bb[door_index(e)] if is_door(e) else struct.unpack(
                    '<6f', secs[e]['bbox'])
                return _bbox_dist(bb, other)
            entries = sorted(entries, key=key)[:199]
            notes.append('section %d: %d candidates capped to 199 nearest by bbox distance'
                         % (i, ns - 1 + len(doors)))
        s['arr4f4'] = struct.pack('<%di' % len(entries), *entries)
    return notes


# --- BSP ----------------------------------------------------------------------------------
def bsp_leaves(m):
    return set(n[1] for n in geom.bsp_nodes(m) if n[1] >= 0)


def assert_leaves(m, sections):
    """A camera inside a section without a leaf reads as another section or -1."""
    missing = [s for s in sections if s not in bsp_leaves(m)]
    if missing:
        raise ValueError('no BSP leaf for section(s) %s' % missing)


def _floor_section(m, point, floors):
    """Ground truth for `point`: among bbox-containing sections with a nearby BVT
    floor, the closest by height. `floors` caches place.Floor across a batch of points."""
    x, y, z = point
    best = None
    for i, sec in enumerate(m['sections']):
        b = struct.unpack('<6f', sec['bbox'])
        if not (b[0] <= x <= b[3] and b[1] <= y <= b[4] and b[2] <= z <= b[5]):
            continue
        if sec['bvt'] is None:
            continue
        if i not in floors:
            floors[i] = place.Floor(m, i)
        fy = floors[i].at(x, z)
        if fy is None:
            continue
        d = abs(fy - y)
        if best is None or d < best[0]:
            best = (d, i, min(x - b[0], b[3] - x, z - b[2], b[5] - z))
    return (best[1], best[2]) if best else (None, None)


def camera_section_audit(m, points):
    """RE_SECTION_LINK.md 1.2: the BSP leaf at each point (bsp_walk) versus the section
    its bbox and floor actually place it in; a disagreement is where the renderer "map hides"."""
    nodes = geom.bsp_nodes(m)
    floors = {}
    rows = []
    for p in points:
        p = tuple(p)
        bsp_sec = nodes[tp.bsp_walk(m, p)][1]
        floor_sec, inset = _floor_section(m, p, floors)
        rows.append({'point': p, 'bsp': bsp_sec, 'floor': floor_sec, 'inset': inset,
                     'agree': floor_sec is None or bsp_sec == floor_sec})
    return rows


# --- nav ------------------------------------------------------------------------------------
def _nav_verts(nav):
    return [struct.unpack_from('<3f', nav['verts'], i * 12) for i in range(nav['nverts'])]


def _nav_centroids(m):
    """One world point per nav node: its header centroid, a self-contained stand-in for
    "hero start, trigger centres, nav node centroids" when a level's own points aren't at hand."""
    nav = m['nav']
    if nav['empty']:
        return []
    return [tp.nav_header(d)['centroid'] for d in nav['nodes']]


def section_nav_nodes(m, section_index, eps=1e-3):
    """Nav nodes whose every vertex lies inside the section bbox."""
    nav = m['nav']
    if nav['empty']:
        return []
    bb = struct.unpack('<6f', m['sections'][section_index]['bbox'])
    verts = _nav_verts(nav)

    def inside(p):
        return all(bb[k] - eps <= p[k] <= bb[k + 3] + eps for k in range(3))
    return [i for i, d in enumerate(nav['nodes'])
            if all(inside(verts[v]) for v in tp.nav_ring(d))]


def _edge(verts, nav, node, e):
    ring = tp.nav_ring(nav['nodes'][node])
    a, b = verts[ring[e]], verts[ring[(e + 1) % len(ring)]]
    mid = tuple((a[k] + b[k]) / 2.0 for k in range(3))
    # Outward in xz for the clockwise rings retail (and the authoring layer) use.
    out = (-(b[2] - a[2]), b[0] - a[0])
    n = math.hypot(*out)
    out = (out[0] / n, out[1] / n) if n else (0.0, 0.0)
    return {'node': node, 'edge': e, 'a': a, 'b': b, 'mid': mid, 'out': out}


def free_edges(m, nodes):
    """Every wall edge (neighbour -1) of the given nodes, with midpoint and outward normal."""
    nav = m['nav']
    verts = _nav_verts(nav)
    out = []
    for i in nodes:
        for e, nb in enumerate(tp.nav_neighbours(nav['nodes'][i])):
            if nb == -1:
                out.append(_edge(verts, nav, i, e))
    return out


def _faces(E, p):
    return E['out'][0] * (p[0] - E['mid'][0]) + E['out'][1] * (p[2] - E['mid'][2]) > 0


def find_free_edge(m, section_index, near_point):
    """(node, edge): the section's nearest wall edge that faces the point."""
    edges = free_edges(m, section_nav_nodes(m, section_index))
    if not edges:
        raise ValueError('section %d has no nav wall edge to join' % section_index)
    pool = [E for E in edges if _faces(E, near_point)] or edges
    best = min(pool, key=lambda E: math.dist(E['mid'], near_point))
    return best['node'], best['edge']


def _best_pair(side, conn):
    """The (side edge, connector edge) pair that face each other most closely."""
    best = None
    for S in side:
        for K in conn:
            d = math.dist(S['mid'], K['mid'])
            facing = (_faces(S, K['mid']) and _faces(K, S['mid'])
                      and S['out'][0] * K['out'][0] + S['out'][1] * K['out'][1] < -0.5)
            key = (0 if facing else 1, d)
            if best is None or key < best[0]:
                best = (key, S, K, facing)
    if best is None:
        raise ValueError('no wall edges to pair')
    return best[1], best[2], best[3]


def _signed_area(pts):
    return sum(pts[i][0] * pts[(i + 1) % len(pts)][2] - pts[(i + 1) % len(pts)][0] * pts[i][2]
               for i in range(len(pts)))


def _convex_cw(pts):
    n = len(pts)
    for i in range(n):
        a, b, c = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        if (b[0] - a[0]) * (c[2] - b[2]) - (b[2] - a[2]) * (c[0] - b[0]) > 1e-6:
            return False
    return _signed_area(pts) < 0


def refill_header(nav, node, group=None):
    """Centroid from the ring; the two heights keep their offsets from the old centroid."""
    verts = _nav_verts(nav)
    d = nav['nodes'][node]
    h = tp.nav_header(d)
    pts = [verts[v] for v in tp.nav_ring(d)]
    c = tuple(sum(p[k] for p in pts) / len(pts) for k in range(3))
    dy = c[1] - h['centroid'][1]
    h.update({'centroid': c, 'hmin': h['hmin'] + dy, 'hmax': h['hmax'] + dy})
    if group is not None:
        h['group'] = group
    d['hdr'] = tp.pack_nav_header(h)


def _stitch(nav, S, K):
    """Give connector edge K the side edge S's vertices (reversed, so both rings stay
    clockwise) and list each node on the other's edge."""
    Sd, Kd = nav['nodes'][S['node']], nav['nodes'][K['node']]
    s_ring, k_ring = tp.nav_ring(Sd), tp.nav_ring(Kd)
    k_ring[K['edge']] = s_ring[(S['edge'] + 1) % len(s_ring)]
    k_ring[(K['edge'] + 1) % len(k_ring)] = s_ring[S['edge']]
    verts = _nav_verts(nav)
    if not _convex_cw([verts[v] for v in k_ring]):
        raise ValueError('joining nav node %d edge %d to node %d edge %d twists the polygon'
                         % (K['node'], K['edge'], S['node'], S['edge']))
    Kd['v'] = tp._pack_u32s(k_ring)
    nb = tp.nav_neighbours(Kd)
    nb[K['edge']] = S['node']
    Kd['nb'] = struct.pack('<%di' % len(nb), *nb)
    nb = tp.nav_neighbours(Sd)
    nb[S['edge']] = K['node']
    Sd['nb'] = struct.pack('<%di' % len(nb), *nb)
    refill_header(nav, K['node'])


def compact_groups(nav):
    """Component ids dense from 0 and f their count, the invariant every shipped set keeps."""
    ids = [tp.nav_header(d)['group'] for d in nav['nodes']]
    remap = {g: i for i, g in enumerate(sorted(set(ids)))}
    for d, g in zip(nav['nodes'], ids):
        if remap[g] != g:
            h = tp.nav_header(d)
            h['group'] = remap[g]
            d['hdr'] = tp.pack_nav_header(h)
    nav['f'] = len(remap)
    return nav['f']


def _walkable(nav, extra_index):
    return struct.unpack('<I', nav['extra'][extra_index]['b'])[0] & WALKABLE_MASK != 0


def ensure_walkable(nav, nodes):
    """Point nodes whose part fails the default terrain mask at a walkable part."""
    moved = []
    part = None
    for i in nodes:
        d = nav['nodes'][i]
        e = struct.unpack('<I', d['f15c'])[0]
        if _walkable(nav, e):
            continue
        if part is None:
            part = next((k for k in range(nav['nextra']) if _walkable(nav, k)), None)
        if part is None:
            nav['extra'].append({'a': struct.pack('<I', PART_TAG), 'b': struct.pack('<I', 3),
                                 'c': struct.pack('<f', 1.0), 'rows': b''})
            nav['nextra'] = len(nav['extra'])
            part = nav['nextra'] - 1
        d['f15c'] = struct.pack('<I', part)
        moved.append(i)
    return moved


def join_nav(m, host_node, connector_nodes, room_nodes):
    """RE_SECTION_LINK.md 4.6: stitch the connector's end polygons onto a host wall edge
    and a room wall edge, then give every appended node the host's component id.
    host_node is an index, or (index, edge) to pin the edge."""
    nav = m['nav']
    if nav['empty'] or not connector_nodes:
        raise ValueError('nothing to join: empty nav or no connector nodes')
    if isinstance(host_node, tuple):
        hn, he = host_node
        host_edges = [E for E in free_edges(m, [hn]) if E['edge'] == he]
        if not host_edges:
            raise ValueError('nav node %d edge %d is not a wall edge' % (hn, he))
    else:
        hn = host_node
        host_edges = free_edges(m, [hn])
    conn_edges = free_edges(m, connector_nodes)
    S, K, facing = _best_pair(host_edges, conn_edges)
    _stitch(nav, S, K)
    report = {'host': (S['node'], S['edge']), 'connector_host': (K['node'], K['edge']),
              'host_facing': facing, 'room': None, 'connector_room': None}
    if room_nodes:
        rest = [E for E in conn_edges if (E['node'], E['edge']) != (K['node'], K['edge'])]
        S2, K2, facing2 = _best_pair(free_edges(m, room_nodes), rest)
        _stitch(nav, S2, K2)
        report.update({'room': (S2['node'], S2['edge']),
                       'connector_room': (K2['node'], K2['edge']), 'room_facing': facing2})
    gid = tp.nav_header(nav['nodes'][hn])['group']
    for i in list(connector_nodes) + list(room_nodes):
        h = tp.nav_header(nav['nodes'][i])
        h['group'] = gid
        nav['nodes'][i]['hdr'] = tp.pack_nav_header(h)
    report['group'] = gid
    report['f'] = compact_groups(nav)
    report['reparted'] = ensure_walkable(nav, connector_nodes)
    return report


# --- the whole link -------------------------------------------------------------------------
def link_room(m, host, connector, room, quads=None, form='doorway'):
    """Everything a grafted room behind a connector needs: portals or b2 visibility,
    a BSP leaf each, and the nav joined through the connector. Returns what was done."""
    ns = len(m['sections'])
    if len({host, connector, room}) != 3 or not all(0 <= i < ns for i in (host, connector, room)):
        raise ValueError('host, connector and room must be three distinct sections')
    if 'origdatasize' not in m:
        geom.capture(m)
    assert_leaves(m, (host, connector, room))
    if quads is None:
        try:
            quads = connector_mouths(m, connector, host, room)
        except ValueError:
            if form == 'doorway':
                raise
    report = {'form': form, 'doorways': link_visibility(m, host, connector, room, form, quads),
              'nav': _link_nav(m, host, connector, room, quads)}
    # The doorway table and the PVS lists sit in the fixed part: every blob offset moves.
    geom.relayout(m)
    return report


def _link_nav(m, host, connector, room, quads):
    if m['nav']['empty']:
        return None
    host_nodes = set(section_nav_nodes(m, host))
    room_nodes = [i for i in section_nav_nodes(m, room) if i not in host_nodes]
    skip = host_nodes | set(room_nodes)
    conn_nodes = [i for i in section_nav_nodes(m, connector) if i not in skip]
    if not conn_nodes:
        return None
    if quads is not None:
        near = tuple(sum(p[k] for p in quads[0]) / len(quads[0]) for k in range(3))
    else:
        bb = struct.unpack('<6f', m['sections'][connector]['bbox'])
        near = tuple((bb[k] + bb[k + 3]) / 2.0 for k in range(3))
    report = join_nav(m, find_free_edge(m, host, near), conn_nodes, room_nodes)
    report.update({'connector_nodes': conn_nodes, 'room_nodes': room_nodes})
    return report


# --- the check ------------------------------------------------------------------------------
def doorway_pvs_warnings(m):
    """A doorway's two sections not listing each other is legal (an occluded or bent
    connection, or 1.6's empty-PVS case) but worth a look: 7 of the 20 shipped sets do it."""
    secs = m['sections']
    bad = []
    for k, r in enumerate(read_doorways(m)):
        a, b = r['a'], r['b']
        if not (0 <= a < len(secs) and 0 <= b < len(secs)):
            continue  # check_links's own bounds check already flags this record
        pa, pb = _pvs(secs[a]), _pvs(secs[b])
        if b not in pa:
            bad.append('warning: doorway %d: section %d PVS omits %d' % (k, a, b))
        if a not in pb:
            bad.append('warning: doorway %d: section %d PVS omits %d' % (k, b, a))
    return bad


def camera_audit_warnings(m):
    """camera_section_audit's real disagreements at the set's own nav centroids: common
    even in retail (the k-d BSP is known to misplace), so never a hard invariant."""
    points = _nav_centroids(m)
    if not points:
        return []
    return ['warning: camera/floor disagreement at %s: bsp section %s, floor section %s'
           % (r['point'], r['bsp'], r['floor'])
           for r in camera_section_audit(m, points) if not r['agree']]


def check_links(m, warn=False):
    """Doorways, PVS lists, neighbour lists and the nav graph against the invariants every
    shipped set keeps. [] = clean; warn=True adds real but corpus-common diagnostics."""
    bad = []
    secs = m['sections']
    ns = len(secs)
    fx = m['fx']
    nd = len(fx) // DOORWAY
    counts = [0] * ns
    for k in range(nd):
        r = fx[k * DOORWAY:(k + 1) * DOORWAY]
        n, a0, b0 = struct.unpack_from('<Iii', r, 0)
        nv = struct.unpack_from('<I', r, 0x7C)[0]
        a, b, z1, z2 = struct.unpack_from('<iiII', r, 0x108)
        tag = 'doorway %d' % k
        if not (0 <= a < ns and 0 <= b < ns):
            bad.append('%s sections %d %d outside %d' % (tag, a, b, ns))
            continue
        if a == b:
            bad.append('%s joins section %d to itself' % (tag, a))
        if (n, a0, b0) != (2, a, b):
            bad.append('%s header %s disagrees with sections %d %d' % (tag, (n, a0, b0), a, b))
        if not 1 <= nv <= 8:
            bad.append('%s has %d vertices' % (tag, nv))
        if (z1, z2) != (0, 0):
            bad.append('%s ships blockers %d %d: closed at load' % (tag, z1, z2))
        counts[a] += 1
        counts[b] += 1
    for i, s in enumerate(secs):
        tag = 'section %d' % i
        if counts[i] > MAX_DOORWAYS:
            bad.append('%s has %d doorways > %d' % (tag, counts[i], MAX_DOORWAYS))
        pvs = _pvs(s)
        if len(pvs) > tp.CAP['arr4f4']:
            bad.append('%s arr4f4 %d entries > %d' % (tag, len(pvs), tp.CAP['arr4f4']))
        if len(pvs) != len(set(pvs)):
            bad.append('%s arr4f4 has duplicates' % tag)
        for v in pvs:
            if is_door(v) and door_index(v) >= nd:
                bad.append('%s arr4f4 doorway %d >= %d' % (tag, door_index(v), nd))
            elif not is_door(v) and (v >= ns or v == i):
                bad.append('%s arr4f4 section %d %s'
                           % (tag, v, 'is itself' if v == i else 'invalid'))
        nbs = _u32s(s['arr470'])
        if len(nbs) > tp.CAP['arr470']:
            bad.append('%s arr470 %d entries > %d' % (tag, len(nbs), tp.CAP['arr470']))
        if len(nbs) != len(set(nbs)):
            bad.append('%s arr470 has duplicates' % tag)
        for v in nbs:
            if v >= ns or v == i:
                bad.append('%s arr470 section %d %s'
                           % (tag, v, 'is itself' if v == i else 'invalid'))
        slots = counts[i] + (sum(1 for v in nbs if v < ns and secs[v]['b1']) if s['b1'] else 0)
        if slots > MAX_LINKS:
            bad.append('%s needs %d link slots > %d' % (tag, slots, MAX_LINKS))
    bad += check_nav(m['nav'])
    if warn:
        bad += doorway_pvs_warnings(m) + camera_audit_warnings(m)
    return bad


def check_nav(nav):
    """Symmetric links, edge cap, component ids equal to the connected components, dense."""
    bad = []
    if nav['empty']:
        return bad
    nodes = nav['nodes']
    nn = len(nodes)
    nbs = [tp.nav_neighbours(d) for d in nodes]
    for i, nb in enumerate(nbs):
        if len(nb) > MAX_EDGES:
            bad.append('nav node %d has %d edges > %d' % (i, len(nb), MAX_EDGES))
        for j in nb:
            if j == -1:
                continue
            if not 0 <= j < nn:
                bad.append('nav node %d neighbour %d outside %d nodes' % (i, j, nn))
            elif i not in nbs[j]:
                bad.append('nav link %d -> %d is one-way' % (i, j))
    comp = [-1] * nn
    ncomp = 0
    for s in range(nn):
        if comp[s] >= 0:
            continue
        comp[s] = ncomp
        stack = [s]
        while stack:
            x = stack.pop()
            for j in nbs[x]:
                if 0 <= j < nn and comp[j] < 0:
                    comp[j] = ncomp
                    stack.append(j)
        ncomp += 1
    ids = [tp.nav_header(d)['group'] for d in nodes]
    by_comp = {}
    by_id = {}
    for i in range(nn):
        by_comp.setdefault(comp[i], set()).add(ids[i])
        by_id.setdefault(ids[i], set()).add(comp[i])
    for c, gs in sorted(by_comp.items()):
        if len(gs) > 1:
            bad.append('nav component %d carries ids %s' % (c, sorted(gs)))
    for g, cs in sorted(by_id.items()):
        if len(cs) > 1:
            bad.append('nav id %d spans %d components' % (g, len(cs)))
    if sorted(set(ids)) != list(range(nav['f'])):
        bad.append('nav ids %s are not 0..f-1 (f=%d)' % (sorted(set(ids)), nav['f']))
    return bad


# --- CLI -------------------------------------------------------------------------------------
def _load(path):
    with open(path, 'rb') as fh:
        return bst.parse(fh.read())


def cmd_check_links(a):
    rc = 0
    for f in a.files:
        bad = check_links(_load(f))
        print('%s %s: %d problems' % ('OK  ' if not bad else 'FAIL', f, len(bad)))
        for b in bad:
            print('    ' + b)
        rc |= bool(bad)
    return rc


def cmd_doorways(a):
    m = _load(a.file)
    for k, r in enumerate(read_doorways(m)):
        names = [geom.name_str(m['sections'][i]['name']) for i in (r['a'], r['b'])]
        c = _corners(r['verts'])
        print('%3d  %3d %-24s %3d %-24s %d verts, %d corners'
              % (k, r['a'], names[0], r['b'], names[1], len(r['verts']), len(c)))
        if a.verbose:
            for p in c:
                print('        %9.2f %9.2f %9.2f' % p)
    return 0


def cmd_openings(a):
    m = _load(a.file)
    for q in openings(m, a.section):
        f = opening_face(m, a.section, q)
        print('face %s%s centre (%.2f, %.2f, %.2f) extent (%.2f, %.2f, %.2f)'
              % (f['face'][0], '+' if f['face'][1] > 0 else '-', *f['centre'], *f['extent']))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('check-links', help='doorway, PVS and nav-graph invariants of a set')
    p.add_argument('files', nargs='+')
    p.set_defaults(fn=cmd_check_links)
    p = sub.add_parser('doorways', help='list the doorway records of a set')
    p.add_argument('file')
    p.add_argument('-v', '--verbose', action='store_true')
    p.set_defaults(fn=cmd_doorways)
    p = sub.add_parser('openings', help="one section's openings and the faces they sit on")
    p.add_argument('file')
    p.add_argument('section', type=int)
    p.set_defaults(fn=cmd_openings)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())

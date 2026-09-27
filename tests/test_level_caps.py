"""Mouths: the loft ends on the room's real rim, the capped box meets its mouth, and
openings that lead nowhere close with one flat colour."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

from _run import SRC, gbtvgr, ok
from gbtvgr.level import caps, connector, host, lighting, link, place, transplant as tp
from gbtvgr.mesh.bvt import bvt_bbox
from gbtvgr.sets import bst, geom

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEMO = os.path.join(ROOT, 'mods', 'TransplantDemo')
A, B, C = 0, 1, 2                  # Room25, Room26, the connector, as graft2 lays them out
A_SECTION, B_SECTION = 42, 43      # their cemetery1 indices
B_OFFSET = (0.0, 0.0, 30.0)
MATERIAL = 'graveyard\\ugp_rock'
FLOOR_CEIL = {A: -18.0, B: -14.0, C: -12.0}
TOUCHED = ['section 0 header', 'fx', 'nav nodes[:old]']
_CACHE = {}


def _read(path):
    with open(path, 'rb') as fh:
        return fh.read()


def _gates(path):
    ok(gbtvgr('bst', 'verify', path))
    out = ok(gbtvgr('bst-geom', 'check', path))
    assert 'FAIL' not in out, out


def _round_trip(m):
    out = bst.build(m)
    after = bst.parse(out)
    assert bst.build(after) == out
    return out, after


def _corridor_floor(m):
    mb = link._mesh_bbox(m['sections'][C])
    cx = (mb[0] + mb[3]) / 2.0
    fl = place.Floor(m, C, below=FLOOR_CEIL[C])
    return mb, [fl.at(cx, z) for z in ((mb[2] + mb[5]) / 2.0, mb[2] + 0.5, mb[5] - 0.5)]


def _centre(q):
    return tuple(sum(p[k] for p in q) / len(q) for k in range(3))


# --- graft2's set rebuilt from the corpus, as make_demo2.build_set does it -------------------
def _rebuild(bst_corpus, profile, light=False):
    path = os.path.join(bst_corpus, 'cemetery1.bst')
    if not os.path.isfile(path):
        pytest.skip('needs cemetery1 in the .bst corpus')
    donor = bst.parse(_read(path))
    seed, _ = host.new_set_from_section(donor, A_SECTION, 'graft2')
    seed_bytes = bst.build(seed)
    m = bst.parse(seed_bytes)
    m, _st, bi = place.place_section(m, donor, B_SECTION, B_OFFSET, lightmap_dir='graft2')
    assert bi == B
    tp.append_nav(m, donor, B_SECTION, B_OFFSET)
    qa = [q for q in link.openings(donor, A_SECTION)
          if link.opening_face(m, A, q)['face'] == ('z', 1)]
    qb = [q for q in link.translated_openings(donor, B_SECTION, B_OFFSET)
          if link.opening_face(m, B, q)['face'] == ('z', -1)]
    b = qb[0]
    a = min(qa, key=lambda q: abs(_centre(q)[0] - _centre(b)[0]))
    fa, fb = link.opening_face(m, A, a), link.opening_face(m, B, b)
    span = (max(fa['span'][0], fb['span'][0]), min(fa['span'][1], fb['span'][1]))
    top = min(fa['height'][1], fb['height'][1])
    cx = (span[0] + span[1]) / 2.0
    floor_a = place.Floor(m, A, below=FLOOR_CEIL[A]).at(cx, _centre(a)[2] - 2.0)
    floor_b = place.Floor(m, B, below=FLOOR_CEIL[B]).at(cx, _centre(b)[2] + 2.0)
    height = top - min(floor_a, floor_b)
    nav_before = m['nav']['nnodes']
    m = connector.make_connector(m, A, ('z', 1), B, ('z', -1), span[1] - span[0], height,
                                 MATERIAL, span=span, cell=6.0, profile=profile, light=light)
    chain = list(range(nav_before, m['nav']['nnodes']))
    probe = tp._u32s(m['sections'][A]['portidx'])[0]
    m['sections'][C]['portidx'] = tp._pack_u32s([probe])
    report = link.link_room(m, A, C, B, quads=(a, b), form='doorway')
    out, parsed = _round_trip(m)
    return dict(donor=donor, seed=bst.parse(seed_bytes), parsed=parsed, out=out, chain=chain,
                report=report, span=span, height=height)


def _box(bst_corpus):
    """The plain box corridor, uncapped, unlit, unplugged: what the user saw first."""
    if 'box' not in _CACHE:
        _CACHE['box'] = _rebuild(bst_corpus, 'box')
    return _CACHE['box']


def _loft(bst_corpus, light=False):
    key = 'loft_lit' if light else 'loft'
    if key not in _CACHE:
        _CACHE[key] = _rebuild(bst_corpus, 'loft', light)
    return _CACHE[key]


def _mouth_rect(m):
    """(lo, hi, centre) of the box corridor's cross-section: the rect the loft is cast in."""
    mb = link._mesh_bbox(m['sections'][C])
    lo, hi = (mb[0], mb[1], mb[2]), (mb[3], mb[4], mb[5])
    return lo, hi, [(lo[k] + hi[k]) / 2.0 for k in range(3)]


def _tri_dist(p, tri):
    """Distance from p to the nearest of the (v0, v1, v2) triangle arrays."""
    v0, v1, v2 = (np.asarray(v, np.float64) for v in tri)
    p = np.asarray(p, np.float64)
    ab, ac, ap = v1 - v0, v2 - v0, p - v0
    n = np.cross(ab, ac)
    nn = np.einsum('ij,ij->i', n, n)
    d = np.einsum('ij,ij->i', n, ap) / np.where(nn > 1e-18, nn, 1.0)
    q = p - d[:, None] * n
    # inside test on the projection, else the nearest edge
    c0 = np.einsum('ij,ij->i', np.cross(v1 - v0, q - v0), n) >= 0
    c1 = np.einsum('ij,ij->i', np.cross(v2 - v1, q - v1), n) >= 0
    c2 = np.einsum('ij,ij->i', np.cross(v0 - v2, q - v2), n) >= 0
    inside = c0 & c1 & c2
    best = np.where(inside, np.abs(d) * np.sqrt(np.where(nn > 1e-18, nn, 1.0)), np.inf)
    for a, b in ((v0, v1), (v1, v2), (v2, v0)):
        e = b - a
        t = np.clip(np.einsum('ij,ij->i', p - a, e) / np.maximum(np.einsum('ij,ij->i', e, e), 1e-18),
                    0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(a + t[:, None] * e - p, axis=1))
    return float(best.min())


# --- the mask ------------------------------------------------------------------------------
def test_box_mouths_are_round_and_smaller_than_the_box(bst_corpus):
    r = _box(bst_corpus)
    m = r['parsed']
    mb = link._mesh_bbox(m['sections'][C])
    assert abs(mb[0] - r['span'][0]) < 1e-3 and abs(mb[3] - r['span'][1]) < 1e-3
    assert abs(mb[4] - mb[1] - r['height']) < 1e-3
    face_w, face_h = mb[3] - mb[0], mb[4] - mb[1]
    for mk, room in zip(caps.mouths(m, C, (A, B)), (A, B)):
        assert mk.axis == 'z'
        assert 0 < mk.n_open < mk.grid.size, room
        c0, y0, c1, y1 = mk.open_rect
        assert c1 - c0 < face_w - 1.0 and y1 - y0 < face_h - 0.5, \
            'room %d: the open region should be well inside the box face' % room
        # round: the open cells do not fill their own bounding rectangle
        cells = (c1 - c0) / mk.dc * (y1 - y0) / mk.dy
        assert mk.n_open < 0.9 * cells, room
        # the corners of the box face are closed on both mouths
        for j, i in ((0, 0), (0, mk.nc - 1), (mk.ny - 1, 0), (mk.ny - 1, mk.nc - 1)):
            assert not mk.grid[j, i], (room, j, i)
        # the middle of the floor row is open: the floors meet, no lip in the path
        assert mk.grid[0, mk.nc // 2] or np.isfinite(mk.riser[0, mk.nc // 2]), room


def test_mask_needs_a_sane_rectangle_and_axis(bst_corpus):
    m = _box(bst_corpus)['parsed']
    with pytest.raises(ValueError):
        caps.mouth_mask(m, A, -90.86, 'y', 1, (0, 0, 0), (1, 1, 1))
    with pytest.raises(ValueError):
        caps.mouth_mask(m, A, -90.86, 'z', 1, (50, -20, 0), (50, -20, 0))


# --- the traced outline -----------------------------------------------------------------------
def test_trace_follows_the_mouth(bst_corpus):
    m = _box(bst_corpus)['parsed']
    for mk, room in zip(caps.mouths(m, C, (A, B)), (A, B)):
        caps.trace(m, room, mk)
        bands = mk.bands
        assert abs(bands[0][0] - mk.y_lo) < 1e-6 and abs(bands[-1][1] - mk.y_hi) < 1e-6
        for k in range(1, len(bands)):
            assert abs(bands[k][0] - bands[k - 1][1]) < 1e-6, 'bands must tile the face'
        widths = []
        for y0, y1, ivals in bands:
            assert caps.BAND * 0.5 <= y1 - y0 <= caps.BAND * 1.5 + 1e-6
            assert ivals == sorted(ivals)
            for (l0, r0), (l1, r1) in zip(ivals, ivals[1:]):
                assert r0 < l1
            for left, right in ivals:
                assert mk.c_lo <= left < right <= mk.c_hi
                assert right - left >= caps.MIN_GAP
            widths.append(sum(r - l for l, r in ivals))
        # a round mouth: the floor band is open, the widest band sits in the middle,
        # the top closes
        assert widths[0] > 0, 'the floors meet, so the bottom band is open at the centre'
        assert widths[-1] == 0
        peak = int(np.argmax(widths))
        assert 0.25 * len(widths) < peak < 0.75 * len(widths)
        assert max(widths) > 9.0
        # the scanned edges: the band's centre is open just inside the stored edge, and
        # a closed point lies within one scan step beyond the boundary the edge overlaps
        for y0, y1, ivals in bands[1:-2]:
            hs = [y0 + 0.01, (y0 + y1) / 2.0, y1 - 0.01]
            for left, right in ivals:
                for edge, out in (((left, -1) if left > mk.c_lo + 1e-6 else (None, 0)),
                                  ((right, 1) if right < mk.c_hi - 1e-6 else (None, 0))):
                    if edge is None:
                        continue
                    assert caps.open_points(m, room, mk, [edge - out * 0.02], [hs[1]])[0]
                    boundary = edge + out * caps.OVERLAP
                    probes = [boundary + out * d for d in (0.005, 0.01, 0.015, 0.02)]
                    assert not all(caps.open_points(m, room, mk, [c] * 3, hs).all()
                                   for c in probes), (room, y0, edge)
        # the cap and the hole tile the face exactly
        R = (mk.c_lo, mk.c_hi, mk.y_lo, mk.y_hi)
        tris = caps.band_triangles(bands, R)
        area = sum(abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0
                   for a, b, c in tris)
        hole = sum((y1 - y0) * w for (y0, y1, _iv), w in zip(bands, widths))
        assert abs(area + hole - (R[1] - R[0]) * (R[3] - R[2])) < 1e-3
        assert all(abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) > 1e-6
                   for a, b, c in tris), 'no degenerate strip triangle'


def test_strip_triangulation_has_no_t_junctions():
    bands = [(0.0, 1.0, [(2.0, 8.0)]), (1.0, 2.0, [(1.0, 9.0)]),
             (2.0, 3.0, [(3.0, 4.0), (6.0, 7.0)]), (3.0, 4.0, [])]
    R = (0.0, 10.0, 0.0, 4.0)
    tris = caps.band_triangles(bands, R)
    area = sum(abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0
               for a, b, c in tris)
    assert abs(area - (40.0 - 6.0 - 8.0 - 2.0)) < 1e-9
    # every vertex that lies on another triangle's edge is a vertex of that triangle
    verts = {p for t in tris for p in t}
    for a, b, c in tris:
        for p in verts:
            if p in (a, b, c):
                continue
            for u, v in ((a, b), (b, c), (c, a)):
                cross = (v[0] - u[0]) * (p[1] - u[1]) - (v[1] - u[1]) * (p[0] - u[0])
                between = min(u[0], v[0]) - 1e-9 <= p[0] <= max(u[0], v[0]) + 1e-9 and \
                    min(u[1], v[1]) - 1e-9 <= p[1] <= max(u[1], v[1]) + 1e-9
                assert not (abs(cross) < 1e-9 and between), ('T-junction', p, (u, v))


# --- the cap -------------------------------------------------------------------------------
@pytest.mark.parametrize('method', ['trace', 'cells'])
def test_cap_connector_on_the_box(bst_corpus, tmp_path, method):
    before = _box(bst_corpus)['parsed']
    assert caps.check_caps(before) != [], 'the plain box corridor shows the void'
    _mb, floors_before = _corridor_floor(before)
    m = bst.parse(_box(bst_corpus)['out'])
    geom.capture(m)
    masks, reports = caps.cap_connector(m, C, A, B, method=method)
    assert all(r['method'] == method and r['rects'] > 0 and r['verts'] > 0 for r in reports)
    added = sum(r['verts'] for r in reports)
    assert 100 < added < 2000, added
    out, after = _round_trip(m)
    assert tp.check_refs(after) == []
    assert tp.check_additive(before, after) == ['section 2 header', 'section 2 blob']
    assert link.check_links(after) == []
    assert caps.check_caps(after) == [] and caps.check_caps(after, cell=0.5) == []
    assert after['nav'] == before['nav'], 'the nav chain must not move'
    for i in (A, B):
        assert after['sections'][i]['blob'] == before['sections'][i]['blob']
    mb, floors = _corridor_floor(after)
    assert floors == floors_before and abs(floors[0] - mb[1]) < 1e-3
    new = after['sections'][C]
    bb, rb = struct.unpack('<6f', new['bbox']), bvt_bbox(new['bvt'])
    assert max([bb[k] - rb[k] for k in range(3)] + [rb[k] - bb[k] for k in range(3, 6)]) <= 0.01
    me = new['meshes'][0]
    assert me['pkt']['nverts'] == before['sections'][C]['meshes'][0]['pkt']['nverts'] + added
    # the cap keeps the corridor's vertex colours, so it is lit like the corridor
    cols = set(geom.decode_mesh(me['pkt'])['color'][11])
    assert cols == set(geom.decode_mesh(before['sections'][C]['meshes'][0]['pkt'])['color'][11])
    path = str(tmp_path / ('graft2_%s.bst' % method))
    with open(path, 'wb') as fh:
        fh.write(out)
    _gates(path)


def test_cap_method_is_checked(bst_corpus):
    m = bst.parse(_box(bst_corpus)['out'])
    mk = caps.mouths(m, C, (A,))[0]
    with pytest.raises(ValueError, match='method'):
        caps.cap_mouth(m, C, A, mk, method='smooth')


def test_cap_rects_cover_exactly_the_closed_cells(bst_corpus):
    m = _box(bst_corpus)['parsed']
    mk = caps.mouths(m, C, (A,))[0]
    R = (mk.c_lo, mk.c_hi, mk.y_lo, mk.y_hi)
    rects = caps.cap_rects(mk, R)
    area = sum((c1 - c0) * (y1 - y0) for c0, c1, y0, y1 in rects)
    closed = 0.0
    for j in range(mk.ny):
        for i in range(mk.nc):
            if mk.grid[j, i]:
                continue
            c0, c1, y0, y1 = mk.cell(i, j)
            top = mk.riser[j, i] if np.isfinite(mk.riser[j, i]) else y1
            closed += (c1 - c0) * (top - y0)
    assert abs(area - closed) < 1e-3
    # a face wider than the mask gets a frame
    wide = (mk.c_lo - 2.0, mk.c_hi + 2.0, mk.y_lo, mk.y_hi + 1.0)
    extra = sum((c1 - c0) * (y1 - y0) for c0, c1, y0, y1 in caps.cap_rects(mk, wide)) - area
    assert abs(extra - (4.0 * (mk.y_hi - mk.y_lo) + (wide[1] - wide[0]) * 1.0)) < 1e-3


# --- the rim ---------------------------------------------------------------------------------
def test_mouth_rim_is_graft2s_round_tube(bst_corpus):
    """96 rays from the doorway centre: every one hits the tube at 0.15 ft in, none is
    clamped, the outline is round, and it sits on the mesh to 0.05 ft."""
    m = _box(bst_corpus)['parsed']
    lo, hi, centre = _mouth_rect(m)
    for room, plane, sign in ((A, lo[2], 1), (B, hi[2], -1)):
        rim = caps.mouth_rim(m, room, plane, centre, 'z', rect=(lo, hi), sign=sign)
        assert rim.n == 96 and rim.sign == sign and not rim.clamped.any()
        assert np.all(rim.depth == caps.RIM_LADDER[0])
        assert not rim.is_rect()
        r = np.linalg.norm(rim.pts - np.asarray(rim.centre)[None, :], axis=1)
        assert 4.5 < r.min() and r.max() < 6.5, 'a round tube about 10 ft across'
        assert lo[0] < rim.pts[:, 0].min() and rim.pts[:, 0].max() < hi[0]
        assert abs(rim.lowest[1] - (-21.8)) < 0.1
        c0, c1 = rim.chord(rim.lowest[1] + connector.NAV_LIFT)
        assert 4.5 < c1 - c0 < 6.0 and c0 < rim.lowest[0] < c1
        rtri = caps._geometry(m, room)[1]
        for p in rim.points3(rim.depth):
            assert _tri_dist(p, rtri) < 1e-3, 'the probe point is on the mesh'
        # the tube is straight, so a deeper fan lands on the same outline
        deeper = caps.mouth_rim(m, room, plane, centre, 'z', rect=(lo, hi), sign=sign,
                                insets=(0.3,))
        assert np.abs(deeper.pts - rim.pts).max() < 0.05
        # projected on to the plane, the rim is within 0.05 ft of the rock where the rock
        # reaches the plane; Room25's ends 0.12 ft short of its bbox face
        P = np.concatenate(rtri)
        near = ((P[:, 0] > lo[0]) & (P[:, 0] < hi[0]) & (P[:, 1] > lo[1]) & (P[:, 1] < hi[1])
                & ((P[:, 2] - plane) * sign <= 0.0))
        gap = abs(plane - (P[near, 2].max() if sign > 0 else P[near, 2].min()))
        worst = max(_tri_dist(p, rtri) for p in rim.points3())
        assert worst < 0.05 + gap + 1e-3, (room, worst, gap)
        if gap < 0.01:
            assert worst < 0.05


def test_mouth_rim_of_a_flat_doorway_is_its_rectangle(bst_corpus):
    """library1b's rectangular doorways (Rm_SpCollect to its stair, the breaker room to its
    hall) rim as rectangles; a side with no rock near the plane clamps to the record."""
    path = os.path.join(bst_corpus, 'library1b.bst')
    if not os.path.isfile(path):
        pytest.skip('needs library1b in the .bst corpus')
    m = bst.parse(_read(path))
    recs = link.read_doorways(m)
    names = {i: geom.name_str(s['name']) for i, s in enumerate(m['sections'])}

    def rims(k):
        rec = recs[k]
        lo, hi, axis, plane = caps._doorway_geometry(m, rec)
        centre = [(lo[j] + hi[j]) / 2.0 for j in range(3)]
        return [(s, caps.mouth_rim(m, s, plane, centre, axis, rect=(lo, hi))) for s in (rec['a'], rec['b'])], lo, hi
    for k, pair in ((28, {'Rm_SpCollect', 'Rm_StacksB_Stair1'}),
                    (3, {'Rm_Breaker', 'Rm_Breaker_Hall_A'})):
        out, lo, hi = rims(k)
        assert {names[s] for s, _r in out} == pair
        for s, rim in out:
            assert rim.is_rect(0.05) and not rim.clamped.any(), (k, names[s])
            assert rim.pts[:, 0].min() >= lo[caps.CROSS[rim.axis]] - 0.05
    out, lo, hi = rims(20)
    clamped = [rim for _s, rim in out if rim.clamped.all()]
    assert len(clamped) == 1
    rim = clamped[0]
    c = caps.CROSS[rim.axis]
    assert rim.is_rect(1e-6)
    assert np.allclose(rim.pts.min(axis=0), (lo[c], lo[1])) and \
        np.allclose(rim.pts.max(axis=0), (hi[c], hi[1]))


def test_rim_needs_geometry_or_a_rect(bst_corpus):
    m = _box(bst_corpus)['parsed']
    lo, hi, centre = _mouth_rect(m)
    with pytest.raises(ValueError):
        caps.mouth_rim(m, A, lo[2], centre, 'y', rect=(lo, hi))
    with pytest.raises(ValueError, match='no geometry'):
        caps.mouth_rim(m, A, lo[2] - 40.0, (centre[0], 40.0, 0.0), 'z', sign=1)


# --- the loft --------------------------------------------------------------------------------
def _edge_counts(tris):
    seen = {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            seen[(min(u, v), max(u, v))] = seen.get((min(u, v), max(u, v)), 0) + 1
    return seen


def test_loft_profile_graft2(bst_corpus, tmp_path):
    """The default profile: one 4-ring mesh from rim to rim, buried 0.15 ft into each room,
    narrower than the box, on the tube's own floor, every gate clean."""
    box = _box(bst_corpus)['parsed']
    r = _loft(bst_corpus)
    parsed = r['parsed']
    bb = link._mesh_bbox(box['sections'][C])
    lo, hi, centre = _mouth_rect(box)
    rims = [caps.mouth_rim(box, A, lo[2], centre, 'z', rect=(lo, hi), sign=1),
            caps.mouth_rim(box, B, hi[2], centre, 'z', rect=(lo, hi), sign=-1)]
    mb, floors = _corridor_floor(parsed)
    a_plane = struct.unpack('<6f', parsed['sections'][A]['bbox'])[5]
    b_plane = struct.unpack('<6f', parsed['sections'][B]['bbox'])[2]
    assert abs(mb[2] - (a_plane - caps.RIM_LADDER[0])) < 1e-3
    assert abs(mb[5] - (b_plane + caps.RIM_LADDER[0])) < 1e-3
    f = connector.RING_FLANGE
    assert mb[0] > bb[0] + 0.4 and mb[3] < bb[3] - 0.4, 'the loft hugs the tube'
    assert mb[4] < bb[4] - 0.5 and bb[1] - 0.5 - f < mb[1] < bb[1], 'the tube floor, a little lower'
    rim_lo = min(r.lowest[1] for r in rims)
    assert abs(mb[1] - (rim_lo - f)) < 0.02, 'the flange reaches below the rim'
    meshes = [me for me in parsed['sections'][C]['meshes'] if not (me['flags'] & 0x20000)]
    assert len(meshes) == 1
    pkt = meshes[0]['pkt']
    assert pkt['nverts'] == 6 * (caps.RIM_N + 1) and pkt['nprims'] == 5 * 2 * caps.RIM_N
    assert struct.unpack('<I', meshes[0]['f6c'])[0] == 2 and 12 not in geom.decl_slots(pkt['decl'])
    g = geom.decode_mesh(pkt)
    pos = np.array(g['vec'][0])
    # closed along its length: every edge shared by two triangles but the flanges' rims
    key = {}
    for i, p in enumerate(map(tuple, np.round(pos, 4))):
        key.setdefault(p, len(key))
    tris = [tuple(key[tuple(np.round(pos[v], 4))] for v in t) for t in g['tris']]
    counts = _edge_counts(tris)
    assert sum(1 for c in counts.values() if c == 1) == 2 * caps.RIM_N
    assert all(c <= 2 for c in counts.values())
    # the tube faces inward, the two flanges face their rooms
    nrm = np.array(g['vec'][1])
    axis_pt = np.array([(mb[0] + mb[3]) / 2.0, (mb[1] + mb[4]) / 2.0])
    inward = ((axis_pt[None, :] - pos[:, :2]) * nrm[:, :2]).sum(axis=1)
    ends = (np.abs(pos[:, 2] - mb[2]) < 1e-3) | (np.abs(pos[:, 2] - mb[5]) < 1e-3)
    assert np.all(inward[~ends] > 0)
    assert np.all(nrm[np.abs(pos[:, 2] - mb[2]) < 1e-3][:, 2] < 0)
    assert np.all(nrm[np.abs(pos[:, 2] - mb[5]) < 1e-3][:, 2] > 0)
    # the floor: the tube's bottom at the centre line, found at the middle and both ends
    assert all(f is not None for f in floors)
    assert all(rim_lo - 1e-3 <= f <= rim_lo + 0.2 for f in floors), floors
    # gates
    assert tp.check_refs(parsed) == []
    assert link.check_links(parsed) == []
    assert caps.check_caps(parsed) == [] and caps.check_caps(parsed, cell=0.5) == []
    assert caps.check_caps(parsed, cell=0.25) == []
    assert tp.check_additive(r['seed'], parsed) == TOUCHED
    assert len(r['chain']) == 5
    j = r['report']['nav']
    assert j['host_facing'] and j['room_facing'] and len(j['connector_nodes']) == 5
    # the nav strip: along the bottom, as wide as the tube half a foot up
    verts = link._nav_verts(parsed['nav'])
    mid = tp.nav_ring(parsed['nav']['nodes'][r['chain'][2]])
    ys = [verts[v][1] for v in mid]
    xs = sorted(verts[v][0] for v in mid)
    assert max(ys) - min(ys) < 0.05 and abs(min(ys) - rim_lo) < 0.1
    assert 4.5 < xs[-1] - xs[0] < 6.0 and mb[0] < xs[0] and xs[-1] < mb[3]
    assert place.section_at(parsed, (mb[0] + mb[3]) / 2.0, mb[1] + 1.0, (mb[2] + mb[5]) / 2.0) == C
    path = str(tmp_path / 'graft2_loft.bst')
    with open(path, 'wb') as fh:
        fh.write(r['out'])
    _gates(path)


def test_mask_is_an_alias_of_loft(bst_corpus):
    """Build scripts that asked for the capped box get the loft, byte for byte."""
    assert _rebuild(bst_corpus, 'mask')['out'] == _loft(bst_corpus)['out']


def test_capped_profile_graft2(bst_corpus, tmp_path):
    """The old masked box with traced caps is still there under 'capped'."""
    box = _box(bst_corpus)['parsed']
    r = _rebuild(bst_corpus, 'capped')
    parsed = r['parsed']
    mb, floors = _corridor_floor(parsed)
    bb = link._mesh_bbox(box['sections'][C])
    assert mb[0] > bb[0] + 0.4 and mb[3] < bb[3] - 0.4, 'the corridor should hug the hole'
    assert mb[4] < bb[4] - 0.5 and abs(mb[1] - bb[1]) < 1e-3, 'lower ceiling, same floor'
    assert abs(mb[2] - bb[2]) < 1e-3 and abs(mb[5] - bb[5]) < 1e-3
    assert all(abs(f - mb[1]) < 1e-3 for f in floors)
    assert tp.check_refs(parsed) == []
    assert link.check_links(parsed) == []
    assert caps.check_caps(parsed) == [] and caps.check_caps(parsed, cell=0.5) == []
    assert tp.check_additive(r['seed'], parsed) == TOUCHED
    assert len(r['chain']) == 5
    me = parsed['sections'][C]['meshes'][0]
    assert 100 < me['pkt']['nverts'] < 2000
    path = str(tmp_path / 'graft2_capped.bst')
    with open(path, 'wb') as fh:
        fh.write(r['out'])
    _gates(path)


def test_profile_rejects_unknown_names(bst_corpus):
    m = bst.parse(_box(bst_corpus)['out'])
    with pytest.raises(ValueError, match='profile'):
        connector.make_connector(m, A, ('z', 1), B, ('z', -1), 10.0, 10.0, MATERIAL,
                                 profile='round')


# --- the plug ---------------------------------------------------------------------------------
class _DirReader:
    """library.read over the mod tree, where graft2's restaged tiles live."""

    def __init__(self, root):
        self.root = root

    def read(self, name):
        p = os.path.join(self.root, *name.replace('/', '\\').split('\\'))
        if not os.path.isfile(p):
            return None
        return _read(p)


def _spare_opening(r):
    m = bst.parse(r['out'])
    carried = link.openings(m, B)
    unused = [q for q in link.translated_openings(r['donor'], B_SECTION, B_OFFSET)
              if not any(max(abs(x - y) for x, y in zip(_centre(q), _centre(c))) < 1.0
                         for c in carried)]
    assert len(unused) == 1, 'Room26 has one tunnel end that leads nowhere in graft2'
    return m, unused[0]


def _plug_vertices(before, after, room):
    """(decoded mesh, index of the first new vertex) of the mesh the plug grew."""
    for me_b, me_a in zip(before['sections'][room]['meshes'], after['sections'][room]['meshes']):
        if me_a['pkt']['nverts'] != me_b['pkt']['nverts']:
            return geom.decode_mesh(me_a['pkt']), me_b['pkt']['nverts']
    raise AssertionError('no mesh grew')


def test_plug_opening_on_graft2(bst_corpus, tmp_path):
    r = _box(bst_corpus)
    before = r['parsed']
    m, q = _spare_opening(r)
    geom.capture(m)
    f = link.opening_face(m, B, q)
    axis, sign = f['face']
    plane = f['centre'][caps.AX[axis]]
    open_before = caps.mouth_mask(before, B, plane, axis, sign, f['lo'], f['hi'], 1.0).n_open
    assert open_before > 0
    rep = caps.plug_opening(m, B, q)
    assert rep['verts'] == 4 and rep['tris'] == 4 and rep['texel'] is None
    out, after = _round_trip(m)
    assert tp.check_refs(after) == []
    assert tp.check_additive(before, after) == ['section 1 header', 'section 1 blob']
    assert caps.mouth_mask(after, B, plane, axis, sign, f['lo'], f['hi'], 1.0).n_open == 0
    assert after['sections'][B]['bvtflag'] > before['sections'][B]['bvtflag']
    # one flat colour: the four vertices share a lightmap texel and a vertex colour
    g, n0 = _plug_vertices(before, after, B)
    assert len({g['uv'][12][i] for i in range(n0, n0 + 4)}) == 1
    assert len({g['color'][11][i] for i in range(n0, n0 + 4)}) == 1
    p = str(tmp_path / 'graft2_plugged.bst')
    with open(p, 'wb') as fh:
        fh.write(out)
    _gates(p)


def test_plug_takes_the_ambient_texel_with_a_library(bst_corpus):
    """With the room's tiles readable the shared texel is lit, its neighbours lit, and its
    colour the nearest to the room's ambient."""
    if not os.path.isdir(os.path.join(DEMO, 'art', 'lightmap', 'graft2')):
        pytest.skip('needs the shipped mods/TransplantDemo tiles')
    r = _box(bst_corpus)
    before = r['parsed']
    m, q = _spare_opening(r)
    geom.capture(m)
    lib = _DirReader(DEMO)
    texel = caps.ambient_texel(m, B, lib)
    assert texel is not None and all(0.0 < v < 1.0 for v in texel)
    rep = caps.plug_opening(m, B, q, library=lib)
    assert rep['texel'] == texel
    after = bst.parse(bst.build(m))
    g, n0 = _plug_vertices(before, after, B)
    uv = g['uv'][12][n0]
    assert all(g['uv'][12][i] == uv for i in range(n0, n0 + 4))
    assert max(abs(a - b) for a, b in zip(uv, texel)) < 2e-3, 'half floats'
    target = np.asarray(lighting.room_ambient(m, B, lib))
    from gbtvgr.level import textures
    imgs = [textures.decode(lib.read(tp.pod_path(tp.slot_name(
        m['sections'][B]['names3'][k * tp.SLOT:(k + 1) * tp.SLOT]))))[2] for k in range(3)]
    h, w = imgs[0].shape[:2]
    x, y = int(texel[0] * w), int(texel[1] * h)
    for im in imgs:
        block = im[y - 1:y + 2, x - 1:x + 2, :3]
        assert block.shape == (3, 3, 3) and (block.max(axis=2) > 0).all(), 'lit all around'
    mean = np.mean([im[y, x, :3].astype(float) for im in imgs], axis=0)
    assert np.abs(mean - target).max() < 8.0, (mean, target)
    assert caps.ambient_texel(m, B, None) is None


# --- the gate on shipped sets ---------------------------------------------------------------
def test_check_caps_on_shipped_sets(bst_corpus):
    """The two sets graft2 is made of, plus the smallest and the biggest."""
    for name in ('abyss', 'cemetery1', 'firehouse', 'timessquare1'):
        path = os.path.join(bst_corpus, name + '.bst')
        if not os.path.isfile(path):
            continue
        bad = caps.check_caps(bst.parse(_read(path)))
        assert bad == [], (name, bad)


def test_check_caps_skips_hatches_lines_and_skewed_quads(bst_corpus):
    m = _box(bst_corpus)['parsed']
    rec = link.read_doorways(m)[0]
    lo, hi, axis, _plane = caps._doorway_geometry(m, rec)
    assert axis == 'z'
    # a 45 degree quad: the thin coordinate follows the cross coordinate
    skew = dict(rec, verts=[(x, y, z + (x - lo[0])) for x, y, z in rec['verts']])
    assert caps._doorway_geometry(m, skew)[2] is None
    # a hatch: thinnest on y
    flat = dict(rec, verts=[(x, -20.0, z + (x - lo[0]) * 0.3) for x, y, z in rec['verts']])
    assert caps._doorway_geometry(m, flat)[2] == 'y'


# --- the build script -----------------------------------------------------------------------
def test_graft2_build_script_set_stage(game_dir, tmp_path):
    """make_demo2.py's set stage, untouched, asks for profile='mask' and gets the loft:
    every gate clean, one rim-to-rim mesh, both plugs one flat colour."""
    script = os.path.join(DEMO, 'gen', 'make_demo2.py')
    dante = os.path.join(ROOT, 'dante-toolkit', 'src')
    if not os.path.isfile(script) or not os.path.isdir(dante):
        pytest.skip('needs mods/TransplantDemo/gen/make_demo2.py and dante-toolkit')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([SRC, dante, os.path.join(ROOT, 'mods')]),
               GAME_DIR=game_dir)
    stage = str(tmp_path / 'stage')
    r = subprocess.run([sys.executable, script, '--stage', stage, 'set'],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    path = os.path.join(stage, 'sets', 'graft2.bst')
    m = bst.parse(_read(path))
    assert tp.check_refs(m) == []
    assert link.check_links(m) == []
    assert caps.check_caps(m) == [] and caps.check_caps(m, cell=0.5) == []
    assert caps.check_caps(m, cell=0.25) == []
    assert lighting.check_lighting(m) == []
    meshes = [me for me in m['sections'][C]['meshes'] if not (me['flags'] & 0x20000)]
    assert len(meshes) == 1 and meshes[0]['pkt']['nverts'] == 6 * (caps.RIM_N + 1)
    mb = link._mesh_bbox(m['sections'][C])
    for room, face, end in ((A, 5, 2), (B, 2, 5)):
        plane = struct.unpack('<6f', m['sections'][room]['bbox'])[face]
        assert abs(abs(mb[end] - plane) - caps.RIM_LADDER[0]) < 1e-3
    for room in (A, B):
        for me in m['sections'][room]['meshes']:
            if me['flags'] & 0x20000:
                continue
            g = geom.decode_mesh(me['pkt'])
            n = g['nverts']
            last = np.array(g['vec'][0][n - 4:])
            if n < 4 or np.ptp(last[:, 2]) > 1e-3 or np.ptp(last[:, 0]) < 1.0:
                continue
            if len({g['uv'][12][i] for i in range(n - 4, n)}) == 1:
                break
        else:
            raise AssertionError('no flat-colour plug quad in section %d' % room)
    _gates(path)

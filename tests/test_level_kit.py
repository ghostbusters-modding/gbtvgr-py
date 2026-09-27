"""The architectural detail kit, through a recording emitter (no corpus)."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import math

import pytest

from gbtvgr.level import kit


class Rec:
    """Geo's quad protocol, recorded: (sec, mat, room, pts, n, collide, mesh)."""
    def __init__(self, tess=None):
        self.quads, self.cquads = [], []
        if tess is not None:
            self.tess = tess

    def quad(self, sec, mat, room, pts, n, collide=True, mesh=None):
        assert len(pts) == 4 and len(n) == 3
        assert abs(math.sqrt(sum(c * c for c in n)) - 1.0) < 1e-5, 'unit normal'
        self.quads.append((sec, mat, room, [tuple(p) for p in pts], tuple(n),
                           collide, mesh))
        if collide:
            self.cquads.append([tuple(p) for p in pts])

    def collide_quad(self, pts):
        self.cquads.append([tuple(p) for p in pts])

    def by_mat(self, mat):
        return [q for q in self.quads if q[1] == mat]


def centre(pts):
    return tuple(sum(p[k] for p in pts) / len(pts) for k in range(3))


def plane_rect(q):
    """(axis, coord, lo2, hi2) for an axis-aligned quad, else None."""
    pts, n = q[3], q[4]
    k = max(range(3), key=lambda i: abs(n[i]))
    if abs(abs(n[k]) - 1.0) > 1e-6:
        return None
    coords = {round(p[k], 4) for p in pts}
    if len(coords) != 1:
        return None
    o = [i for i in range(3) if i != k]
    lo = tuple(min(p[i] for p in pts) for i in o)
    hi = tuple(max(p[i] for p in pts) for i in o)
    return k, coords.pop(), lo, hi


def coplanar_overlaps(quads):
    """Pairs of axis-aligned quads on one plane whose rects share area."""
    groups = {}
    for q in quads:
        pr = plane_rect(q)
        if pr is not None:
            groups.setdefault((pr[0], pr[1]), []).append((pr[2], pr[3], q))
    bad = []
    for key, rs in groups.items():
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                (alo, ahi, _qa), (blo, bhi, _qb) = rs[a], rs[b]
                if (min(ahi[0], bhi[0]) - max(alo[0], blo[0]) > 1e-4 and
                        min(ahi[1], bhi[1]) - max(alo[1], blo[1]) > 1e-4):
                    bad.append((key, alo, ahi, blo, bhi))
    return bad


def outward(quads, c):
    return all(sum(n[k] * (centre(pts)[k] - c[k]) for k in range(3)) > 0
               for _s, _m, _r, pts, n, _c, _me in quads)


def test_box_six_faces_outward():
    g = Rec()
    boxes = kit.box(g, 0, 'r', 0, 0, 0, 4, 6, 2, 'm', tess=0)
    assert boxes == [(0, 0, 0, 4, 6, 2)]
    assert len(g.quads) == 6
    assert outward(g.quads, (2, 3, 1))
    assert {q[4] for q in g.quads} == {(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                       (0, -1, 0), (0, 0, 1), (0, 0, -1)}
    assert len(g.cquads) == 6, 'each face collides as one quad'


def test_box_skip_and_tessellation():
    g = Rec()
    kit.box(g, 0, 'r', 0, 0, 0, 4, 6, 2, 'm', top=False, skip={'-x'}, tess=0)
    assert len(g.quads) == 4
    assert (0, 1, 0) not in {q[4] for q in g.quads}
    assert (-1, 0, 0) not in {q[4] for q in g.quads}
    g = Rec(tess=2.0)
    kit.box(g, 0, 'r', 0, 0, 0, 4, 6, 2, 'm')
    # +x face is 2 x 6 ft -> 1 x 3 pieces; the collision stays one quad a face
    assert len([q for q in g.quads if q[4] == (1, 0, 0)]) == 3
    assert len(g.cquads) == 6
    g = Rec(tess=2.0)
    kit.box(g, 0, 'r', 0, 0, 0, 4, 6, 2, 'm', tess=1.0)
    assert len([q for q in g.quads if q[4] == (1, 0, 0)]) == 12


def test_step_of_reads_the_emitter():
    class Mod:
        TESS = 3.0

    class G:
        mod = Mod()
    assert kit.step_of(G(), None) == 3.0
    assert kit.step_of(Rec(), None) == kit.DEFAULT_TESS
    assert kit.step_of(Rec(tess=1.5), None) == 1.5
    assert kit.step_of(Rec(), 0) is None
    assert kit.step_of(Rec(), 4.0) == 4.0


def test_column_bands_no_coplanar_faces():
    g = Rec()
    boxes = kit.column(g, 0, 'r', 10, 10, 12, 12, 0, 30, 'shaft', 'band',
                       base_h=3.0, cap_h=1.5, tess=0)
    assert len(boxes) == 3
    assert coplanar_overlaps(g.quads) == []
    band = g.by_mat('band')
    assert band and all(outward([q], (11, 15, 11)) for q in band
                        if abs(q[4][1]) < 0.5), 'band sides face out'
    # the exposed band tops are rings around the shaft, facing up on the base
    # and down on the cap; the shaft has no top or bottom of its own
    ups = [q for q in band if q[4] == (0, 1, 0)]
    downs = [q for q in band if q[4] == (0, -1, 0)]
    assert len(ups) == 4 + 1 and len(downs) == 4, 'rings, plus the cap top'
    assert not [q for q in g.by_mat('shaft') if abs(q[4][1]) > 0.5]
    # standing on the floor: no bottom face at y=0
    assert not [q for q in g.quads if q[4] == (0, -1, 0) and q[3][0][1] == 0]


def test_column_plain_and_top_off():
    g = Rec()
    kit.column(g, 0, 'r', 0, 0, 2, 2, 0, 30, 'm', top=False, tess=0)
    assert len(g.quads) == 4


def test_truss_counts_and_no_collision():
    g = Rec()
    boxes = kit.truss(g, 0, 'r', 'x', 30.0, 0.0, 60.0, 24.0, 4.0, 'm', tess=0)
    # 15 bays: 2 chords x 6, 16 verticals x 4, 15 diagonals x 4
    assert len(g.quads) == 12 + 64 + 60
    assert coplanar_overlaps(g.quads) == []
    g = Rec()
    kit.truss(g, 0, 'r', 'x', 30.0, 0.0, 60.0, 24.0, 4.0, 'm', caps=(False, True), tess=0)
    assert len(g.quads) == 10 + 63 + 60, 'chord ends and the first vertical lose the wall face'
    assert not [q for q in g.quads if all(p[0] == 0.0 for p in q[3])]
    assert g.cquads == []
    assert boxes == [(0.0, 24.0, 29.75, 60.0, 28.0, 30.25)]
    g = Rec()
    kit.truss(g, 0, 'r', 'z', 0.0, 0.0, 12.0, 20.0, 3.0, 'm', bays=2, tess=0)
    assert len(g.quads) == 12 + 12 + 8
    assert all(abs(q[4][0]) < 1e-9 for q in g.quads if 1e-6 < abs(q[4][1]) < 0.99), \
        'a z truss keeps its diagonals in the yz plane'


def test_diagonal_bar_faces_out():
    g = Rec()
    kit._bar(g, 0, 'm', 'r', 'x', 5.0, (0.0, 0.0), (4.0, 3.0), 0.5, False,
             None, 0)
    assert len(g.quads) == 4
    assert outward(g.quads, (2.0, 1.5, 5.0))


def test_stair_slope_matches_treads():
    g = Rec()
    boxes, slope = kit.stair(g, 0, 'r', 'z', +1, 10.0, 20.0, 6.0, 2.0, 0.5, 18,
                             'tread', 'riser', 'stringer', y0=1.0, tess=0)
    tops = [q for q in g.by_mat('tread') if q[4] == (0, 1, 0)]
    assert len(tops) == 18
    assert max(p[1] for q in tops for p in q[3]) == pytest.approx(10.0)
    flat = [c for c in g.cquads if len({p[1] for p in c}) == 1]
    assert len(flat) == 18, 'every tread collides, nothing else flat does'
    assert sorted(c[0][1] for c in flat) == pytest.approx([1.5 + 0.5 * i for i in range(18)])
    risers = [q for q in g.by_mat('riser') if q[4] == (0, 0, -1)]
    assert len(risers) == 18
    assert [p for p in slope] == [(10.0, 1.0, 20.0), (16.0, 1.0, 20.0),
                                  (16.0, 10.0, 56.0), (10.0, 10.0, 56.0)]
    assert boxes == [(10.0 - 0.4, 1.0, 20.0, 16.0 + 0.4, 10.0, 56.0)]
    # the soffit faces down and toward the climb: the space under the flight
    sof = [q for q in g.by_mat('riser') if q[4][1] < 0]
    assert len(sof) == 1 and sof[0][4][2] > 0
    assert abs(sof[0][4][1] * 0.5 + sof[0][4][2] * 2.0) < 1e-9, 'normal to the slope'
    stringers = g.by_mat('stringer')
    assert len(stringers) == 2 * (2 + 3), 'two plates, foot, top and bottom edges'
    plates = [q for q in stringers if abs(q[4][0]) > 0.99]
    assert len(plates) == 4 and sum(q[4][0] for q in plates) == 0
    assert coplanar_overlaps(g.quads) == []
    assert not [q for q in stringers if all(abs(p[2] - 56.0) < 1e-9 for p in q[3])], \
        'no end face against the landing'
    with pytest.raises(AssertionError):
        kit.stair(Rec(), 0, 'r', 'x', +1, 0, 0, 4, 1.4, 0.5, 4, 't', 'r')
    g = Rec()
    _b, slope = kit.stair(g, 0, 'r', 'x', -1, 40.0, 0.0, 4.0, 2.0, 0.5, 4,
                          't', 'r', tess=0)
    assert slope[2][0] == 32.0 and slope[2][1] == 2.0


def test_railing_posts_and_barrier():
    g = Rec()
    boxes = kit.railing(g, 0, 'r', 0.0, 5.0, 24.0, 5.0, 9.0, 'post', 'rail',
                        post_pitch=6.0, tess=0)
    posts = g.by_mat('post')
    assert len(posts) == 5 * 5, 'five posts, five faces each (no bottom)'
    assert len(g.by_mat('rail')) == 2 * 4 * 4, 'two rails, four bays, ends buried'
    assert coplanar_overlaps(g.quads) == []
    assert len(g.cquads) == 25 + 3, 'posts, plus the three-face barrier'
    assert boxes[0][1] == 9.0 and boxes[0][4] == 12.5
    xs = sorted({q[3][0][0] for q in posts if q[4] == (-1, 0, 0)})
    assert xs == pytest.approx([0.0, 5.9375, 11.875, 17.8125, 23.75])
    with pytest.raises(AssertionError):
        kit.railing(Rec(), 0, 'r', 0, 0, 10, 4, 0, 'p', 'r')
    g = Rec()
    kit.railing(g, 0, 'r', 0.0, 5.0, 24.0, 5.0, 9.0, 'post', 'rail',
                wall_ends=(False, True), tess=0)
    assert len(g.by_mat('post')) == 5 * 5 - 1
    assert not [q for q in g.by_mat('post') if all(p[0] == 24.0 for p in q[3])]


def test_railing_without_collide_quad_collides_rails():
    class Plain:
        def __init__(self):
            self.quads = []

        def quad(self, sec, mat, room, pts, n, collide=True, mesh=None):
            self.quads.append((mat, collide))
    g = Plain()
    kit.railing(g, 0, 'r', 0.0, 0.0, 0.0, 12.0, 0.0, 'post', 'rail', tess=0)
    assert all(c for m, c in g.quads if m == 'rail')


def test_platform_posts_and_faces():
    g = Rec()
    posts = kit.platform(g, 0, 'r', 0, 0, 24, 12, 9.0, 0.5, 'deck', 'edge',
                         mat_support='post', post_pitch=12.0, tess=0)
    assert len(posts) == 3 * 2
    assert all(b[1] == 0.0 and b[4] == 8.5 for b in posts)
    assert len(g.by_mat('deck')) == 1 and g.by_mat('deck')[0][4] == (0, 1, 0)
    assert len(g.by_mat('edge')) == 1 + 4
    assert len(g.by_mat('post')) == 6 * 4, 'posts have no top or bottom'
    g = Rec()
    kit.platform(g, 0, 'r', 0, 0, 24, 12, 9.0, 0.5, 'deck', 'edge',
                 supports=False, skip={'-x'}, tess=0)
    assert (-1, 0, 0) not in {q[4] for q in g.quads}
    g = Rec()
    kit.platform(g, 0, 'r', 0, 0, 24, 12, 9.0, 0.5, 'deck', 'edge',
                 mat_support='post', post_pitch=12.0, skip={'+x'}, tess=0)
    assert not [q for q in g.by_mat('post') if q[4] == (1, 0, 0)
                and q[3][0][0] == 24.0], 'posts on the wall side show no wall face'


def test_container_budget_and_ribs():
    g = Rec()
    boxes = kit.container(g, 0, 'r', 0.0, 0.0, 40.0, 8.0, 8.5, 'side', 'end',
                          'top')
    assert len(g.quads) < 200, len(g.quads)
    assert boxes == [(0.0, 0.0, 0.0, 40.0, 8.5, 8.0)]
    assert coplanar_overlaps(g.quads) == []
    assert len(g.by_mat('end')) == 2 and len(g.by_mat('top')) == 1
    zs = {round(p[2], 3) for q in g.by_mat('side') for p in q[3]}
    assert zs == {-0.15, 0.0, 8.0, 8.15}, 'ribs stand 0.15 proud both sides'
    g = Rec()
    kit.container(g, 0, 'r', 0.0, 0.0, 20.0, 8.0, 8.5, 'side', 'end', 'top',
                  axis='z')
    assert len(g.by_mat('end')) == 2
    assert {q[4] for q in g.by_mat('end')} == {(0, 0, 1), (0, 0, -1)}


def test_duct_elbow_no_coplanar():
    g = Rec()
    boxes = kit.duct(g, 0, 'r', [(0, 0), (20, 0), (20, 15)], 20.0, 2.0, 1.5, 'm',
                     tess=0)
    assert len(boxes) == 3
    assert coplanar_overlaps(g.quads) == []
    # two runs shortened into the joint, each open on that end: 5 faces; the
    # joint loses the two faces the runs open into: 4
    assert len(g.quads) == 5 + 5 + 4
    assert (20 - 1.0, 20.0, -1.0, 21.0, 21.5, 1.0) in boxes
    g = Rec()
    kit.duct(g, 0, 'r', [(0, 0), (20, 0)], 20.0, 2.0, 1.5, 'm',
             end_caps=(False, True), tess=0)
    assert len(g.quads) == 5 and (-1, 0, 0) not in {q[4] for q in g.quads}


def test_pipe_facets_and_brackets():
    g = Rec()
    boxes = kit.pipe(g, 0, 'r', 'x', 10.0, 20.0, 0.0, 30.0, 1.0, 'm', tess=0)
    sides = [q for q in g.quads if abs(q[4][0]) < 1e-9]
    assert len(sides) == 8 and len(g.quads) == 8 + 6
    assert outward(sides, (15.0, 20.0, 10.0))
    for q in sides:
        c = centre(q[3])
        assert abs(math.hypot(c[1] - 20.0, c[2] - 10.0) - math.cos(math.radians(22.5))) < 1e-6
    assert boxes == [(0.0, 19.0, 9.0, 30.0, 21.0, 11.0)]
    g = Rec()
    boxes = kit.pipe(g, 0, 'r', 'z', 10.0, 20.0, 0.0, 30.0, 1.0, 'm',
                     brackets=10.0, bracket_to=('y', 24.0), mat_bracket='b',
                     tess=0)
    br = g.by_mat('b')
    assert len(br) == 4 * 4, 'four hangers of four faces'
    assert all(abs(q[4][1]) < 1e-9 for q in br)
    assert len(boxes) == 5
    g = Rec()
    kit.pipe(g, 0, 'r', 'z', 10.0, 20.0, 0.0, 30.0, 1.0, 'm', brackets=15.0,
             bracket_to=('x', 8.0), mat_bracket='b', tess=0)
    assert len(g.by_mat('b')) == 3 * 4


def test_door_frame_clear_of_wall_plane():
    g = Rec()
    boxes = kit.door_frame(g, 0, 'r', 'z', 0.0, 10.0, 16.0, 0.0, 9.0, 0.5, 0.75,
                           'm', sign=+1, tess=0)
    assert len(boxes) == 3
    assert len(g.quads) == 3 + 3 + 4 + 1, 'jambs, lintel, its underside'
    assert not [q for q in g.quads if plane_rect(q) and plane_rect(q)[0] == 2
                and plane_rect(q)[1] == 0.0], 'nothing drawn on the wall plane'
    assert not [q for q in g.quads if q[4] == (0, -1, 0) and q[3][0][1] == 0.0]
    assert coplanar_overlaps(g.quads) == []
    assert min(b[2] for b in boxes) == 0.0 and max(b[5] for b in boxes) == 0.5
    g = Rec()
    kit.door_frame(g, 0, 'r', 'x', 50.0, 10.0, 16.0, 0.0, 9.0, 0.5, 0.75, 'm',
                   sign=-1, tess=0)
    assert all(p[0] <= 50.0 + 1e-9 for q in g.quads for p in q[3])


def test_window_bay_faces_into_pocket():
    g = Rec()
    assert kit.window_bay(g, 0, 'r', 'x', 0.0, 10.0, 16.0, 4.0, 8.0, 1.0,
                          'reveal', 'glass', sign=+1, tess=0) == []
    pane = g.by_mat('glass')
    assert len(pane) == 1 and pane[0][4] == (1, 0, 0)
    assert all(p[0] == -1.0 for p in pane[0][3])
    rev = {q[4]: q for q in g.by_mat('reveal')}
    assert set(rev) == {(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0)}
    assert rev[(0, 0, 1)][3][0][2] == 10.0 and rev[(0, -1, 0)][3][0][1] == 8.0
    g = Rec()
    kit.window_bay(g, 0, 'r', 'z', 30.0, 10.0, 16.0, 4.0, 8.0, 1.0, 'rv', 'gl',
                   sign=-1, tess=0)
    assert g.by_mat('gl')[0][4] == (0, 0, -1)
    assert all(p[2] == 31.0 for p in g.by_mat('gl')[0][3])


def test_cornice_dado_pilasters():
    g = Rec()
    kit.cornice(g, 0, 'r', 'z', 0.0, 0.0, 40.0, 28.0, 30.0, 1.0, 'm', top=False,
                tess=0)
    assert len(g.quads) == 4 and (0, 0, -1) not in {q[4] for q in g.quads}
    assert not any(q[5] for q in g.quads)
    g = Rec()
    kit.dado(g, 0, 'r', 'x', 60.0, 0.0, 40.0, 3.0, 3.5, 0.3, 'm', sign=-1, tess=0)
    assert len(g.quads) == 5 and len(g.cquads) == 5
    g = Rec()
    boxes = kit.pilasters(g, 0, 'r', 'z', 0.0, 0.0, 48.0, 0.0, 30.0, 12.0, 2.0,
                          1.0, 'm', mat_band='b', tess=0)
    assert len(boxes) == 5 * 3
    assert coplanar_overlaps(g.quads) == []
    assert not [q for q in g.quads if plane_rect(q) and plane_rect(q)[0] == 2
                and plane_rect(q)[1] == 0.0]
    assert all(p[2] >= 0.0 for q in g.quads for p in q[3])


def test_gantry_catwalk_ladder_monitor_small():
    g = Rec()
    legs = kit.gantry(g, 0, 'r', 'x', 30.0, 0.0, 60.0, 26.0, 'leg', 'beam',
                      tess=0)
    assert len(legs) == 2 and len(g.by_mat('leg')) == 8
    assert len(g.by_mat('beam')) == 6 and not any(q[5] for q in g.by_mat('beam'))
    g = Rec()
    boxes, slope = kit.catwalk(g, 0, 'r', 0.0, 0.0, 36.0, 6.0, 9.0, 0.5, 'deck',
                               'edge', 'post', 'rail', access='stair', tess=0)
    assert slope is not None and slope[2][1] == 9.0 and slope[0][0] == -36.0
    assert len(g.by_mat('deck')) == 1 + 18, 'the deck and 18 treads'
    g = Rec()
    boxes, slope = kit.catwalk(g, 0, 'r', 0.0, 0.0, 6.0, 36.0, 9.0, 0.5, 'deck',
                               'edge', 'post', 'rail', access='ladder',
                               access_end='hi', tess=0)
    assert slope is None and all(len(b) == 6 for b in boxes)
    g = Rec()
    assert kit.skylight_monitor(g, 0, 'r', 10, 10, 30, 20, 30.0, 8.0, 'w', 'gl',
                                'top', tess=0) == []
    assert len(g.by_mat('gl')) == 4 and len(g.by_mat('w')) == 8
    assert g.by_mat('top')[0][4] == (0, -1, 0)
    assert all(sum(n[k] * ((20, 34, 15)[k] - centre(pts)[k]) for k in range(3)) > 0
               for _s, _m, _r, pts, n, _c, _me in g.quads), 'a monitor faces in'
    g = Rec()
    assert kit.bollard(g, 0, 'r', 5.0, 5.0, 0.0, 'm') == [(4.5, 0.0, 4.5, 5.5, 3.0, 5.5)]
    assert len(g.quads) == 8 + 3 and outward(g.quads, (5.0, 1.0, 5.0))
    g = Rec()
    kit.ladder(g, 0, 'r', 'x', 0.0, +1, 2.0, 4.0, 0.0, 12.0, 'm')
    assert len(g.quads) == 2 * 5 + 11 * 4
    g = Rec()
    assert kit.kerb_run(g, 0, 'r', 0, 0, 40, 1, 0.0, 0.5, 'm', tess=0) == \
        [(0, 0.0, 0, 40, 0.5, 1)]
    assert len(g.quads) == 5
    g = Rec()
    kit.purlins(g, 0, 'r', 'z', 0.0, 60.0, 0.0, 72.0, 8.0, 28.0, 0.5, 'm',
                top=False, tess=0)
    assert len(g.quads) == 10 * 5
    g = Rec()
    kit.purlins(g, 0, 'r', 'z', 0.0, 60.0, 0.0, 72.0, 8.0, 28.0, 0.5, 'm',
                top=False, over=[(11.75, 12.25), (35.75, 36.25)], skip={'-z', '+z'},
                tess=0)
    assert len(g.quads) == 10 * (2 + 3), 'two sides, three underside spans'
    assert not [q for q in g.quads if q[4] == (0, -1, 0)
                and min(p[2] for p in q[3]) < 12.25 < max(p[2] for p in q[3])]


def test_nav_exclusions():
    ex = kit.nav_exclusions([(2, 0, 2, 6, 30, 6), (8, 0, 2, 12, 30, 6)])
    assert ex == [(0.0, 0.0, 12.0, 6.0)], 'two adjacent 4 ft columns merge'
    assert kit.nav_exclusions([(5, 0, 5, 7, 30, 7)]) == [], 'a 2 ft post on a corner'
    assert kit.nav_exclusions([(0, 24, 0, 60, 28, 60)], floor_y=0.0) == []
    assert kit.nav_exclusions([(0, 3, 0, 60, 28, 60)], floor_y=0.0) == \
        [(0.0, 0.0, 60.0, 60.0)]
    ex = kit.nav_exclusions([(2, 0, 2, 6, 9, 6), (2, 0, 8, 6, 9, 12),
                             (20, 0, 20, 22, 9, 22)])
    assert ex == [(0.0, 0.0, 6.0, 12.0), (18.0, 18.0, 24.0, 24.0)]
    assert kit.nav_exclusions([(1, 0, 1, 5, 4, 5)], origin=(-3.0, -3.0)) == \
        [(-3.0, -3.0, 9.0, 9.0)]


def test_everything_returns_boxes():
    g = Rec()
    out = []
    out += kit.box(g, 0, 'r', 0, 0, 0, 1, 1, 1, 'm')
    out += kit.column(g, 0, 'r', 0, 0, 2, 2, 0, 20, 'm', 'b')
    out += kit.beam(g, 0, 'r', 'z', 3.0, 0.0, 30.0, 20.0, 21.0, 1.0, 'm')
    out += kit.truss(g, 0, 'r', 'x', 5.0, 0.0, 40.0, 20.0, 4.0, 'm')
    out += kit.purlins(g, 0, 'r', 'x', 0.0, 40.0, 0.0, 30.0, 6.0, 24.0, 0.5, 'm')
    out += kit.duct(g, 0, 'r', [(0, 0), (10, 0)], 18.0, 2.0, 2.0, 'm')
    out += kit.pipe(g, 0, 'r', 'x', 2.0, 18.0, 0.0, 30.0, 0.5, 'm')
    out += kit.railing(g, 0, 'r', 0, 0, 12, 0, 0.0, 'p', 'r')
    boxes, slope = kit.stair(g, 0, 'r', 'x', 1, 0, 0, 4, 2.0, 0.5, 6, 't', 'r', 's')
    out += boxes
    out += kit.platform(g, 0, 'r', 0, 0, 12, 12, 8.0, 0.5, 'd', 'e')
    out += kit.door_frame(g, 0, 'r', 'x', 0, 2, 6, 0, 8, 0.5, 0.5, 'm')
    out += kit.window_bay(g, 0, 'r', 'x', 0, 2, 6, 3, 7, 1.0, 'm', 'g')
    out += kit.cornice(g, 0, 'r', 'x', 0, 0, 30, 19, 20, 0.5, 'm')
    out += kit.dado(g, 0, 'r', 'x', 0, 0, 30, 3, 3.5, 0.2, 'm')
    out += kit.pilasters(g, 0, 'r', 'x', 0, 0, 30, 0, 20, 10, 2, 1, 'm')
    out += kit.container(g, 0, 'r', 0, 0, 20, 8, 8.5, 's', 'e', 't')
    out += kit.gantry(g, 0, 'r', 'z', 5.0, 0.0, 30.0, 18.0, 'l', 'b')
    boxes, _slope = kit.catwalk(g, 0, 'r', 0, 0, 24, 5, 9.0, 0.5, 'd', 'e', 'p',
                                'r', access='ladder')
    out += boxes
    out += kit.skylight_monitor(g, 0, 'r', 0, 0, 10, 10, 30.0, 6.0, 'w', 'g', 't')
    out += kit.bollard(g, 0, 'r', 1.0, 1.0, 0.0, 'm')
    out += kit.kerb_run(g, 0, 'r', 0, 0, 10, 1, 0.0, 0.5, 'm')
    out += kit.ladder(g, 0, 'r', 'z', 0.0, 1, 0.0, 2.0, 0.0, 10.0, 'm')
    assert out and all(len(b) == 6 and b[0] <= b[3] and b[1] <= b[4]
                       and b[2] <= b[5] for b in out)
    assert len(slope) == 4
    assert all(len(q[3]) == 4 for q in g.quads)

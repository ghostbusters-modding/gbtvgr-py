"""Link: check_links clean on every shipped set, the doorway codec byte-exact, and
graft1's cave linked to the platform through CONNECT's corridor in both forms."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import glob
import os
import struct

import pytest

from _run import gbtvgr, ok
from gbtvgr.level import connector as conn
from gbtvgr.level import link as lk
from gbtvgr.level import transplant as tp
from gbtvgr.sets import bst

GRAFT1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'mods', 'TransplantDemo', 'sets', 'graft1.bst')
HOST, ROOM, CONNECTOR = 1, 2, 3          # graft1: Room02, Room25, the corridor appended last
CAVE = 42                                # Room25 in cemetery1
MATERIAL = 'graveyard\\ugp_rock'
GRAFT2 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'mods', 'TransplantDemo', 'sets', 'graft2.bst')
HARBOR1A = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'mods', 'TheHarbor', 'sets', 'harbor1a.bst')


def _read(path):
    with open(path, 'rb') as fh:
        return fh.read()


def _corpus_set(bst_corpus, name):
    path = os.path.join(bst_corpus, name + '.bst')
    if not os.path.isfile(path):
        pytest.skip('needs %s in the .bst corpus' % name)
    return path


def _gates(path):
    ok(gbtvgr('bst', 'verify', path))
    out = ok(gbtvgr('bst-geom', 'check', path))
    assert 'FAIL' not in out, out


def _components(nav):
    nbs = [tp.nav_neighbours(d) for d in nav['nodes']]
    comp = [-1] * len(nbs)
    n = 0
    for s in range(len(nbs)):
        if comp[s] >= 0:
            continue
        comp[s] = n
        stack = [s]
        while stack:
            x = stack.pop()
            for j in nbs[x]:
                if j >= 0 and comp[j] < 0:
                    comp[j] = n
                    stack.append(j)
        n += 1
    return n


def _graft_offset(m, donor):
    dbb = struct.unpack('<6f', donor['sections'][CAVE]['bbox'])
    gbb = struct.unpack('<6f', m['sections'][ROOM]['bbox'])
    off = tuple(round(gbb[k] - dbb[k], 3) for k in range(3))
    assert all(abs(gbb[k + 3] - dbb[k + 3] - off[k]) < 1e-3 for k in range(3))
    return off


def test_check_links_clean_on_corpus(bst_corpus):
    files = sorted(glob.glob(os.path.join(bst_corpus, '**', '*.bst'), recursive=True))
    assert files
    bad = {}
    for f in files:
        problems = lk.check_links(bst.parse(_read(f)))
        if problems:
            bad[os.path.basename(f)] = problems[:5]
    assert not bad, bad


def test_doorway_roundtrip(bst_corpus):
    for name in ('cemetery1', 'hotel1a'):
        m = bst.parse(_read(_corpus_set(bst_corpus, name)))
        fx = m['fx']
        recs = lk.read_doorways(m)
        assert len(recs) == len(fx) // lk.DOORWAY
        lk.write_doorways(m, recs)
        assert m['fx'] == fx
        assert lk.read_doorways(m) == recs
    quad = [(1.0, 2.0, 3.0), (4.0, 2.0, 3.0), (4.0, 5.0, 3.0), (1.0, 5.0, 3.0)]
    r = lk.pack_doorway({'a': 3, 'b': 7, 'verts': quad})
    assert len(r) == lk.DOORWAY
    assert struct.unpack_from('<Iii', r, 0)[:3] == (2, 3, 7)
    assert struct.unpack_from('<iiII', r, 0x108) == (3, 7, 0, 0)
    assert r[12:0x7C] == lk.DOORWAY_FILL * (0x7C - 12)
    assert lk.read_doorways({'fx': r}) == [{'a': 3, 'b': 7, 'verts': quad}]


def _linked(bst_corpus, form):
    """graft1 + Room25's own nav + CONNECT's corridor, linked. (before, after, report)."""
    if not os.path.isfile(GRAFT1):
        pytest.skip('needs %s' % GRAFT1)
    donor = bst.parse(_read(_corpus_set(bst_corpus, 'cemetery1')))
    before = bst.parse(_read(GRAFT1))
    m = bst.parse(_read(GRAFT1))
    room_nodes = tp.append_nav(m, donor, CAVE, _graft_offset(m, donor))
    assert len(room_nodes) == 34 and m['nav']['f'] == 3
    m = conn.make_connector(m, ROOM, ('x', 1), HOST, ('x', -1), 8.0, 9.0, MATERIAL)
    assert len(m['sections']) == 4
    unjoined = _components(m['nav'])
    report = lk.link_room(m, HOST, CONNECTOR, ROOM, form=form)
    out = bst.build(m)
    after = bst.parse(out)
    assert bst.build(after) == out
    return before, after, out, report, unjoined


def _check_join(nav, side, conn_end, group):
    """RE_SECTION_LINK.md 4.6 on one stitched pair: both directions on the exact edges,
    the connector edge carrying the side node's two vertices reversed, headers filled."""
    (sn, se), (kn, ke) = side, conn_end
    S, K = nav['nodes'][sn], nav['nodes'][kn]
    s_ring, k_ring = tp.nav_ring(S), tp.nav_ring(K)
    assert tp.nav_neighbours(S)[se] == kn
    assert tp.nav_neighbours(K)[ke] == sn
    assert k_ring[ke] == s_ring[(se + 1) % len(s_ring)]
    assert k_ring[(ke + 1) % len(k_ring)] == s_ring[se]
    assert len(k_ring) <= lk.MAX_EDGES
    verts = lk._nav_verts(nav)
    pts = [verts[v] for v in k_ring]
    assert lk._convex_cw(pts), 'the stretched connector polygon is not a convex clockwise ring'
    h = tp.nav_header(K)
    c = [sum(p[k] for p in pts) / len(pts) for k in range(3)]
    assert max(abs(h['centroid'][k] - c[k]) for k in range(3)) < 1e-3
    assert h['hmin'] < h['hmax'] and h['n1'][1] > 0.99
    assert h['group'] == group and tp.nav_header(S)['group'] == group
    e = struct.unpack('<I', K['f15c'])[0]
    assert struct.unpack('<I', nav['extra'][e]['b'])[0] & lk.WALKABLE_MASK


@pytest.mark.parametrize('form', ['doorway', 'b2'])
def test_link_graft1(bst_corpus, tmp_path, form):
    """The cave behind the corridor, both recipes of 1.7. The linked sets are also
    written to $GB_BUILD/backend/link/ for the in-game experiments when it is set."""
    before, after, out, report, unjoined = _linked(bst_corpus, form)
    assert tp.check_refs(after) == []
    assert lk.check_links(after) == []
    touched = ['section %d header' % HOST, 'section %d header' % ROOM, 'nav nodes[:old]']
    if form == 'doorway':
        touched.insert(2, 'fx')
    assert tp.check_additive(before, after) == touched
    # nav: the room's island and the corridor chain joined the platform's component
    nav = after['nav']
    assert unjoined == 4 and _components(nav) == 2 == nav['f']
    assert before['nav']['f'] == 2
    j = report['nav']
    assert j['host_facing'] and j['room_facing']
    assert len(j['connector_nodes']) == 7 and len(j['room_nodes']) == 34
    group = tp.nav_header(nav['nodes'][j['host'][0]])['group']
    assert j['group'] == group
    _check_join(nav, j['host'], j['connector_host'], group)
    _check_join(nav, j['room'], j['connector_room'], group)
    assert j['connector_host'][0] != j['connector_room'][0]
    for i in j['connector_nodes'] + j['room_nodes']:
        assert tp.nav_header(nav['nodes'][i])['group'] == group
    # visibility
    H, C, R = (after['sections'][i] for i in (HOST, CONNECTOR, ROOM))
    if form == 'doorway':
        d1, d2 = report['doorways']
        recs = lk.read_doorways(after)
        assert (recs[d1]['a'], recs[d1]['b'], recs[d2]['a'], recs[d2]['b']) == (HOST, CONNECTOR,
                                                                                CONNECTOR, ROOM)
        q_h, q_r = lk.connector_mouths(after, CONNECTOR, HOST, ROOM)
        assert recs[d1]['verts'] == q_h and recs[d2]['verts'] == q_r
        assert lk._pvs(H) == [lk.door_id(d1), CONNECTOR, lk.door_id(d2), ROOM]
        assert lk._pvs(C) == [lk.door_id(d1), HOST, lk.door_id(d2), ROOM]
        assert lk._pvs(R) == [lk.door_id(d2), CONNECTOR, lk.door_id(d1), HOST]
        assert (H['b2'], C['b2'], R['b2']) == (0, 0, 0)
        assert lk.reachable(after, HOST) == [HOST, ROOM, CONNECTOR]
        assert ROOM in lk.reachable(after, HOST, within=set(lk._pvs(H)) | {HOST})
    else:
        assert report['doorways'] == [] and len(after['fx']) == 0
        assert (C['b2'], R['b2']) == (1, 1)
        assert lk._pvs(H) == [CONNECTOR, ROOM]
        assert lk._pvs(C) == [HOST, ROOM] and lk._pvs(R) == [CONNECTOR, HOST]
        assert tp._u32s(H['arr470']) == [CONNECTOR]
        assert tp._u32s(C['arr470']) == [HOST, ROOM] and tp._u32s(R['arr470']) == [CONNECTOR]
    assert lk.can_see(after, HOST, ROOM) and lk.can_see(after, ROOM, HOST)
    assert lk.can_see(after, HOST, CONNECTOR) and not lk.can_see(after, 0, ROOM)
    assert lk.bsp_leaves(after) >= {HOST, CONNECTOR, ROOM}
    path = str(tmp_path / ('graft1_linked_%s.bst' % form))
    with open(path, 'wb') as fh:
        fh.write(out)
    _gates(path)
    build = os.environ.get('GB_BUILD')
    if build:
        dst = os.path.join(build, 'backend', 'link', 'graft1_linked_%s.bst' % form)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'wb') as fh:
            fh.write(out)
        assert _read(dst) == out


def test_room25_openings(bst_corpus):
    """Room25's real openings sit on its +z face (two) and its ceiling (one), so the
    corridor's +x face carries none; the connector is met at its own mouth there."""
    donor = bst.parse(_read(_corpus_set(bst_corpus, 'cemetery1')))
    faces = [lk.opening_face(donor, CAVE, q)['face'] for q in lk.openings(donor, CAVE)]
    assert sorted(faces) == [('y', 1), ('z', 1), ('z', 1)]
    assert ('x', 1) not in faces
    big = max(lk.openings(donor, CAVE), key=lambda q: lk.opening_face(donor, CAVE, q)['extent'][0])
    f = lk.opening_face(donor, CAVE, big)
    assert len(big) == 4 and f['face'] == ('z', 1)
    assert abs(f['span'][0] - 59.92) < 0.01 and abs(f['span'][1] - 95.44) < 0.01
    assert abs(f['height'][0] + 29.28) < 0.01 and abs(f['height'][1] + 14.12) < 0.01
    # The same quad seen from Room21 across it is that section's -z face.
    assert lk.opening_face(donor, 41, big)['face'] == ('z', -1)
    if os.path.isfile(GRAFT1):
        m = bst.parse(_read(GRAFT1))
        off = _graft_offset(m, donor)
        moved = lk.translated_openings(donor, CAVE, off)
        assert [lk.opening_face(m, ROOM, q)['face'] for q in moved] == faces
        assert abs(lk.opening_face(m, ROOM, moved[1])['centre'][2] - (-91.0 + off[2])) < 0.02


def test_connector_span_follows_an_opening(bst_corpus):
    """make_connector(span=) centres the corridor where an opening says, not on the
    rooms' bbox overlap; the mouths link_room writes follow it."""
    if not os.path.isfile(GRAFT1):
        pytest.skip('needs %s' % GRAFT1)
    m = bst.parse(_read(GRAFT1))
    m = conn.make_connector(m, ROOM, ('x', 1), HOST, ('x', -1), 8.0, 9.0, MATERIAL,
                            span=(30.0, 42.0))
    mb = lk._mesh_bbox(m['sections'][CONNECTOR])
    assert abs(mb[2] - 30.0) < 0.01 and abs(mb[5] - 42.0) < 0.01
    q_h, q_r = lk.connector_mouths(m, CONNECTOR, HOST, ROOM)
    assert {round(p[2], 2) for p in q_h} == {30.0, 42.0} == {round(p[2], 2) for p in q_r}
    assert {round(p[0], 2) for p in q_h} == {-256.28}
    assert {round(p[0], 2) for p in q_r} == {-298.83}


def test_pvs_everything_caps_at_199_nearest():
    """Past the 200 cap, only the 199 nearest by bbox distance survive, self and
    duplicates excluded; none of the real fixtures are big enough to hit this."""
    secs = [{'bbox': struct.pack('<6f', float(i), 0.0, 0.0, float(i) + 1.0, 1.0, 1.0),
            'arr4f4': b''} for i in range(250)]
    m = {'sections': secs, 'fx': b''}
    notes = lk.pvs_everything(m)
    assert len(notes) == 250
    assert sorted(lk._i32s(secs[0]['arr4f4'])) == list(range(1, 200))
    for i, s in enumerate(secs):
        e = lk._i32s(s['arr4f4'])
        assert len(e) <= 199 and i not in e and len(e) == len(set(e))


def test_pvs_everything_graft2():
    """Every section sees every other section and every doorway directly: the debug
    PVS that isolates a render bug from a PVS-authoring one."""
    if not os.path.isfile(GRAFT2):
        pytest.skip('needs %s' % GRAFT2)
    m = bst.parse(_read(GRAFT2))
    ns = len(m['sections'])
    nd = len(lk.read_doorways(m))
    assert lk.pvs_everything(m) == []
    for i, s in enumerate(m['sections']):
        pvs = lk._pvs(s)
        assert sorted(v for v in pvs if not lk.is_door(v)) == [j for j in range(ns) if j != i]
        assert sorted(lk.door_index(v) for v in pvs if lk.is_door(v)) == list(range(nd))
    assert lk.check_links(m, warn=True) == []
    for c in range(ns):
        for r in range(ns):
            if c != r:
                assert lk.can_see(m, c, r), (c, r)


def test_camera_section_audit_graft2():
    """A compact, freshly-linked set: the BSP leaf at every nav centroid agrees with
    the bbox+floor section. The retail picture is noisier, see the harbor1a report."""
    if not os.path.isfile(GRAFT2):
        pytest.skip('needs %s' % GRAFT2)
    m = bst.parse(_read(GRAFT2))
    points = lk._nav_centroids(m)
    assert points
    rows = lk.camera_section_audit(m, points)
    assert [r for r in rows if not r['agree']] == []
    assert lk.camera_audit_warnings(m) == []


def test_camera_section_audit_harbor1a():
    """Informational, not a gate: how often the shipped k-d BSP disagrees with the
    floor on a real authored set, worst five by inset (RE_SECTION_LINK.md 1.2)."""
    if not os.path.isfile(HARBOR1A):
        pytest.skip('needs %s' % HARBOR1A)
    m = bst.parse(_read(HARBOR1A))
    rows = lk.camera_section_audit(m, lk._nav_centroids(m))
    bad = sorted((r for r in rows if not r['agree']), key=lambda r: -r['inset'])
    print('\nharbor1a: %d/%d camera/floor disagreements' % (len(bad), len(rows)))
    for r in bad[:5]:
        print('  bsp %-3s floor %-3s inset %6.2f  %s' % (r['bsp'], r['floor'], r['inset'], r['point']))
    assert len(bad) <= len(rows)


def test_doorway_pvs_warnings():
    """A doorway's far section missing from one side's PVS is flagged, but only in
    warn mode -- 7 of the 20 shipped sets do this legitimately (RE_SECTION_LINK.md 1.6)."""
    if not os.path.isfile(GRAFT2):
        pytest.skip('needs %s' % GRAFT2)
    m = bst.parse(_read(GRAFT2))
    assert lk.doorway_pvs_warnings(m) == []
    r = lk.read_doorways(m)[0]
    pvs = [v for v in lk._pvs(m['sections'][r['a']]) if v != r['b']]
    m['sections'][r['a']]['arr4f4'] = struct.pack('<%di' % len(pvs), *pvs)
    expect = ['warning: doorway 0: section %d PVS omits %d' % (r['a'], r['b'])]
    assert lk.doorway_pvs_warnings(m) == expect
    assert lk.check_links(m) == []
    assert lk.check_links(m, warn=True) == expect


def test_check_links_warn_mode_on_corpus(bst_corpus):
    """warn=True runs clean (no crashes) on the shipped corpus and only ever adds
    'warning: ' lines after the same hard-check prefix check_links always returns."""
    files = sorted(glob.glob(os.path.join(bst_corpus, '**', '*.bst'), recursive=True))
    assert files
    for f in files:
        m = bst.parse(_read(f))
        base = lk.check_links(m)
        full = lk.check_links(m, warn=True)
        assert full[:len(base)] == base
        assert all(w.startswith('warning: ') for w in full[len(base):])

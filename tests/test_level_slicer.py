"""Slicer: hotel2's Room64 seen from inside, from West St and from behind its front, the
lobby leaking through its one-sided wall, slices that pass the set gates, the CLI."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import collections
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

from _run import SRC, gbtvgr, ok
from gbtvgr.level import host, place, slicer
from gbtvgr.level import link as lk
from gbtvgr.level import transplant as tp
from gbtvgr.mesh.bvt import bvt_bbox, bvt_collect
from gbtvgr.sets import bst, geom

ROOM64, ROOM65 = 73, 74                 # the forecourt and the lobby behind its front
STREET, SIDEWALK, FRONT = 30, 19, 2     # Room64: the road, the pavement, the brick front
ENTRANCE = (3, 5, 7)                    # Room64: window and entrance meshes on the front
INSIDE = (-49.0, 0.0, 175.0, 49.0, 12.0, 225.0)      # Room64's own footprint
# West St as harbor1a laid it out, moved back into hotel2's frame (offset 0, -3, -225 undone)
WEST_ST = (-48.0, 3.0, 225.0, 48.0, 15.0, 285.0)
BEHIND = (-48.0, 0.0, 110.0, 48.0, 12.0, 170.0)      # behind the front, in the lobby's footprint
CAVE = 42                                             # cemetery1's Room25, the seed host


def _read(path):
    with open(path, 'rb') as fh:
        return fh.read()


def _corpus_set(bst_corpus, name):
    path = os.path.join(bst_corpus, name + '.bst')
    if not os.path.isfile(path):
        pytest.skip('needs %s in the .bst corpus' % name)
    return path


@pytest.fixture(scope='module')
def hotel2(bst_corpus):
    return bst.parse(_read(_corpus_set(bst_corpus, 'hotel2')))


@pytest.fixture(scope='module')
def inside_report(hotel2):
    return slicer.facing_report(hotel2, 'Room64', INSIDE, eyes=6)


@pytest.fixture(scope='module')
def behind_report(hotel2):
    return slicer.facing_report(hotel2, 'Room64', BEHIND)


def _rows(rep):
    return {r['index']: r for r in rep['meshes']}


def _share(rep, verdict):
    return rep['totals'][verdict] / float(rep['tris'])


# --- synthetic: the ray test on its own ---------------------------------------------------
def _quad(x0, y0, z0, x1, y1, z1, axis, flip=False):
    """Two triangles filling a box face normal to `axis`; winding normal points +axis."""
    lo, hi = np.array([x0, y0, z0], float), np.array([x1, y1, z1], float)
    u, v = [k for k in range(3) if k != axis]
    p = [lo.copy() for _ in range(4)]
    p[1][u] = hi[u]
    p[2][u], p[2][v] = hi[u], hi[v]
    p[3][v] = hi[v]
    tris = [(0, 1, 2), (0, 2, 3)]
    a, b, c = np.array(p[0]), np.array(p[1]), np.array(p[2])
    if (np.cross(b - a, c - a)[axis] > 0) == flip:
        tris = [(t[0], t[2], t[1]) for t in tris]
    return np.array(p), np.array(tris)


def test_wall_blocks_from_its_front_only():
    wall, wt = _quad(-5, 0, 5, 5, 10, 5, axis=2)            # faces +z, toward an eye at z 10
    eye = np.array([[0.0, 5.5, 10.0]])
    target = np.array([[0.0, 5.0, 0.0]])
    occ = slicer.Occluders(wall, wt)
    assert occ.blocked(eye, target - eye).tolist() == [True]
    back, bt = _quad(-5, 0, 5, 5, 10, 5, axis=2, flip=True)  # the same wall, back to the eye
    assert slicer.Occluders(back, bt).blocked(eye, target - eye).tolist() == [False]
    occ = slicer.Occluders(back, bt)
    assert occ.blocked(eye, target - eye, backs_block=True).tolist() == [True]
    # a hit inside the skin of either end does not count
    near, nt = _quad(-5, 0, 0.05, 5, 10, 0.05, axis=2)
    assert slicer.Occluders(near, nt).blocked(eye, target - eye).tolist() == [False]
    assert slicer.Occluders(near, nt).blocked(eye, target - eye, skin=0.0).tolist() == [True]


def test_verdicts_on_a_box_room():
    floor, ft = _quad(-10, 0, -10, 10, 0, 10, axis=1)           # up
    ceil_, ct = _quad(-10, 8, -10, 10, 8, 10, axis=1, flip=True)  # down
    wall, wt = _quad(-10, 0, 10, 10, 8, 10, axis=2, flip=True)    # the +z wall, facing -z (in)
    shelf, st = _quad(-2, 7, -2, 2, 7, 2, axis=1)                 # a top face above the eye
    V = np.concatenate([floor, ceil_, wall, shelf])
    T = np.concatenate([ft, ct + 4, wt + 8, st + 12])
    box = (-10, 0, -10, 10, 8, 10)
    codes, front = slicer.tri_verdicts(V, T, box, slicer.eye_points(box, 2), slicer.Occluders(V, T))
    assert codes[:6].tolist() == [slicer.SEEN] * 6 and front[:6].all()
    assert codes[6:].tolist() == [slicer.UNSEEN] * 2 and front[6:].all()
    outside = (-10, 0, 12, 10, 8, 20)
    codes, front = slicer.tri_verdicts(V, T, outside, slicer.eye_points(outside, 2),
                                       slicer.Occluders(V, T))
    assert codes[4:6].tolist() == [slicer.BACKFACE] * 2 and not front[4:6].any()
    assert codes[:4].tolist() == [slicer.SEEN] * 4      # floor and ceiling, through the culled wall
    low = (-10, -8, 12, 10, -1, 20)
    codes, front = slicer.tri_verdicts(V, T, low, slicer.eye_points(low, 2), slicer.Occluders(V, T))
    assert codes[:2].tolist() == [slicer.BACKFACE] * 2       # the floor, from under it
    assert codes[2:4].tolist() == [slicer.SEEN] * 2     # the ceiling, through the culled floor


def test_eye_points_and_box():
    assert slicer.view_box((5, 1, 9, -5, 0, -9)) == (-5.0, 0.0, -9.0, 5.0, 1.0, 9.0)
    E = slicer.eye_points((0, 0, 0, 30, 20, 30), 3)
    assert E.shape == (9, 3) and E[0].tolist() == [5.0, 5.5, 5.0]
    assert E[-1].tolist() == [25.0, 5.5, 25.0]
    assert slicer.eye_points((0, 0, 0, 30, 3, 30), 1).tolist() == [[15.0, 3.0, 15.0]]
    assert slicer.eye_points((0, 0, 0, 30, 20, 30), 2, height=(5.5, 9.0)).shape == (8, 3)


# --- hotel2's Room64: the reproduction ----------------------------------------------------
def test_room64_seen_from_inside(inside_report):
    rep = inside_report
    assert rep['tris'] == 9657 and len(rep['meshes']) == 157
    assert _share(rep, 'seen') >= 0.80
    assert rep['mesh_verdicts']['seen'] >= 150 and rep['mesh_verdicts']['backface-only'] == 0
    rows = _rows(rep)
    for i in (STREET, SIDEWALK, FRONT) + ENTRANCE:
        assert rows[i]['verdict'] == 'seen' and rows[i]['front_share'] == 1.0, i
    assert rows[STREET]['material'] == 'TimesSquare\\street'
    assert 'embedded' in {r['material'] for r in rep['meshes']}


def test_room64_from_west_st_faces_the_player(hotel2):
    """The forecourt is open on the West St side: its front, road and pavement face the box."""
    rep = slicer.facing_report(hotel2, ROOM64, WEST_ST)
    rows = _rows(rep)
    assert rep['mesh_verdicts']['backface-only'] == 0
    for i in (STREET, SIDEWALK, FRONT) + ENTRANCE:
        assert rows[i]['verdict'] == 'seen' and rows[i]['seen_share'] >= 0.7, i
    assert _share(rep, 'seen') >= 0.6 and _share(rep, 'backface-only') <= 0.25


def test_lobby_leaks_through_its_front_wall(hotel2):
    """Above the entrance arch the front is the lobby's own wall, single-sided toward the
    lobby: from West St its backs face the eye and the pendants behind it read as seen."""
    rep = slicer.facing_report(hotel2, ROOM65, WEST_ST, others=[ROOM64], eyes=2, samples=1)
    V, T, owner = slicer.section_soup(hotel2, ROOM65)
    codes = np.concatenate([r['tri_verdict'] for r in rep['meshes']])
    cen = V[T].mean(axis=1)
    sec = hotel2['sections'][ROOM65]
    mats = np.array([slicer.material_label(hotel2, sec['meshes'][o]) for o in owner])
    band = ((mats == 'hotel\\Stonebricks') & (cen[:, 2] > 173.5) & (cen[:, 1] > 19)
            & (cen[:, 1] < 31) & (np.abs(cen[:, 0]) < 38))
    assert band.sum() >= 10 and (codes[band] == slicer.BACKFACE).all()
    pendants = [r for r in rep['meshes'] if r['material'] == 'Hotel\\Pendant']
    assert len(pendants) == 4
    assert all(r['verdict'] == 'seen' and r['seen_share'] > 0.5 for r in pendants)
    brick = [r for r in rep['meshes'] if r['material'] == 'hotel\\Stonebricks' and r['back_bbox']]
    assert brick and max(r['back_bbox'][5] for r in brick) >= 174.0


def test_room64_from_behind_shows_backs(behind_report):
    rep = behind_report
    rows = _rows(rep)
    backs = [r['index'] for r in rep['meshes'] if r['verdict'] == 'backface-only']
    assert len(backs) >= 10 and set(ENTRANCE) <= set(backs)
    assert all(rows[i]['seen_share'] == 0.0 for i in ENTRANCE)
    assert rows[STREET]['verdict'] == 'seen' and rows[SIDEWALK]['verdict'] == 'seen'
    assert rows[FRONT]['verdict'] == 'seen' and rows[FRONT]['counts']['backface-only'] > 0
    text = slicer.format_report(rep)
    assert 'backface-only 12' in text or 'backface-only %d' % len(backs) in text
    assert 'backs at' in text


# --- slices ---------------------------------------------------------------------------------
def _gates(path):
    ok(gbtvgr('bst', 'verify', path))
    out = ok(gbtvgr('bst-geom', 'check', path))
    assert 'FAIL' not in out, out


def test_slice_from_inside_keeps_the_room(hotel2, inside_report, bst_corpus, tmp_path):
    rec, rep = slicer.slice_section(hotel2, ROOM64, report=inside_report)
    src = hotel2['sections'][ROOM64]
    st, ss = geom.section_stats(rec), geom.section_stats(src)
    assert st['tris'] >= 0.80 * ss['tris'] and st['meshes'] >= 150
    kept = slicer.kept_indices(rep)
    assert len(kept) == len(rec['meshes'])
    for mi, me in zip(kept, rec['meshes']):
        orig = src['meshes'][mi]
        assert me['name'] == orig['name'] and me['h70'] == orig['h70'] and me['f6c'] == orig['f6c']
        p, q = orig['pkt'], me['pkt']
        assert q['decl'] == p['decl'] and q['nverts'] <= p['nverts'] and q['nprims'] <= p['nprims']
        st_ = len(p['vdata']) // p['nverts']
        pool = set(p['vdata'][v * st_:(v + 1) * st_] for v in range(p['nverts']))
        assert all(q['vdata'][v * st_:(v + 1) * st_] in pool for v in range(q['nverts']))
        g = geom.decode_mesh(q)
        assert len(g['tris']) == q['nprims'] and max(max(t) for t in g['tris']) < q['nverts']
        assert set(g['uv']) == set(geom.decode_mesh(p)['uv'])
    assert rec['names3'] == src['names3'] and rec['lrefs'] == src['lrefs']
    # the record's blob is laid out, so a fresh parse that takes it in captures the right ballast
    fresh = bst.parse(_read(_corpus_set(bst_corpus, 'hotel2')))
    fresh['sections'][ROOM64] = rec
    geom.capture(fresh)
    assert fresh['sections'][ROOM64]['ballast'] == src['ballast'] and len(src['ballast']) > 0
    # the BVT is rebuilt over the kept triangles, both windings, inside the new bbox
    cv, ct, cs, cf = bvt_collect(rec['bvt'])
    assert len(ct) == 2 * st['tris'] and rec['bvtflag'] > 0
    assert max(cs) < len(rec['lrefs']) and collections.Counter(cs)[2] > 0
    bb, root = struct.unpack('<6f', rec['bbox']), bvt_bbox(rec['bvt'])
    assert all(bb[k] <= root[k] + 1e-3 for k in range(3))
    assert all(root[k] <= bb[k] + 1e-3 for k in range(3, 6))
    assert bb[0] >= -49.02 and bb[3] <= 49.02 and bb[2] >= 174.48 and bb[5] <= 225.02
    # placed into a seed it passes every gate the transplant path passes
    cemetery1 = bst.parse(_read(_corpus_set(bst_corpus, 'cemetery1')))
    seed, _ = host.new_set_from_section(cemetery1, CAVE, 'slicetest')
    seed = bst.parse(bst.build(seed))
    donor = dict(hotel2)
    donor['sections'] = list(hotel2['sections'])
    donor['sections'][ROOM64] = rec
    seed, staged, idx = place.place_section(seed, donor, ROOM64, (0.0, 0.0, 200.0),
                                            lightmap_dir='slicetest')
    data = bst.build(seed)
    back = bst.parse(data)
    assert idx == 1 and bst.build(back) == data
    assert tp.check_refs(back) == [] and lk.check_links(back) == []
    names3 = back['sections'][1]['names3']
    assert [tp.slot_name(names3[k * tp.SLOT:(k + 1) * tp.SLOT]) for k in range(3)] == \
        ['lightmap\\slicetest\\1_%d.tga' % k for k in range(3)]
    path = str(tmp_path / 'room64_inside.bst')
    with open(path, 'wb') as fh:
        fh.write(data)
    _gates(path)


def test_slice_from_behind_drops_the_front(hotel2, behind_report):
    rec, rep = slicer.slice_section(hotel2, ROOM64, report=behind_report)
    kept = slicer.kept_indices(rep)
    assert not set(ENTRANCE) & set(kept) and STREET in kept and SIDEWALK in kept
    assert len(kept) == rep['mesh_verdicts']['seen'] == 145
    st = geom.section_stats(rec)
    assert st['tris'] == rep['totals']['seen']
    # whole meshes for scenery: the same meshes, none of them cut
    bd, brep = slicer.backdrop(hotel2, ROOM64, BEHIND, report=behind_report)
    assert slicer.kept_indices(brep, tris=False) == \
        [r['index'] for r in brep['meshes'] if r['verdict'] == 'seen']
    src = hotel2['sections'][ROOM64]
    for mi, me in zip(slicer.kept_indices(brep, tris=False), bd['meshes']):
        assert me['pkt']['nprims'] == src['meshes'][mi]['pkt']['nprims']
    assert geom.section_stats(bd)['tris'] > st['tris']
    # keep 'keep' honest: dropping nothing but backs keeps every other triangle
    all_, _r = slicer.slice_section(hotel2, ROOM64, report=behind_report,
                                    keep=('seen', 'occluded', 'unseen'), bvt='keep')
    assert geom.section_stats(all_)['tris'] == rep['tris'] - rep['totals']['backface-only']
    assert all_['bvt'] is src['bvt'] or bvt_collect(all_['bvt'])[1] == bvt_collect(src['bvt'])[1]
    none, _r = slicer.slice_section(hotel2, ROOM64, report=behind_report, bvt='drop')
    assert none['bvt'] is None and none['bvtflag'] == 0


def test_two_sided_doubles_and_flips(hotel2):
    rec = slicer.two_sided(hotel2, ROOM64, [FRONT, 'hotel_lobby_renovated01_42'])
    src = hotel2['sections'][ROOM64]
    for mi, me in enumerate(rec['meshes']):
        orig = src['meshes'][mi]
        if mi not in (FRONT, 146):
            assert me['pkt'] == orig['pkt']
            continue
        g0, g1 = geom.decode_mesh(orig['pkt']), geom.decode_mesh(me['pkt'])
        n = g0['nverts']
        assert g1['nverts'] == 2 * n and len(g1['tris']) == 2 * len(g0['tris'])
        assert g1['decl'] == g0['decl'] and me['pkt']['bbox'] == orig['pkt']['bbox']
        assert g1['vec'][0][n:] == g0['vec'][0]
        assert g1['uv'] == {s: v + v for s, v in g0['uv'].items()}
        assert g1['vec'][1][n:] == [(-x, -y, -z) for x, y, z in g0['vec'][1]]
        assert g1['vec'][8][n:] == [(-x, -y, -z) for x, y, z in g0['vec'][8]]
        assert g1['vec'][5][n:] == g0['vec'][5]
        assert g1['tris'][len(g0['tris']):] == [(a + n, c + n, b + n) for a, b, c in g0['tris']]
        P = np.array(g1['vec'][0])
        for t0, t1 in zip(g0['tris'], g1['tris'][len(g0['tris']):]):
            n0 = np.cross(P[t0[1]] - P[t0[0]], P[t0[2]] - P[t0[0]])
            n1 = np.cross(P[t1[1]] - P[t1[0]], P[t1[2]] - P[t1[0]])
            assert np.allclose(n0, -n1)
        assert geom.encode_mesh(geom.decode_mesh(me['pkt']))['vdata'] == me['pkt']['vdata']
    assert rec['bbox'] == src['bbox']
    with pytest.raises(slicer.SliceError):
        slicer.two_sided(hotel2, ROOM64, ['no_such_mesh'])


# --- CLI --------------------------------------------------------------------------------------
def _slicer(*args):
    env = dict(os.environ, PYTHONPATH=SRC)
    return subprocess.run([sys.executable, '-m', 'gbtvgr.level.slicer'] + [str(a) for a in args],
                          capture_output=True, text=True, env=env)


def test_cli_round_trip(bst_corpus, tmp_path):
    path = _corpus_set(bst_corpus, 'hotel2')
    box = [str(v) for v in BEHIND]
    out = ok(_slicer('report', path, 'Room64', '--box', *box, '--eyes', 2, '--samples', 1))
    assert out.startswith('Room64 (section 73): 157 meshes, 9657 triangles')
    assert 'backface-only' in out
    dst = str(tmp_path / 'r64slice.bst')
    out = ok(_slicer('slice', path, ROOM64, '--box', *box, '--eyes', 2, '--samples', 1, '-o', dst))
    assert 'wrote %s' % dst in out and 'lightmap\\r64slice\\0_0.tex' in out
    m = bst.parse(_read(dst))
    assert len(m['sections']) == 1 and geom.name_str(m['sections'][0]['name']) == 'Room64'
    assert tp.check_refs(m) == [] and lk.check_links(m) == []
    st = geom.section_stats(m['sections'][0])
    assert st['meshes'] < 157 and 0 < st['tris'] < 9657 and st['colltris'] == 2 * st['tris']
    _gates(dst)
    out = ok(_slicer('report', dst, 0, '--box', *box, '--eyes', 2, '--samples', 1))
    assert out.startswith('Room64 (section 0): %d meshes, %d triangles'
                          % (st['meshes'], st['tris']))
    assert 'backface-only 0' in out.splitlines()[2]
    dst2 = str(tmp_path / 'r64backdrop.bst')
    out = ok(_slicer('backdrop', path, 'Room64', '--box', *box, '--eyes', 2, '--samples', 1,
                     '-o', dst2, '--bvt', 'keep'))
    m2 = bst.parse(_read(dst2))
    assert geom.section_stats(m2['sections'][0])['colltris'] == 758
    bad = _slicer('slice', path, 'Room64', '--box', *box, '-o', dst, '--keep', 'bogus')
    assert bad.returncode == 2 and 'unknown verdict' in bad.stderr

"""A generated corridor appended between two existing sections: the loft, the box."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import os
import struct

import numpy as np
import pytest

from _run import gbtvgr, ok
from gbtvgr.level import caps
from gbtvgr.level import connector as conn
from gbtvgr.level import transplant as tp
from gbtvgr.mesh.bvt import bvt_bbox
from gbtvgr.sets import bst

GRAFT1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'mods', 'TransplantDemo', 'sets', 'graft1.bst')


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


def _nav_chain(nav, n_old):
    """(is one component, symmetric, no link leaving the new range)."""
    def nb_of(d):
        ec = len(d['v']) // 4
        return list(struct.unpack('<%di' % ec, d['nb']))
    adj = {i: [n for n in nb_of(d) if n >= 0] for i, d in enumerate(nav['nodes'])}
    new_idx = set(range(n_old, nav['nnodes']))
    if not new_idx:
        return False, False, []
    start = min(new_idx)
    seen, q = {start}, [start]
    while q:
        c = q.pop()
        for n in adj[c]:
            if n in new_idx and n not in seen:
                seen.add(n)
                q.append(n)
    one_chain = seen == new_idx
    symmetric = all(i in adj[n] for i in new_idx for n in adj[i])
    leaks = [(i, n) for i in new_idx for n in adj[i] if n not in new_idx]
    return one_chain, symmetric, leaks


def _check_connector(before, after, n_old_sections, n_old_nav):
    """The full battery: structurally clean and additive (the wedge-split BSP
    shape is one check_additive knows)."""
    assert tp.check_refs(after) == []
    diffs = tp.check_additive(before, after)
    assert diffs == [], diffs
    new = after['sections'][-1]
    bb = struct.unpack('<6f', new['bbox'])
    rb = bvt_bbox(new['bvt'])
    exc = max([bb[k] - rb[k] for k in range(3)] + [rb[k] - bb[k] for k in range(3, 6)])
    assert exc <= 0.01, 'BVT root escapes its own section bbox by %.3f' % exc
    one_chain, symmetric, leaks = _nav_chain(after['nav'], n_old_nav)
    assert one_chain, 'the new nav nodes are not a single chain'
    assert symmetric, 'a new nav link is one-way'
    assert leaks == [], 'a new nav node links to a pre-existing one (LINK\'s job, not CONNECT\'s)'


def test_connector_between_two_sections(bst_corpus, tmp_path):
    """abyss section 1 (Room02) to section 0 (room01): a real interior corridor.

    insert_bsp_leaf only wraps the tree as a bookend (new leaf entirely outside
    every existing section), which a corridor between two rooms never is: it
    shares its meeting faces with sections that are already part of the tree's
    envelope. make_connector falls back to its own local wedge split, which
    grows the bsp by 3 nodes instead of insert_bsp_leaf's 2 -- check_additive's
    bsp diff assumes the latter shape, so it reports the count change; every
    other additivity check (materials, sections, nav, lights, portals) is
    clean, verified separately below.
    """
    path = _corpus_set(bst_corpus, 'abyss')
    before = bst.parse(_read(path))
    n_old_nav = before['nav']['nnodes']
    host = bst.parse(_read(path))
    host = conn.make_connector(host, 1, ('x', 1), 0, ('x', -1), 10.0, 10.0,
                               'graveyard\\pavementwet')
    out = bst.build(host)
    after = bst.parse(out)
    assert bst.build(after) == out
    assert len(after['sections']) == len(before['sections']) + 1
    _check_connector(before, after, len(before['sections']), n_old_nav)

    mid_x = (struct.unpack('<6f', before['sections'][1]['bbox'])[3]
             + struct.unpack('<6f', before['sections'][0]['bbox'])[0]) / 2.0
    new = after['sections'][-1]
    mid_z = (struct.unpack('<6f', new['bbox'])[2] + struct.unpack('<6f', new['bbox'])[5]) / 2.0
    floor_y = conn.floor_height(new['bvt'], mid_x, mid_z)
    assert floor_y is not None
    assert abs(floor_y - struct.unpack('<6f', new['bbox'])[1] - 10.0) < 1.0, \
        'the probed floor should sit near the mesh floor, not the padded bbox edge'

    path_out = str(tmp_path / 'abyss_connector.bst')
    with open(path_out, 'wb') as fh:
        fh.write(out)
    _gates(path_out)


def test_connector_reuses_an_existing_material(bst_corpus):
    """material_name already in the host's table: no duplicate is appended."""
    path = _corpus_set(bst_corpus, 'abyss')
    before = bst.parse(_read(path))
    existing = before['materials'][0]['ref'][0].decode('latin1')
    host = bst.parse(_read(path))
    host = conn.make_connector(host, 1, ('x', 1), 0, ('x', -1), 10.0, 10.0, existing)
    assert len(host['materials']) == len(before['materials'])
    idx = [struct.unpack('<H', me['h70'])[0] for me in host['sections'][-1]['meshes']]
    assert idx == [0] * len(idx)


def test_connector_rejects_a_corridor_through_another_room():
    """Room25 straight to room01 would pass through Room02, which sits between
    them: make_connector must refuse rather than build intersecting geometry."""
    if not os.path.isfile(GRAFT1):
        pytest.skip('needs %s' % GRAFT1)
    host = bst.parse(_read(GRAFT1))
    with pytest.raises(ValueError, match='Room02'):
        conn.make_connector(host, 2, ('x', 1), 0, ('x', -1), 8.0, 9.0,
                            'graveyard\\ugp_rock')


def test_connector_graft1_second_append(tmp_path):
    """A second append on top of the wave-2 transplant: Room25 (section 2) to
    Room02 (section 1), the one straight path between them that is actually
    clear. Proves the mechanism composes with an already-grafted set."""
    if not os.path.isfile(GRAFT1):
        pytest.skip('needs %s' % GRAFT1)
    before = bst.parse(_read(GRAFT1))
    n_old_nav = before['nav']['nnodes']
    host = bst.parse(_read(GRAFT1))
    host = conn.make_connector(host, 2, ('x', 1), 1, ('x', -1), 8.0, 9.0,
                               'graveyard\\ugp_rock')
    out = bst.build(host)
    after = bst.parse(out)
    assert bst.build(after) == out
    assert len(after['sections']) == len(before['sections']) + 1
    _check_connector(before, after, len(before['sections']), n_old_nav)

    path_out = str(tmp_path / 'graft1_connector.bst')
    with open(path_out, 'wb') as fh:
        fh.write(out)
    _gates(path_out)


def test_corridor_mesh_is_open_at_both_ends():
    """The two faces the corridor should meet carry no geometry: an inline
    quad wall there would seal the room off instead of connecting it."""
    import numpy as np
    for axis, la in (('x', 0), ('z', 2)):
        m = conn._corridor_mesh(axis, 0.0, 40.0, -5.0, 5.0, 100.0, 10.0, 8.0)
        n = m.face_normals()
        assert m.nfaces == 8, 'floor + ceiling + two side walls, no end caps'
        assert not np.any(np.abs(n[:, la]) > 0.99), 'an end-cap face survived deletion'


# --- the loft -----------------------------------------------------------------------------
def _rect_rim(axis, sign, plane, c0, c1, y0, y1, n=96, depth=0.15):
    """A clamped rim: the fan from the rectangle's centre cut at its boundary."""
    cc, cy = (c0 + c1) / 2.0, (y0 + y1) / 2.0
    th = 2.0 * np.pi * np.arange(n) / n
    d = np.stack([np.cos(th), np.sin(th)], axis=1)
    t = caps._rect_exit(cc, cy, d, (c0, c1), (y0, y1))
    pts = np.array([cc, cy])[None, :] + t[:, None] * d
    return caps.Rim(axis, sign, plane, (cc, cy), pts, np.full(n, depth), np.ones(n, bool))


@pytest.mark.parametrize('axis', ['x', 'z'])
def test_loft_between_two_rectangles_is_a_box_tube(axis):
    """Rims clamped to a rectangle loft to that rectangle: four rings, every vertex on
    its edges, faces inward, no end walls, lightmap uvs 0..1 along and i/n around."""
    lo = _rect_rim(axis, 1, 10.0, -4.0, 4.0, 0.0, 8.0)
    hi = _rect_rim(axis, -1, 40.0, -4.0, 4.0, 0.0, 8.0)
    assert lo.is_rect(1e-9) and hi.is_rect(1e-9)
    mesh, lmuv = conn._loft_mesh(axis, lo, hi, 8.0)
    n = lo.n
    ax, ca = conn.AX[axis], conn.AX['z' if axis == 'x' else 'x']
    # clamped rims get no flange, so the collar is the 0.02 lip alone
    assert mesh.nverts == 6 * n and mesh.nfaces == 5 * 2 * n and lmuv.shape == (mesh.nfaces, 3, 2)
    v = mesh.verts
    assert np.allclose(np.unique(np.round(v[:, ax].astype(float), 3)), [9.85, 10.0, 40.0, 40.15])
    on_edge = np.minimum(np.minimum(abs(v[:, ca] + 4.0), abs(v[:, ca] - 4.0)),
                         np.minimum(abs(v[:, 1]), abs(v[:, 1] - 8.0)))
    on_plane = (np.abs(v[:, ax] - 10.0) < 1e-4) | (np.abs(v[:, ax] - 40.0) < 1e-4)
    assert on_edge[on_plane].max() < 1e-4, 'the rings at the planes are the rectangle'
    assert on_edge[~on_plane].max() <= conn.RING_LIP + 1e-4, 'the end rings sit just inside'
    ends = v[~on_plane]
    assert np.all(np.abs(ends[:, ca]) <= 4.0 + 1e-6) and np.all(ends[:, 1] >= -1e-6) \
        and np.all(ends[:, 1] <= 8.0 + 1e-6)
    fn = mesh.face_normals()
    cen = mesh.face_centers()
    to_axis = -cen.copy()
    to_axis[:, ax] = 0.0
    to_axis[:, 1] = 4.0 - cen[:, 1]
    tube = (cen[:, ax] > 9.85 + 1e-4) & (cen[:, ax] < 40.15 - 1e-4)
    assert np.all((fn * to_axis).sum(axis=1)[tube] > 0), 'every tube face looks at the axis'
    assert not np.any(np.abs(fn[tube, ax]) > 0.5), 'no end wall'
    # the flanges at the ends face their rooms
    assert np.all(fn[np.abs(cen[:, ax] - 9.85) < 1e-4, ax] < 0)
    assert np.all(fn[np.abs(cen[:, ax] - 40.15) < 1e-4, ax] > 0)
    u, w = lmuv[:, :, 0], lmuv[:, :, 1]
    assert u.min() == 0.0 and u.max() == 1.0 and w.min() == 0.0 and w.max() == 1.0
    assert set(np.round(w.ravel() * n).astype(int)) == set(range(n + 1))
    assert np.all(np.diff(u[np.argsort(cen[:, ax])].mean(axis=1)) >= -1e-6)
    # the material uv runs along the axis in u and around the ring in v, world units over 8
    assert abs(mesh.uvs[:, :, 1].max() - lo.arcs[-1] / 8.0) < 1e-4
    assert abs(mesh.uvs[:, :, 0].min() - (10.0 - 0.15) / 8.0) < 1e-4


def test_loft_rims_must_match():
    lo = _rect_rim('z', 1, 0.0, -4.0, 4.0, 0.0, 8.0, n=96)
    hi = _rect_rim('z', -1, 30.0, -4.0, 4.0, 0.0, 8.0, n=64)
    with pytest.raises(ValueError, match='lofted'):
        conn._loft_mesh('z', lo, hi, 8.0)


def test_corridor_nav_grades_between_its_ends():
    """end=(lo, hi, y) at la_hi: width and height run linearly along the chain; without
    it the strip is the flat box strip."""
    def chain(end):
        m = {'nav': {'empty': True}}
        conn._append_corridor_nav(m, 'z', 0.0, 30.0, -2.0, 2.0, 10.0, 6.0, end=end)
        nav = m['nav']
        verts = [struct.unpack_from('<3f', nav['verts'], i * 12) for i in range(nav['nverts'])]
        return nav, verts
    nav, verts = chain(None)
    assert nav['nnodes'] == 5 and len(verts) == 12
    assert {round(v[1], 6) for v in verts} == {10.0} and {round(v[0], 6) for v in verts} == {-2.0, 2.0}
    nav, verts = chain((-3.0, 1.0, 13.0))
    for k in range(6):
        f = k / 5.0
        a, b = verts[2 * k], verts[2 * k + 1]
        assert abs(a[2] - 6.0 * k) < 1e-5 and abs(b[2] - 6.0 * k) < 1e-5
        assert abs(a[1] - (10.0 + 3.0 * f)) < 1e-5 and abs(b[1] - (10.0 + 3.0 * f)) < 1e-5
        assert abs(a[0] - (2.0 - f)) < 1e-5 and abs(b[0] - (-2.0 - f)) < 1e-5
    for i, d in enumerate(nav['nodes']):
        h = tp.nav_header(d)
        assert abs(h['centroid'][1] - (10.0 + 3.0 * (i + 0.5) / 5.0)) < 1e-4
        assert abs(h['hmin'] - (h['centroid'][1] - 1.0)) < 1e-4

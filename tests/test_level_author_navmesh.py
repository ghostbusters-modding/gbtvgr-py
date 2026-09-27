"""Navmesh generation over a floor, an obstacle, a ramp and a platform."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import numpy as np

from gbtvgr.sets.bst import read_nav, write_nav
from gbtvgr.wire import R, W

from gbtvgr.level import navmesh as nm
from gbtvgr.level import geometry_primitives as primitives
from gbtvgr.level.scene import Document, MeshNode, SectionNode


def _mesh_node(name, mesh, pos=(0, 0, 0), walkable=True, collide=True):
    n = MeshNode(name)
    n.mesh = mesh
    n.pos = np.array(pos, np.float32)
    n.props['walkable'] = walkable
    n.props['collide'] = collide
    return n


def _scene():
    doc = Document('navtest')
    sec = SectionNode('room')
    sec.set_bounds((-40, -5, -40), (40, 30, 40))
    doc.add_node(sec)
    doc.add_node(_mesh_node('floor', primitives.plane(60, 60, 1)), parent=sec)
    doc.add_node(_mesh_node('crate', primitives.box(8, 8, 8), (10, 0, 10), walkable=False),
                 parent=sec)
    doc.add_node(_mesh_node('ramp', primitives.ramp(24, 12, 8), (-20, 0, -24)), parent=sec)
    doc.add_node(_mesh_node('platform', primitives.plane(12, 12, 1), (-20, 8, 6)), parent=sec)
    return doc


def _build():
    doc = _scene()
    world = nm.WalkableWorld.from_document(doc)
    return nm.build_navmesh(world, cell=6.0, max_step=2.0, max_slope_deg=40.0, agent_height=8.0)


def test_cells_land_where_they_should():
    r = _build()
    assert not r.empty, r.report
    ys = {n['key']: n['centre'][1] for n in r.nodes}
    floor = nm.node_at(r, 20.0, 20.0)
    ramp = nm.node_at(r, -20.0, -12.0)
    plat = nm.node_at(r, -20.0, 6.0)
    assert floor is not None and abs(r.nodes[floor]['centre'][1]) < 0.01
    assert ramp is not None and 1.0 < r.nodes[ramp]['centre'][1] < 7.0
    assert plat is not None and abs(r.nodes[plat]['centre'][1] - 8.0) < 0.01
    for n in r.nodes:
        cx, _cy, cz = n['centre']
        assert not (6.0 < cx < 14.0 and 6.0 < cz < 14.0), 'a node inside the crate'
    assert len(ys) == len(r.nodes)


def test_one_island_and_clean_selftest():
    r = _build()
    assert len(r.islands) == 1, r.report
    probes = [('floor', 20.0, 20.0), ('ramp', -20.0, -12.0), ('platform', -20.0, 8.0, 6.0)]
    assert nm.nav_selftest(r, probes) == []
    climb = 6.0 * np.tan(np.radians(40.0)) + 2.0
    for n in r.nodes:
        for m in n['nb']:
            if m >= 0:
                assert abs(r.nodes[m]['centre'][1] - n['centre'][1]) <= climb


def test_platform_edge_has_no_cliff_link():
    r = _build()
    plat = nm.node_at(r, -20.0, 6.0)
    for m in r.nodes[plat]['nb']:
        if m >= 0:
            assert r.nodes[m]['centre'][1] > 5.0, 'the platform links only to the ramp top'


def test_nav_dict_roundtrips_through_codec():
    r = _build()
    w = W()
    write_nav(w, r.nav)
    blob = w.data()
    back = read_nav(R(blob))
    w2 = W()
    write_nav(w2, back)
    assert w2.data() == blob
    assert back['nnodes'] == len(r.nodes) and back['nverts'] == len(r.verts)
    verts, rings = nm.nav_from_dict(back)
    assert len(rings) == len(r.nodes) and len(verts) == len(r.verts)
    v2, rings2 = nm.nav_to_polys(r)
    assert rings2 == rings


def test_exclusion_box_blocks_cells():
    doc = _scene()
    world = nm.WalkableWorld.from_document(doc)
    world.exclusions.append((np.array([-10.0, -1.0, 16.0]), np.array([0.0, 6.0, 26.0])))
    r = nm.build_navmesh(world, cell=6.0)
    for n in r.nodes:
        cx, _cy, cz = n['centre']
        assert not (-10.0 < cx < 0.0 and 16.0 < cz < 26.0)


def test_empty_world():
    world = nm.WalkableWorld(np.zeros((0, 3)), np.zeros((0, 3), np.int64))
    r = nm.build_navmesh(world)
    assert r.empty and r.nav['empty']
    assert nm.nav_selftest(r) == ['navmesh is empty']

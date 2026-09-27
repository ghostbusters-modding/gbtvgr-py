"""Editable meshes, primitives and terrain."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import numpy as np

from gbtvgr.level import geometry_primitives as primitives
from gbtvgr.level.geometry_mesh import EditMesh, MAX_VERTS
from gbtvgr.level.geometry_terrain import Terrain


def test_box_normals_face_out():
    b = primitives.box(4, 4, 4)
    n = b.face_normals()
    c = b.face_centers() - np.array([0, 2, 0], np.float32)
    assert np.all(np.einsum('ij,ij->i', n, c) > 0), 'every face normal points away from the centre'
    assert primitives.box(4, 4, 4, inside=True).face_normals()[0] @ n[0] < 0


def test_roundtrip_dict():
    m = primitives.cylinder(3, 6, 12)
    m2 = EditMesh.from_dict(m.to_dict())
    assert np.array_equal(m.verts, m2.verts) and np.array_equal(m.faces, m2.faces)
    assert np.array_equal(m.uvs, m2.uvs)


def test_split_corners_welds_shared():
    p = primitives.plane(10, 10, 4)
    pos, nrm, uv, idx, _cmap, _first = p.split_corners()
    assert len(pos) == 25, 'flat plane: one exported vertex per grid vertex'
    assert idx.max() == 24
    b = primitives.box(2, 2, 2)
    assert len(b.split_corners()[0]) == 24, 'flat-shaded box: three corners per cube vertex'


def test_extrude_and_delete():
    p = primitives.plane(4, 4, 1)
    before = p.nfaces
    p.extrude_faces([0, 1], 2.0)
    assert p.nfaces == before + 8
    assert p.verts[:, 1].max() == 2.0
    p.delete_faces([0])
    assert p.nfaces == before + 7


def test_subdivide_and_merge():
    p = primitives.plane(4, 4, 1)
    p.subdivide_faces()
    assert p.nfaces == 8
    p.merge_by_distance()
    assert p.nverts == 9


def test_chunks_respect_cap():
    p = primitives.plane(100, 100, 300)
    assert p.export_vertex_count() > MAX_VERTS
    pieces = p.chunks()
    assert len(pieces) > 1
    assert all(c.export_vertex_count() <= MAX_VERTS for c in pieces)
    assert sum(c.nfaces for c in pieces) == p.nfaces


def test_pick():
    b = primitives.box(2, 2, 2)
    t, f = b.pick((0, 1, -10), (0, 0, 1))
    assert f >= 0 and abs(t - 9.0) < 1e-4


def test_terrain_brushes_and_export():
    t = Terrain(17, 17, 2.0)
    t.raise_(16, 16, 8, 3.0)
    assert t.heights.max() > 2.5
    t.smooth(16, 16, 8, 1.0)
    t.paint(4, 4, 6, 1, 1.0)
    assert abs(t.weights[2, 2].sum() - 1.0) < 1e-5
    m = t.mesh()
    assert m.nfaces == 16 * 16 * 2
    assert set(np.unique(m.face_mat)) <= {0, 1}
    chunks = t.export_chunks()
    assert sum(c.nfaces for _m, c in chunks) == m.nfaces
    h = t.height_at(16, 16)
    assert h is not None and h > 0.5
    t2 = Terrain.from_dict(t.to_dict())
    assert np.array_equal(t2.heights, t.heights)

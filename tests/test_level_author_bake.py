"""Vertex, lightmap and probe bakes over a lit room."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import numpy as np
import pytest

from gbtvgr import tex as gtex

from gbtvgr.level import bake
from gbtvgr.level import geometry_primitives as primitives
from gbtvgr.level import textures
from gbtvgr.level.scene import Document, LightNode, MeshNode, ProbeNode, SectionNode

LIGHT = (10.0, 4.0, 10.0)


def _room():
    doc = Document('baketest')
    sec = SectionNode('room')
    sec.set_bounds((-25, -5, -25), (25, 20, 25))
    doc.add_node(sec)
    room = MeshNode('walls')
    room.mesh = primitives.box(40, 12, 40, inside=True)
    room.mesh.subdivide_faces()
    room.mesh.subdivide_faces()
    doc.add_node(room, parent=sec)
    light = LightNode('lamp')
    light.pos = np.array(LIGHT, np.float32)
    light.props.update(color=[255, 180, 100], radius=20.0, gain=1.5)
    doc.add_node(light)
    return doc, sec, room


def _dist(pts):
    return np.linalg.norm(pts - np.array(LIGHT), axis=1)


def test_vertex_bake_lights_near_vertices():
    doc, _sec, room = _room()
    out = bake.bake_vertex_colors(doc, {'ao': False})
    cols = out[room.id]
    assert cols.dtype == np.uint8 and cols.shape == (room.mesh.nverts, 4)
    assert room.mesh.colors is cols
    d = _dist(room.mesh.verts)
    lum = cols[:, :3].astype(float).mean(axis=1)
    near = lum[np.argmin(d)]
    far = lum[np.argmax(d)]
    assert near > far + 40, (near, far)
    assert cols[:, 3].min() == 255


def test_vertex_bake_with_ao_darkens_corners():
    doc, _sec, room = _room()
    plain = bake.bake_vertex_colors(doc, {'ao': False})[room.id].copy()
    with_ao = bake.bake_vertex_colors(doc, {'ao': True, 'ao_rays': 8})[room.id]
    assert with_ao.shape == plain.shape
    d = _dist(room.mesh.verts)
    far = np.argmax(d)
    assert with_ao[far, :3].mean() <= plain[far, :3].mean()


def test_lightmap_plan_and_bake():
    doc, sec, room = _room()
    plan = bake.plan_lightmaps(doc, sec, 256, {'texels_per_unit': 1.0})
    w, h = plan['size']
    assert w == h and 64 <= w <= 256
    lmuv = plan['lmuv'][room.id]
    assert lmuv.shape == (room.mesh.nfaces, 3, 2)
    assert lmuv.min() >= 0.0 and lmuv.max() <= 1.0
    res = bake.bake_lightmaps(doc, sec, plan, {'ao': False})
    assert len(res['tiles']) == 3 and res['tiles'][0] == res['tiles'][2]
    tw, th, img = textures.decode(res['tiles'][0])
    assert (tw, th) == (w, h)
    hdr = gtex.parse_header(res['tiles'][0])
    assert hdr['fmt'] == 3 and hdr['mipcount'] == 1
    centres = room.mesh.face_centers()
    d = _dist(centres)
    near_f, far_f = np.argmin(d), np.argmax(d)

    def texel(f):
        u, v = lmuv[f].mean(axis=0)
        return img[min(h - 1, int(v * h)), min(w - 1, int(u * w)), :3].astype(float).mean()

    assert texel(near_f) > texel(far_f) + 10, (texel(near_f), texel(far_f))


def test_probe_bake_with_raster_fallback():
    from gbtvgr.level.raster import Rasteriser
    doc, _sec, room = _room()
    bake.bake_vertex_colors(doc, {'ao': False})
    r = Rasteriser()
    try:
        pos, nrm, uv, idx, cmap = room.mesh.split_corners()[:5]
        r.set_geometry(pos, room.mesh.colors[cmap], idx)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    probe = ProbeNode('probe')
    probe.pos = np.array((0.0, 5.0, 0.0), np.float32)
    doc.add_node(probe)
    try:
        amb, env = bake.bake_probes([probe], r.cubemap)[0]
    finally:
        r.close()
    for blob, size in ((amb, 16), (env, 64)):
        faces = textures.decode_cube(blob)
        assert len(faces) == 6 and faces[0].shape == (size, size, 4)
        hdr = gtex.parse_header(blob)
        assert hdr['fmt'] == 24 and hdr['w'] == size
    env_faces = textures.decode_cube(env)
    assert any(f[:, :, :3].max() > 0 for f in env_faces), 'the room was rendered'

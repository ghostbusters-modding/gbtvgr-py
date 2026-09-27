"""Authoring a whole set from primitives and reading it back."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import struct

import numpy as np
import pytest

from gbtvgr.mesh.bvt import bvt_bbox, bvt_collect
from gbtvgr.sets import bst
from gbtvgr.level import sections as gsec
from gbtvgr.level import setfile
from gbtvgr.level import geometry_primitives as primitives
from gbtvgr.level.scene import (Document, LightNode, MeshNode, ProbeNode, SectionNode,
                                TerrainNode)
from gbtvgr.level.geometry_terrain import Terrain


@pytest.fixture(scope="module")
def library(game_dir):
    from gbtvgr.level.library import Library
    lib = Library(game_dir)
    lib.open()
    yield lib
    lib.close()


def two_room_doc():
    doc = Document('roomtest')
    for i, (x0, mat) in enumerate(((0.0, 'graveyard\\pavementwet'), (60.0, 'graveyard\\concrete01'))):
        sec = SectionNode('room%d' % i)
        sec.set_bounds((x0 - 30, -5, -30), (x0 + 30, 25, 30))
        doc.add_node(sec, doc.sections)
        room = MeshNode('walls%d' % i)
        room.mesh = primitives.box(50, 20, 50, inside=True)
        room.pos[:] = (x0, 0, 0)
        room.props['materials'] = [mat]
        sec.add(room)
        light = LightNode('lamp%d' % i)
        light.pos[:] = (x0, 15, 0)
        light.props['radius'] = 40.0
        doc.add_node(light, doc.lights)
    t = TerrainNode('ground')
    t.terrain = Terrain(9, 9, 5.0, (-20.0, 0.0, -20.0))
    t.terrain.raise_(0, 0, 10, 1.0)
    t.props['materials'] = ['graveyard\\pavementwet']
    doc.section_list()[0].add(t)
    probe = ProbeNode('probe0')
    probe.pos[:] = (0, 6, 0)
    doc.add_node(probe, doc.probes)
    return doc


def test_sections_meshes_and_collision():
    doc = two_room_doc()
    sec = doc.section_list()[0]
    meshes = gsec.gather_meshes(sec)
    assert {m.material for m in meshes} == {'graveyard\\pavementwet'}
    rec, box = gsec.mesh_record(meshes[0], 0)
    pkt = rec['pkt']
    assert struct.calcsize('<6f') == len(pkt['bbox'])
    assert len(pkt['vdata']) == pkt['nverts'] * 72
    verts, tris, surf, flags, entries = gsec.collect_collision(doc, sec)
    assert len(tris) % 2 == 0 and len(tris) > 0
    n = len(tris) // 2
    assert np.array_equal(tris[:n][:, [0, 2, 1]], tris[n:]), 'both windings, in order'
    root, arena = gsec.build_section_bvt(verts, tris, surf, flags)
    v2, t2, s2, f2 = bvt_collect(root)
    assert len(t2) == len(tris)
    assert arena > 0


def test_compile_set_roundtrip():
    doc = two_room_doc()
    build = setfile.compile_set(doc)
    assert build.errors == [], build.errors
    m = bst.parse(build.data)
    assert bst.build(m) == build.data
    assert len(m['sections']) == 2
    assert len(m['lights']) == 2 and len(m['portals']) == 1
    for s in m['sections']:
        assert s['names3'][8] == 0, 'the no-lightmap sentinel'
        rb = bvt_bbox(s['bvt'])
        sb = struct.unpack('<6f', s['bbox'])
        assert all(rb[k] >= sb[k] for k in range(3)) and all(rb[k] <= sb[k] for k in range(3, 6))
        assert s['bvtflag'] > 0
    assert m['watervis'] == b'\x01\x01'
    assert len(m['bsp']) // 0x30 >= 3
    assert m['nav']['empty']
    assert struct.unpack('<12I', m['lights'][0]['scal'])[2] == 0xFFFFFF


def test_lightmapped_section_declaration():
    doc = two_room_doc()
    sec = doc.section_list()[1]
    meshes = gsec.gather_meshes(sec)
    rec, _ = gsec.mesh_record(meshes[0], 0, lightmapped=True)
    assert len(rec['pkt']['vdata']) == rec['pkt']['nverts'] * 64
    assert struct.unpack('<I', rec['f6c'])[0] == 1


def test_compile_with_materials_checked(library):
    doc = two_room_doc()
    build = setfile.compile_set(doc, library)
    assert build.errors == [], build.errors
    model_only = None
    for n in library.names('materials')[:400]:
        ref = n[len('materials\\'):-len('.mtb')]
        info = library.material_info(ref)
        if info is not None and info.model_ok and not info.set_ok:
            model_only = ref
            break
    assert model_only, 'no model-only material among the first 400'
    doc.section_list()[0].children[0].props['materials'] = [model_only]
    build = setfile.compile_set(doc, library)
    assert any('0x4' in e for e in build.errors), 'a model-only material is refused'

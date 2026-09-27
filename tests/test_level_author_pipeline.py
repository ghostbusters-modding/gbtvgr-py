"""End to end: a document -> set, level, script, bake, navmesh, archive."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import os

import pytest

from gbtvgr.sets import bst
from gbtvgr.level import importer, package, pipeline, setfile
from gbtvgr.level.scene import ActorNode, Document, GroupNode, TriggerNode

from test_level_author_setfile import two_room_doc


@pytest.fixture(scope="module")
def library(game_dir):
    from gbtvgr.level.library import Library
    lib = Library(game_dir)
    lib.open()
    yield lib
    lib.close()


def playable_doc():
    doc = two_room_doc()
    doc.settings['hero_start'] = [0.0, 0.5, 0.0]
    doc.settings['description'] = 'Two rooms and a spawner.'
    sp = ActorNode('spEmit_a')
    sp.props.update(cls='CSpawn', template_level='cemetery2')
    sp.pos[:] = (60, 1, 0)
    doc.add_node(sp)
    biped = ActorNode('pool_fiend1')
    biped.props.update(cls='CBiped', template='biped_GraveFiend1', template_level='cemetery2',
                       created=False)
    biped.fields['charInfoFilename'] = 'Biped1\\\\Fiend_web.cit'
    biped.fields['charInfoVariantName'] = 'standard'
    biped.pos[:] = (0, -60, 0)
    doc.add_node(biped)
    g = GroupNode('g_wave')
    g.members.append('pool_fiend1')
    doc.add_node(g)
    t = TriggerNode('trig_a')
    t.pos[:] = (30, 5, 0)
    t.props['size'] = [10.0, 10.0, 10.0]
    t.props['on_enter'] = [
        {'block': 'spawn_character', 'params': {'spawner': 'spEmit_a', 'cls': 'CBiped',
                                                'cit': 'Biped1\\Fiend_web.cit', 'variant': 'standard',
                                                'group': 'g_wave'}},
        {'block': 'display_message', 'params': {'text': 'hi'}},
    ]
    doc.add_node(t)
    return doc


def test_build_all(library, tmp_path):
    doc = playable_doc()
    opts = pipeline.BuildOptions(out_dir=str(tmp_path), lightmaps=True, install=False)
    res = pipeline.build_all(doc, library, opts)
    assert res.errors == [], (res.errors, res.report)
    assert os.path.isfile(res.pod_path)
    names = {n.lower() for n in package.list_pod(res.pod_path)}
    assert 'sets\\roomtest.bst' in names and 'world\\roomtest1.lvl' in names
    assert 'world\\roomtest1.dante' in names
    assert any(n.startswith('art\\lightprobe\\roomtest\\') for n in names)
    assert any(n.startswith('art\\lightmap\\roomtest\\') for n in names)
    m = bst.parse(res.set_build.data)
    assert not m['nav']['empty'] and m['nav']['nnodes'] > 10
    assert m['sections'][0]['names3'][8] != 0, 'lightmapped section names its tiles'
    node = doc.section_list()[0].children[0]
    assert node.mesh.colors is not None and node.mesh.colors.shape[1] == 4
    assert res.nav is not None and len(res.nav.islands) >= 1


def test_preflight_warns_on_missing_geometry(library):
    doc = Document('empty')
    w = pipeline.preflight(doc, library)
    assert any('no sections' in x for x in w)


def test_import_set_and_recompile(library):
    doc = Document('abyss_edit')
    s = library.read_set_data('abyss')
    added = importer.import_set(doc, s, sections=[1])
    assert len(added) == 1 and len(added[0].children) > 1000
    assert doc.skybox.props['donor'] == 'abyss'
    build = setfile.compile_set(doc, library)
    assert build.errors == [], build.errors
    m = bst.parse(build.data)
    assert len(m['sections']) == 1 and len(m['sky']['layers']) == len(s.raw['sky']['layers'])
    assert m['sections'][0]['bvtflag'] > 0


def test_import_level_keeps_actors(library):
    from gbtvgr.level import levelfile
    doc = Document('cem')
    added = importer.import_level(doc, library.read_level('cemetery2'), 'cemetery2')
    assert len(added) > 1500
    assert doc.settings['keep_script'] and doc.settings['set_name'] == 'cemetery2'
    assert doc.nodes('group')
    lb = levelfile.compile_level(doc, library)
    assert lb.errors == [], lb.errors[:5]
    text = lb.data.decode('latin1')
    assert 'trig_BellRing_onActorEnter' in text, 'shipped event handlers survive'

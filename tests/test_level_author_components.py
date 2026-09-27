"""Component resolve and export-time dedup."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import os
import types

import pytest

from gbtvgr.mesh import smb
from gbtvgr.level import components as comp
from gbtvgr.level import meshes as level_meshes
from gbtvgr.level.scene import ComponentNode, Document


@pytest.fixture(scope="module")
def library(game_dir):
    from gbtvgr.level.library import Library
    lib = Library(game_dir)
    lib.open()
    # gbtvgr.level.library.Library has no read_model (that lived in the editor's own
    # GameLibrary); component_from_model needs it, so it is stubbed here per B2_COMMON.md.
    lib.read_model = types.MethodType(_read_model, lib)
    yield lib
    lib.close()


def _read_model(self, ref):
    blob = self.read(self.model_ref_path(ref))
    return level_meshes.load_model(blob, ref) if blob else None


def tree_doc():
    doc = Document('trees')
    cd = comp.ComponentDef('tree', 'graveyard\\tree01', ['graveyard\\bark', 'graveyard\\leaves'])
    doc.components[cd.id] = cd
    for i in range(50):
        n = ComponentNode('tree_%d' % i)
        n.props['component'] = cd.id
        n.pos[:] = (i * 5.0, 0.0, 0.0)
        if i % 10 == 0:
            n.props['slots'] = {'1': 'graveyard\\leaves_autumn'}
        if i == 7:
            n.props['textures'] = {'1': {'0': 'graveyard\\leaves_dead_diff.tga'}}
        doc.add_node(n)
    return doc, cd


def test_dedup_counts():
    doc, cd = tree_doc()
    plan = comp.resolve_all(doc)
    assert len(plan.instances) == 50
    assert len(plan.variants) == 3, 'base, recoloured, retextured'
    base = cd.resolve({})
    assert plan.model_ref[base.key] == 'graveyard\\tree01', 'the untouched base ships as the shipped model'
    assert len(plan.material_clones) == 1
    clone = next(iter(plan.material_clones))
    assert clone.startswith('graveyard\\leaves_')


def test_variant_keys_stable():
    doc, cd = tree_doc()
    a = cd.resolve({'slots': {'1': 'x'}})
    b = cd.resolve({'slots': {1: 'x'}})
    assert a.key == b.key
    assert cd.resolve({}).key != a.key


def test_write_plan_against_archives(library, tmp_path):
    first = library.names('meshes')[0]
    ref = first[len('models\\'):-len('.smb')]
    cd = comp.component_from_model(library, ref)
    assert cd.slots
    doc = Document('w')
    doc.components[cd.id] = cd
    base_ref = cd.slots[0] or 'cherub'
    info = library.material_info(base_ref)
    layer = info.layer_of_type(0) if info else 0
    for i in range(3):
        n = ComponentNode('inst_%d' % i)
        n.props['component'] = cd.id
        n.props['textures'] = {'0': {str(layer or 0): 'graveyard\\wet_diff.tga'}}
        doc.add_node(n)
    plan = comp.resolve_all(doc)
    assert len(plan.variants) == 1
    written = comp.write_plan(plan, library, str(tmp_path))
    smbs = [p for p in written.values() if p.endswith('.smb')]
    mtbs = [p for p in written.values() if p.endswith('.mtb')]
    assert len(smbs) == 1 and len(mtbs) == 1
    m = smb.parse(open(smbs[0], 'rb').read())
    refs = [e['ref'].decode('latin1') for e in m['materials']]
    assert any(os.path.basename(mtbs[0])[:-4] in r for r in refs)

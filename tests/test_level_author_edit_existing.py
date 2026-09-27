"""The edit-an-existing-level path: import a shipped set and level, keep the
shipped script, and rebuild the archive."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import pytest

from gbtvgr.sets import bst
from gbtvgr.level import importer, package, pipeline
from gbtvgr.level.scene import Document


@pytest.fixture(scope="module")
def library(game_dir):
    from gbtvgr.level.library import Library
    lib = Library(game_dir)
    lib.open()
    yield lib
    lib.close()


def test_import_abyss_and_rebuild(library, tmp_path):
    doc = Document('abyss')
    s = library.read_set_data('abyss')
    importer.import_set(doc, s, sections=[0, 1])
    importer.import_level(doc, library.read_level('abyss'), 'abyss')
    assert doc.settings['keep_script'] and doc.settings['set_name'].lower() == 'abyss'
    opts = pipeline.BuildOptions(out_dir=str(tmp_path), probes=False, navmesh=False, bake_vertex=False)
    res = pipeline.build_all(doc, library, opts)
    assert res.errors == [], (res.errors, res.report[-5:])
    names = {n.lower() for n in package.list_pod(res.pod_path)}
    assert 'sets\\abyss.bst' in names and 'world\\abyss.lvl' in names, names
    assert 'world\\abyss.dante' not in names, 'the shipped script stays in COMMON.POD'
    m = bst.parse(res.set_build.data)
    assert len(m['sections']) == 2
    assert sum(len(sec['meshes']) for sec in m['sections']) > 0
    assert len(m['lights']) == len(s.lights) and len(m['portals']) == len(s.probes)
    assert len(m['sky']['layers']) == len(s.raw['sky']['layers'])

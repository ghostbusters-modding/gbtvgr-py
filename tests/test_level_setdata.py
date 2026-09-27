"""SetData over Library.read_set_data: numpy views whose counts match bst.parse."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import pytest

from gbtvgr.sets import bst
from gbtvgr.level import setdata


@pytest.fixture(scope="module")
def library(game_dir):
    from gbtvgr.level.library import Library
    lib = Library(game_dir)
    lib.open()
    yield lib
    lib.close()


@pytest.mark.parametrize('stem', ['abyss', 'hotel1a'])
def test_read_set_data_counts_match_parse(library, stem):
    sd = library.read_set_data(stem)
    raw = bst.parse(library.read('sets\\%s.bst' % stem))
    assert len(sd.sections) == len(raw['sections'])
    assert len(sd.lights) == len(raw['lights'])
    assert len(sd.probes) == len(raw['portals'])


def test_abyss_meshes_and_collision_load(library):
    sd = library.read_set_data('abyss')
    sec = sd.sections[1]
    assert sec.mesh_count > 1000
    assert sec.meshes[0].positions is not None
    verts, tris, surf, flags = sec.collision
    assert len(tris) > 0 and surf.max() < len(sec.surfaces)


def test_abyss_rebuild_roundtrip(library):
    """The one set the editor's own test proved byte-identical after rebuild."""
    sd = library.read_set_data('abyss')
    assert setdata.rebuild(sd) == library.read('sets\\abyss.bst')

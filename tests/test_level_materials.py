"""MaterialInfo.diffuse: name decides between same-typed diff/bump/spec layers."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
from gbtvgr.level.materials import MaterialInfo


def test_diffuse_prefers_a_name_ending_in_diff():
    m = MaterialInfo('t', 0, [(0, 'hotel\\fountain_bump.tga'), (0, 'hotel\\fountain_diff.tga')])
    assert m.diffuse == 'hotel\\fountain_diff.tga'


def test_diffuse_is_case_insensitive():
    m = MaterialInfo('t', 0, [(0, 'x_BUMP.tga'), (0, 'x_DIFF.tga')])
    assert m.diffuse == 'x_DIFF.tga'


def test_diffuse_skips_known_map_suffixes_when_no_diff_present():
    m = MaterialInfo('t', 0, [(0, 'x_spec.tga'), (0, 'x_color.tga'), (0, 'x_norm.tga')])
    assert m.diffuse == 'x_color.tga'


def test_diffuse_falls_back_to_the_first_layer_when_all_are_map_suffixes():
    m = MaterialInfo('t', 0, [(0, 'x_bump.tga'), (0, 'x_spec.tga')])
    assert m.diffuse == 'x_bump.tga'


def test_diffuse_none_without_layers():
    assert MaterialInfo('t', 0, []).diffuse is None

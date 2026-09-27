"""The multi-section collision BSP, against authored boxes and the shipped trees."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import os
import struct

import numpy as np
import pytest

from gbtvgr.sets import bst
from gbtvgr.level import bsp


def test_single_section_is_one_leaf():
    nodes_b, glob_b, nodes = bsp.build_bsp([((0, 0, 0), (10, 10, 10))])
    assert len(nodes) == 1 and nodes[0][1] == 0
    assert bsp.query(bsp.parse_bsp(nodes_b), (5000, -5000, 12)) == 0


def test_grid_of_rooms_resolves():
    boxes = []
    for i in range(2):
        for j in range(2):
            boxes.append(((i * 40.0, 0.0, j * 40.0), (i * 40.0 + 40.0, 20.0, j * 40.0 + 40.0)))
    boxes.append(((10.0, 20.0, 10.0), (70.0, 40.0, 70.0)))          # a floor above
    nodes_b, glob_b, nodes = bsp.build_bsp(boxes)
    parsed = bsp.parse_bsp(nodes_b)
    assert len(parsed) == len(nodes)
    tested, problems = bsp.verify(parsed, boxes, samples_per_section=200)
    assert tested == 1000 and problems == [], problems[:5]
    assert bsp.query(parsed, (5, 5, 5)) == 0
    assert bsp.query(parsed, (75, 5, 75)) == 3
    assert bsp.query(parsed, (40, 30, 40)) == 4
    # far outside still lands somewhere
    assert bsp.query(parsed, (9999, 9999, -9999)) >= 0
    # child boxes nest inside their parents, as the shipped trees do
    for plane, sec, front, back, lo, hi in parsed:
        for c in (front, back):
            if c >= 0:
                assert np.all(parsed[c][4] >= lo - 1e-3) and np.all(parsed[c][5] <= hi + 1e-3)


def test_overlapping_boxes_still_map_inside():
    boxes = [((0, 0, 0), (50, 20, 50)), ((20, 0, 20), (30, 10, 30))]
    _b, _g, nodes = bsp.build_bsp(boxes)
    tested, problems = bsp.verify(nodes, boxes)
    assert problems == []


@pytest.mark.parametrize('stem', ['abyss', 'boss_sp_side', 'hotel2', 'timessquare1'])
def test_convention_against_corpus(bst_corpus, stem):
    """The shipped trees, walked with our plane-side rule, resolve section centres."""
    path = os.path.join(bst_corpus, stem + '.bst')
    if not os.path.isfile(path):
        pytest.skip(path)
    m = bst.parse(open(path, 'rb').read())
    nodes = bsp.parse_bsp(m['bsp'])
    hits = 0
    for si, s in enumerate(m['sections']):
        sb = struct.unpack('<6f', s['bbox'])
        c = [(sb[k] + sb[k + 3]) / 2.0 for k in range(3)]
        if bsp.query(nodes, c) == si:
            hits += 1
    assert hits >= 0.85 * len(m['sections'])

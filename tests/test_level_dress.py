"""Prop clusters: seeded, lane-keeping, never overlapping (no corpus)."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import math

import pytest

from gbtvgr.level import dress, kit
from gbtvgr.level.placement import rot_xz


def test_footprint_rotates_with_yaw():
    w = dress.footprint('industrial\\woodcrate04', 0)
    d = dress.footprint('industrial\\woodcrate04', 90)
    assert w[2] - w[0] == pytest.approx(3.06) and w[3] - w[1] == pytest.approx(6.68)
    assert d[2] - d[0] == pytest.approx(6.68) and d[3] - d[1] == pytest.approx(3.06)
    with pytest.raises(KeyError):
        dress.footprint('museum\\no_such_model', 0)
    sizes = {'x\\thing': ((-1, 0, -2), (1, 4, 2), 1)}
    assert dress.footprint('x\\thing', 0, sizes) == (-1, -2, 1, 2)


def test_row_grounds_or_hangs():
    r = dress.row('museum\\metalcrate06', 5.0, 6.0, 0.0, y=2.0)
    assert r['pos'] == (5.0, 2.15, 6.0), 'main box floor lands on y'
    assert r['aabb'][1] == pytest.approx(2.0) and r['solid']
    lamp = dress.row('museum\\swinglight', 0.0, 0.0, 0.0, y=12.0, hang=True)
    assert lamp['pos'][1] == 12.0 and lamp['aabb'][1] == pytest.approx(6.35)
    assert not lamp['solid']
    assert not dress.row('lost_island\\junk8', 0, 0, 0)['solid'], 'too low to collide'
    assert dress.row('industrial\\pallet_wood', 0, 0, 0, solid=True)['solid']


def test_crate_island_deterministic_and_clear():
    a = dress.crate_island((30.0, 30.0), (30.0, 24.0), seed=5)
    b = dress.crate_island((30.0, 30.0), (30.0, 24.0), seed=5)
    assert a == b and len(a) >= 8
    assert dress.crate_island((30.0, 30.0), (30.0, 24.0), seed=6) != a
    for seed in range(6):
        rows = dress.crate_island((30.0, 30.0), (30.0, 24.0), seed=seed)
        assert dress.overlaps(rows) == []
        for r in rows:
            x0, _y0, z0, x1, _y1, z1 = r['aabb']
            assert x0 >= 18.0 - 1e-6 and x1 <= 42.0 + 1e-6, 'the 3 ft lane'
            assert z0 >= 21.0 - 1e-6 and z1 <= 39.0 + 1e-6
    rows = dress.crate_island((0.0, 0.0), (40.0, 30.0), seed=2)
    stacked = [r for r in rows if r['pos'][1] > 0.5]
    assert stacked, 'some crates ride on others'
    for top in stacked:
        base = [r for r in rows if r is not top and abs(r['aabb'][4] - top['aabb'][1]) < 1e-6
                and r['aabb'][0] <= top['aabb'][0] and r['aabb'][3] >= top['aabb'][3]
                and r['aabb'][2] <= top['aabb'][2] and r['aabb'][5] >= top['aabb'][5]]
        assert base, 'a stacked crate sits inside its base footprint'


def test_pallet_stack_and_barrel_row():
    rows = dress.pallet_stack((4.0, 4.0), n=5, seed=1)
    assert len(rows) == 5 and all(r['solid'] for r in rows)
    ys = [r['aabb'][1] for r in rows]
    assert ys == pytest.approx([0.43 * k for k in range(5)])
    assert all(abs(r['yaw'] - 360 if r['yaw'] > 180 else r['yaw']) <= 4 for r in rows)
    rows = dress.barrel_row((10.0, 5.0), 'x', 5, seed=3)
    assert len(rows) == 5 and dress.overlaps(rows) == []
    xs = [r['aabb'] for r in rows]
    assert xs[0][0] == pytest.approx(10.0)
    for a, b in zip(xs, xs[1:]):
        assert b[0] - a[3] == pytest.approx(dress.GAP)
    assert rows == dress.barrel_row((10.0, 5.0), 'x', 5, seed=3)
    rows = dress.barrel_row((0.0, 0.0), 'z', 3, seed=3)
    assert rows[0]['aabb'][2] == pytest.approx(0.0)
    assert rows[-1]['aabb'][5] > rows[0]['aabb'][5]


def test_spools_and_junk():
    rows = dress.cable_spools((10.0, 10.0), (20.0, 14.0), count=6, seed=1)
    assert 2 <= len(rows) <= 6 and dress.overlaps(rows) == []
    assert all(r['model'] == dress.SPOOL and r['pos'][1] == pytest.approx(3.45)
               for r in rows), 'a drum stands on its rim'
    assert rows == dress.cable_spools((10.0, 10.0), (20.0, 14.0), count=6, seed=1)
    pile = dress.net_pile((5.0, 5.0), count=8, seed=2, radius=4.0)
    assert len(pile) == 8 and dress.overlaps(pile) == []
    assert not any(r['solid'] for r in pile)
    for r in pile:
        assert all(abs(v - 5.0) <= 4.0 + 1e-6 for v in (r['aabb'][0], r['aabb'][3],
                                                         r['aabb'][2], r['aabb'][5]))
    assert pile == dress.net_pile((5.0, 5.0), count=8, seed=2, radius=4.0)


def test_lines():
    rows = dress.sawhorse_cordon(((0.0, 0.0), (24.0, 0.0)), pitch=8.0)
    horses = [r for r in rows if r['model'] == dress.SAWHORSE]
    cones = [r for r in rows if r['model'] == dress.CONE]
    assert len(horses) == 4 and len(cones) == 3
    for h in horses:
        lx, lz = rot_xz(0.0, 1.0, h['yaw'])
        assert abs(lx) > 0.99, 'the long axis lies along the line'
    assert [c['pos'][0] for c in cones] == pytest.approx([4.0, 12.0, 20.0])
    assert dress.overlaps(rows) == []
    rows = dress.sawhorse_cordon(((5.0, 0.0), (5.0, 16.0)), pitch=8.0)
    assert all(abs(rot_xz(0.0, 1.0, r['yaw'])[1]) > 0.99 for r in rows
               if r['model'] == dress.SAWHORSE)
    lamps = dress.lamp_post_row(((0.0, 0.0), (48.0, 0.0)), pitch=24.0, height=12.0)
    assert [l['pos'] for l in lamps] == [(0.0, 12.0, 0.0), (24.0, 12.0, 0.0),
                                         (48.0, 12.0, 0.0)]
    assert lamps[0]['aabb'][1] == pytest.approx(6.35) and not lamps[0]['solid']
    bol = dress.bollard_row(((0.0, 0.0), (0.0, 20.0)), pitch=6.0)
    assert [b['pos'][2] for b in bol] == pytest.approx([0.0, 5.0, 10.0, 15.0, 20.0])
    assert all(b['solid'] for b in bol)


def test_tyre_fenders_hang_outside():
    rows = dress.tyre_fenders(((0.0, 0.0), (0.0, 24.0)), pitch=6.0, drop=1.5)
    assert len(rows) == 5 and not any(r['solid'] for r in rows)
    for r in rows:
        x0, y0, z0, x1, y1, z1 = r['aabb']
        assert x1 <= 0.0 and x1 > -0.2, 'hung just off the quay face'
        assert x1 - x0 == pytest.approx(0.84) and z1 - z0 == pytest.approx(2.48)
        assert (y0 + y1) / 2 == pytest.approx(-1.5)
    other = dress.tyre_fenders(((0.0, 0.0), (0.0, 24.0)), side=-1)
    assert all(r['aabb'][0] >= 0.0 for r in other)
    along_x = dress.tyre_fenders(((0.0, 0.0), (24.0, 0.0)))
    assert all(r['aabb'][5] - r['aabb'][2] == pytest.approx(0.84) for r in along_x)


def test_nav_boxes_feed_the_kit():
    rows = dress.crate_island((30.0, 30.0), (30.0, 24.0), seed=1)
    boxes = dress.nav_boxes(rows)
    assert boxes and all(len(b) == 6 for b in boxes)
    assert len(boxes) == sum(1 for r in rows if r['solid'])
    ex = kit.nav_exclusions(boxes, floor_y=0.0)
    assert ex and all(x0 >= 12.0 and x1 <= 48.0 for x0, _z0, x1, _z1 in ex)
    assert dress.nav_boxes(dress.net_pile((0.0, 0.0))) == []

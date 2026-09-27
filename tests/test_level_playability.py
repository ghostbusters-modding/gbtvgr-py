"""Playability gates: spawns in bounds, paths walkable, unlinked voids found."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import copy
import os
import struct

import pytest

from gbtvgr import lvl as _lvl
from gbtvgr.level import playability as pl
from gbtvgr.level import transplant as tp
from gbtvgr.level import wire

TRANSPLANT_DEMO = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'mods', 'TransplantDemo'))

HERO = (77.0, -29.818344, -106.0)          # Ghostbuster0's pos, graft2.lvl
TRIG_RETURN = (53.0, -15.724656, -98.0)    # trig_return's pos, graft2.lvl


def _walkable(nav, i):
    """Independent of playability.py's own _node_walkable: the same RE_SECTION_LINK.md
    4.4 rule, reimplemented so the test does not just call back into the code under test."""
    e = struct.unpack('<I', nav['nodes'][i]['f15c'])[0]
    return struct.unpack('<I', nav['extra'][e]['b'])[0] & 5 != 0


@pytest.fixture(scope='module')
def graft2():
    if not os.path.isdir(TRANSPLANT_DEMO):
        pytest.skip('needs mods/TransplantDemo (graft2)')
    src = wire.DirSource(root=TRANSPLANT_DEMO)
    return wire.open_level(src, 'graft2', load_set=True)


ELEVATED_Y = '-15.635681'     # 5 ft over Room26's floor, PROOF2's original rows


@pytest.fixture(scope='module')
def graft2_elevated(graft2):
    """The installed graft2 has its spawns fixed; the gate tests want the defect back."""
    lv = graft2
    for i in (1, 2, 3):
        x, _y, z = _actor_pos(lv, 'spEmit_tunnel%d' % i)
        lv = _with_actor_field(lv, 'spEmit_tunnel%d' % i, 'pos', '%s, %s, %s' % (x, ELEVATED_Y, z))
    return lv


def _actor_pos(level, tag):
    ab = _lvl.top_block(level.lvl, 'actors')
    for a in ab.children:
        if isinstance(a, _lvl.Block) and a.tag == tag:
            for c in a.children:
                if isinstance(c, _lvl.KV) and c.key == 'pos':
                    return [v.strip() for v in c.value.split(',')]
    raise AssertionError('actor %r not found' % tag)


@pytest.fixture(scope='module')
def cemetery1_level(lvl_corpus, bst_corpus):
    src = wire.DirSource(root=lvl_corpus, sets=bst_corpus)
    return wire.open_level(src, 'cemetery1', load_set=True)


def _with_actor_field(level, tag, key, value):
    """A deep copy of level with one field of one .lvl actor rewritten in place."""
    new_lvl = copy.deepcopy(level.lvl)
    ab = _lvl.top_block(new_lvl, 'actors')
    for a in ab.children:
        if isinstance(a, _lvl.Block) and a.tag == tag:
            for c in a.children:
                if isinstance(c, _lvl.KV) and c.key == key:
                    c.value = value
                    break
            break
    else:
        raise AssertionError('actor %r not found' % tag)
    return wire.Level(level.stem, new_lvl, level.secs, level.set_name, level.bst,
                      level.script, level.lang)


# ============================================================================
# geometry primitives
# ============================================================================

def test_tri_aabb_overlap_and_miss():
    tri = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
    assert pl._tri_aabb(tri, (0.3, 0.3, 0, 5, 5, 5)) is True
    assert pl._tri_aabb(tri, (100, 100, 100, 1, 1, 1)) is False
    assert pl._tri_aabb(tri, (2, 0, 0, 1, 1, 1)) is True   # touches the (1,0,0) vertex


def test_ray_tri_up_only_hits_downward_faces():
    """catalogue.py's ny convention: a ceiling faces down (ny<0), a floor bump up."""
    a, b, c = (0, 5, 0), (1, 5, 0), (1, 5, 1)
    ceiling, floor = (a, b, c), (a, c, b)   # opposite winding, opposite normal
    assert abs(pl._ray_tri_up((0.5, 0, 0.2), *ceiling) - 5.0) < 1e-6
    assert pl._ray_tri_up((0.5, 0, 0.2), *floor) is None


def test_in_poly_xz():
    ring = [(0, 0, 0), (2, 0, 0), (2, 0, 2), (0, 0, 2)]
    assert pl._in_poly_xz((1, 0, 1), ring) is True
    assert pl._in_poly_xz((5, 0, 5), ring) is False


def test_flies_by_own_class_or_a_spawn_row_s_enemy_class():
    assert pl._flies({'cls': 'CFlyerSmall'}) is True
    assert pl._flies({'cls': 'CFloater'}) is False
    assert pl._flies({'cls': 'CSpawn', 'enemy_class': 'CFlyerMedium'}) is True
    assert pl._flies({'cls': 'CSpawn', 'enemy_class': 'CFloater'}) is False


# ============================================================================
# spawn_check
# ============================================================================

def test_spawn_check_graft2_names_the_elevated_rows(graft2_elevated):
    """spEmit_tunnel1..3 raised 5 ft over Room26's floor: the gate names exactly
    those three, not the pool rows or triggers."""
    graft2 = graft2_elevated
    bad = pl.spawn_check(graft2, graft2.bst)
    names = sorted(b.split()[1] for b in bad)
    assert names == ['spEmit_tunnel1', 'spEmit_tunnel2', 'spEmit_tunnel3']
    assert all('above a nav node polygon' in b for b in bad)


def test_spawn_check_fix_moves_the_row_into_tolerance(graft2_elevated):
    fixed = _with_actor_field(graft2_elevated, 'spEmit_tunnel1', 'pos', '50.5, -20.6, -57')
    bad = pl.spawn_check(fixed, fixed.bst)
    assert not any('spEmit_tunnel1' in b for b in bad)
    assert any('spEmit_tunnel2' in b for b in bad)   # its untouched siblings still flag


def test_spawn_check_reports_a_spawn_5ft_outside_the_tunnel(graft2):
    # Room26's bbox tops out at x 59.0; +5 puts spEmit_tunnel1 through the east wall.
    moved = _with_actor_field(graft2, 'spEmit_tunnel1', 'pos', '69.0, -15.635681, -57')
    bad = pl.spawn_check(moved, moved.bst)
    assert any('spEmit_tunnel1' in b and 'bbox does not hold' in b for b in bad)


def test_spawn_check_cemetery1_no_geometric_out_of_bounds(cemetery1_level):
    """39 shipped rows fail the nav-height check (elevated spawn effects, normal
    retail design, e.g. floater ambush rigs). The literal 'out of bounds' case --
    the BSP section not holding the point at all -- is 3 large CTrigger volumes
    whose pos is not their true footprint's centre (a gauntlet/container trigger
    spanning a vertical shaft): explained, not a spawn_check bug."""
    bad = pl.spawn_check(cemetery1_level, cemetery1_level.bst)
    oob = [b for b in bad if 'bbox does not hold' in b or 'BSP walk failed' in b
           or 'has no section' in b]
    names = sorted(b.split()[1] for b in oob)
    assert names == ['trig_Area1PlayerPastFirstDoor', 'trig_containerGauntlet',
                     'trig_undergroundContainer1']


# ============================================================================
# path_check / route_check
# ============================================================================

def test_route_check_hero_to_trig_return_passes(graft2):
    assert pl.route_check(graft2, graft2.bst, [HERO, TRIG_RETURN]) == []


def test_path_check_reports_the_first_failure(graft2):
    """A point far outside the set (no nav within reach along the way) fails
    loudly rather than silently returning a bogus short path."""
    bad = pl.path_check(graft2.bst, HERO, (5000.0, 0.0, 5000.0))
    assert bad and isinstance(bad[0], str)


def test_route_check_cemetery1_same_room_passes(cemetery1_level):
    m = cemetery1_level.bst
    nav = m['nav']
    for i, d in enumerate(nav['nodes']):
        if not _walkable(nav, i):
            continue
        gi = tp.nav_header(d)['group']
        for j in tp.nav_neighbours(d):
            if j >= 0 and _walkable(nav, j) and tp.nav_header(nav['nodes'][j])['group'] == gi:
                start = tp.nav_header(d)['centroid']
                goal = tp.nav_header(nav['nodes'][j])['centroid']
                assert pl.path_check(m, start, goal) == []
                return
    pytest.fail('no walkable same-room nav pair found in cemetery1')


# ============================================================================
# void_openings (exploratory: no corpus-clean claim, see the report)
# ============================================================================

def test_void_openings_graft2_finds_the_documented_room21_gap(graft2):
    """Room25's real opening to Room21 (35.5 x 15.2 ft, LINK's census) carries no
    doorway record in graft2 (Room21 is not in this set): an uncovered void."""
    out = pl.void_openings(graft2.bst)
    room21_like = [o for o in out if o['section'] == 0
                   and o['hi'][0] - o['lo'][0] > 20 and o['hi'][1] - o['lo'][1] > 10]
    assert room21_like, out


# ============================================================================
# CLI
# ============================================================================

def test_cli_spawns(graft2, graft2_elevated, tmp_path, capsys):
    rc = pl.main(['spawns', TRANSPLANT_DEMO, 'graft2'])
    assert rc == 0, 'the installed graft2 has every spawn on nav'
    wire.write_level(graft2_elevated, str(tmp_path))
    rc = pl.main(['spawns', str(tmp_path), 'graft2'])
    assert rc == 1
    assert 'spEmit_tunnel1' in capsys.readouterr().out


def test_cli_route(graft2, capsys):
    rc = pl.main(['route', TRANSPLANT_DEMO, 'graft2',
                  '77,-29.818344,-106', '53,-15.724656,-98'])
    assert rc == 0
    assert '0 problems' in capsys.readouterr().out


def test_cli_void(graft2, capsys):
    rc = pl.main(['void', os.path.join(TRANSPLANT_DEMO, 'sets', 'graft2.bst')])
    assert rc == 1
    assert 'void openings' in capsys.readouterr().out

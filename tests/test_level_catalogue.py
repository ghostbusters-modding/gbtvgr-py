"""The content catalogue: section stats, actor "beats" and audio strings,
cross-checked against the shipped corpus and the level_census reference CSVs."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import csv
import os
import time

import pytest

from gbtvgr.level import catalogue as cat
from gbtvgr.sets import bst

CENSUS_DIR = '/mnt/x/ghostbusters-modding/gbtvgr-docs/formats/level_census'
CENSUS_LEVELS = ('hotel1a', 'cemetery1', 'library1b', 'boss_sp_side')


@pytest.fixture(scope='session')
def sections_all(bst_corpus):
    return cat.build_sections(bst_corpus)


@pytest.fixture(scope='session')
def by_set(sections_all):
    out = {}
    for r in sections_all:
        out.setdefault(r['set'], []).append(r)
    return out


@pytest.fixture(scope='session')
def beats_two(lvl_corpus, bst_corpus, tmp_path_factory):
    """Only the two levels the "Tests" spec names, to keep this file fast."""
    out_dir = tmp_path_factory.mktemp('cat_dn')
    precomputed = cat._default_dn_precomputed(os.environ.get('GB_BUILD'))
    return cat.build_beats(lvl_corpus, bst_corpus, str(out_dir), precomputed,
                            levels=['hotel1a', 'cemetery1'])


# ============================================================================
# 1. section counts match bst.parse, for every shipped set
# ============================================================================

def test_section_counts_match_bst_parse(bst_corpus, by_set):
    assert len(by_set) == 20, sorted(by_set)
    for set_name, rows in by_set.items():
        with open(os.path.join(bst_corpus, set_name + '.bst'), 'rb') as fh:
            m = bst.parse(fh.read())
        assert len(rows) == len(m['sections']), set_name
        assert [r['index'] for r in rows] == list(range(len(rows))), set_name


# ============================================================================
# 2. the four census levels' numbers match level_census/*.csv
# ============================================================================

@pytest.mark.parametrize('level', CENSUS_LEVELS)
def test_matches_level_census_csv(level, by_set):
    path = os.path.join(CENSUS_DIR, '%s_sections.csv' % level)
    if not os.path.isfile(path):
        pytest.skip('needs %s' % path)
    with open(path, newline='') as fh:
        census_rows = list(csv.DictReader(fh))
    rows = by_set[level]
    assert len(rows) == len(census_rows)
    for r, cr in zip(rows, census_rows):
        assert r['name'] == cr['name'], (level, r['index'])
        assert r['mesh_count'] == int(cr['mesh_count']), (level, r['index'])
        assert r['triangle_count'] == int(cr['triangle_count']), (level, r['index'])
        assert r['has_bvt'] == (cr['has_bvt'] == 'yes'), (level, r['index'])
        assert r['probe_count'] == int(cr['probe_count']), (level, r['index'])
        assert r['light_count'] == int(cr['light_count']), (level, r['index'])


# ============================================================================
# 3. openings: doorway record count x2 (A and B), cemetery1
# ============================================================================

def test_cemetery1_openings_count(bst_corpus):
    with open(os.path.join(bst_corpus, 'cemetery1.bst'), 'rb') as fh:
        m = bst.parse(fh.read())
    doorway_count = len(m['fx']) // cat.FX_REC_SIZE
    assert doorway_count > 0
    total = sum(len(cat.openings_for_section(m, i)) for i in range(len(m['sections'])))
    assert total == doorway_count * 2


def test_doorway_records_name_valid_sections(bst_corpus):
    """Every corpus doorway's A/B stay inside that set's own section table."""
    for set_name in sorted(f[:-4] for f in os.listdir(bst_corpus) if f.endswith('.bst')):
        with open(os.path.join(bst_corpus, set_name + '.bst'), 'rb') as fh:
            m = bst.parse(fh.read())
        ns = len(m['sections'])
        for rec in cat._fx_records(m):
            assert 0 <= rec['a'] < ns and 0 <= rec['b'] < ns, (set_name, rec['index'])


# ============================================================================
# 4. at least one CCameraPathActor per hotel1a/cemetery1 with a named driver
# ============================================================================

@pytest.mark.parametrize('level', ('hotel1a', 'cemetery1'))
def test_camera_path_actor_has_named_driver(level, beats_two):
    lv = next(b for b in beats_two if b['level'] == level)
    cps = lv['camera_paths']
    assert cps, level
    named = [r for r in cps if r['drivers']]
    assert named, '%s: no CCameraPathActor with a driver function' % level


def test_camera_path_key_count_and_duration(beats_two):
    lv = next(b for b in beats_two if b['level'] == 'hotel1a')
    row = next(r for r in lv['camera_paths'] if r['name'] == 'cpa_LobbyExt_car1')
    assert row['key_count'] == 3
    assert row['duration'] == pytest.approx(246.75)


def test_beat_actors_carry_a_resolved_section(beats_two):
    """Most placed actors' pos falls inside a real BSP leaf section."""
    lv = next(b for b in beats_two if b['level'] == 'cemetery1')
    triggers = lv['triggers']
    assert triggers
    resolved = [r for r in triggers if r['section'] is not None]
    assert len(resolved) > len(triggers) * 0.5, 'too few triggers resolved a section'


def test_trigger_handlers_captured(beats_two):
    lv = next(b for b in beats_two if b['level'] == 'hotel1a')
    with_handlers = [r for r in lv['triggers'] if r['handlers']]
    assert with_handlers


def test_cinematics_present(beats_two):
    lv = next(b for b in beats_two if b['level'] == 'cemetery1')
    assert lv['cinemat_list'] or lv['cinemats']


# ============================================================================
# 5. find --style street returns hotel1a's street-facing sections
# ============================================================================

def test_find_hotel1a_street_sections(by_set):
    home_c = cat._vocab_counts([_ref for r in by_set['hotel1a'] for _ref in r['materials']])
    rows = by_set['hotel1a']
    hits = [r for r in rows if 'street' in [t.strip() for t in r['style'].split(',')]]
    names = {r['name'] for r in hits}
    assert names, 'no hotel1a section tagged style=street'
    assert names <= {'room_exterior1', 'room_exterior2', 'room_exterior4',
                     'room_exterior5', 'room_exterior6', 'room_loungeExterior3'}
    assert 'room_exterior1' in names


def test_style_tag_falls_back_to_home_style(by_set):
    """An ordinary hotel1a room with no distinctive material carries the home tag."""
    room01 = next(r for r in by_set['hotel1a'] if r['name'] == 'Room01')
    assert room01['style'] == 'hotel'


# ============================================================================
# interior/exterior guess
# ============================================================================

def test_interior_exterior_guess_matches_known_rooms(by_set):
    by_name = {r['name']: r for r in by_set['hotel1a']}
    assert by_name['room_exterior4']['interior'] == 'exterior'
    assert by_name['Room06']['interior'] == 'interior'


# ============================================================================
# the CLI entry point: main(argv=None), no dependency on cli.py
# ============================================================================

def test_cli_find_via_main(bst_corpus, tmp_path, capsys):
    out = tmp_path / 'catalogue'
    rc = cat.main(['find', '--bst-corpus', bst_corpus, '--out', str(out),
                  '--style', 'street'])
    assert rc == 0
    printed = capsys.readouterr().out
    assert 'hotel1a' in printed
    assert (out / 'sections.json').is_file()


def test_cli_build_under_budget(bst_corpus, lvl_corpus, tmp_path):
    out = tmp_path / 'catalogue_full'
    start = time.time()
    rc = cat.main(['build', '--bst-corpus', bst_corpus, '--lvl-corpus', lvl_corpus,
                  '--out', str(out)])
    elapsed = time.time() - start
    assert rc == 0
    assert elapsed < 240, 'catalogue build took %.1fs, over the 4-minute budget' % elapsed
    for name in ('sections.json', 'sections.csv', 'beats.json', 'beats.csv', 'audio.json'):
        assert (out / name).is_file()
    assert (out / 'cache').is_dir()
    assert len(list((out / 'cache').glob('*.json'))) == 20

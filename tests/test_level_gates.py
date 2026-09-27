"""Offline gates: extern cross-check, .sec existence, class/property checks,
the headless VM harness, and .smb placement bounds -- all off a wire.Level."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import copy
import glob
import os
import re
import struct

import pytest

from dante import module as dante_module

from gbtvgr import lvl as _lvl
from gbtvgr.sets import bst as bstmod
from gbtvgr.level import checks, lang, placement, vmharness, wire

pytestmark = pytest.mark.filterwarnings('ignore')


def _stems(lvl_corpus):
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in
                  glob.glob(os.path.join(lvl_corpus, 'world', '*.lvl')))


# ----------------------------------------------------------------------------
# 1. extern cross-check, both directions, over every shipped level.
#
# Findings that are pre-existing shipped content, not check bugs (verified
# this pass, see B2_REPORT_GATES.md): a handful of dangling debug externs
# (the class/name pair is placed nowhere, in any level) and a few level-local
# event-field bindings to a function name that is genuinely unexported --
# either a vestigial donor-block field the engine's reflection lookup simply
# never fires, or (the _onActorEnter/_onArrival family) an auto-derived slot
# name the engine tolerates missing.  Anything NOT on these lists is a real
# finding and fails the test.
# ----------------------------------------------------------------------------
KNOWN_DANGLING_EXTERNS = {
    ('CAniModel', 'Door1'), ('CCinematicSkeletonModel', 'csm_LibrarianBook'),
    ('CGhostbuster', 'player'), ('CWayPoint', 'WP_WarpMe'),
    ('CWayPoint', 'WP_RayStart'),
}
KNOWN_UNEXPORTED_HANDLERS = {
    'GenericCancelDamage_preGetHurt', 'trig_flashlight_On',
    'pp_sA_1h_advancePoint_onArrival', 'ppAdvance_StaypuftLobbyRay_onArrival',
    'ppAdvance_StaypuftThrowEndRay_onArrival', 'ppAdvance_StaypuftThrow_onArrival',
}
_EXTERN_MISS_RE = re.compile(r'^extern (\w+) (\w+): no such actor')
_HANDLER_MISS_RE = re.compile(r'^the level binds (\w+)\(\), which the script does not export$')


def _explained(b, global_defined):
    m = _EXTERN_MISS_RE.match(b)
    if m:
        return (m.group(1), m.group(2)) in KNOWN_DANGLING_EXTERNS
    m = _HANDLER_MISS_RE.match(b)
    if m:
        fn = m.group(1)
        if fn in global_defined or fn in KNOWN_UNEXPORTED_HANDLERS:
            return True
        # a CTrigger's auto-derived slot name (checks.TRIG_SLOTS): the engine
        # calls it if exported and silently skips it otherwise, never fatal.
        return any(fn.endswith('_' + slot) for slot in checks.TRIG_SLOTS)
    return False


def test_extern_cross_check_corpus(lvl_corpus):
    src = wire.DirSource(lvl_corpus)
    stems = _stems(lvl_corpus)
    assert stems, 'no .lvl files under %s/world' % lvl_corpus
    global_path = os.path.join(lvl_corpus, 'world', 'global.dante')
    global_defined = (checks.script_handlers(dante_module.load(global_path))
                      if os.path.exists(global_path) else set())

    total, unexplained = 0, []
    for stem in stems:
        level = wire.open_level(src, stem, load_set=False)
        bad, stats = checks.cross_check_level(level)
        assert stats['nactors'] > 0, '%s: no actors resolved at all' % stem
        total += len(bad)
        for b in bad:
            if not _explained(b, global_defined):
                unexplained.append('%s: %s' % (stem, b))
    print('cross_check_level: %d findings across %d shipped levels, all '
          'explained (dangling debug externs / vestigial event fields)' % (total, len(stems)))
    assert not unexplained, 'unexplained cross-check findings:\n  ' + '\n  '.join(unexplained)


# ----------------------------------------------------------------------------
# 2. every .sec a section-list names must have actually loaded.
# ----------------------------------------------------------------------------
def test_sec_existence_check(lvl_corpus, tmp_path):
    src = wire.DirSource(lvl_corpus)
    level = wire.open_level(src, 'cemetery1', load_set=False)
    assert checks.check_sections(level) == []

    # a hand-mutated Level (no filesystem): a section-list entry that names
    # nothing check_sections has in level.secs must be flagged, fatally.
    doc = copy.deepcopy(level.lvl)
    sl = _lvl.top_block(doc, 'section-list')
    kv = next(c for c in sl.children if isinstance(c, _lvl.KV))
    kv.value = 'NoSuchSectionAtAll'
    mutated = wire.Level(level.stem, doc, secs=level.secs)
    bad = checks.check_sections(mutated)
    assert any('NoSuchSectionAtAll' in b for b in bad)

    # the loader itself is just as strict: a copy of the corpus with one
    # section-list value renamed to a file that does not exist fails to open.
    world_dir = tmp_path / 'world'
    world_dir.mkdir()
    for name in os.listdir(os.path.join(lvl_corpus, 'world')):
        if name.lower() in ('cemetery1.lvl', 'cemetery1art.sec', 'cemetery1sound1.sec'):
            (world_dir / name).write_bytes(
                open(os.path.join(lvl_corpus, 'world', name), 'rb').read())
    raw = (world_dir / 'cemetery1.lvl').read_bytes()
    # target the section-list line itself, not an earlier Dependencies/actor
    # mention of the same substring
    needle = b'Section_2 = Cemetery1ART\r\n'
    assert raw.count(needle) == 1
    mutated_raw = raw.replace(needle, b'Section_2 = Cemetery1ARTMISSING\r\n', 1)
    assert mutated_raw != raw
    (world_dir / 'cemetery1.lvl').write_bytes(mutated_raw)
    with pytest.raises(wire.LevelError):
        wire.open_level(str(tmp_path), 'cemetery1', load_set=False)


# ----------------------------------------------------------------------------
# 3. class check (fatal on a genuinely unknown class) and property check
# (warn only) over a hand-mutated copy of a shipped level.
# ----------------------------------------------------------------------------
def test_class_and_property_check(lvl_corpus):
    src = wire.DirSource(lvl_corpus)
    level = wire.open_level(src, 'boss_sp_side', load_set=False)

    doc = copy.deepcopy(level.lvl)
    al = _lvl.top_block(doc, 'actor-list')
    kv = next(c for c in al.children if isinstance(c, _lvl.KV))
    kv.value = 'CTotallyFakeClassXYZ'
    mutated = wire.Level(level.stem, doc, secs=[])
    bad = checks.class_check(mutated)
    assert any('CTotallyFakeClassXYZ' in b for b in bad)
    # the untouched level must not itself trip on its OWN class names here
    assert not any(kv.key in b for b in bad if 'CTotallyFakeClassXYZ' not in b)

    doc2 = copy.deepcopy(level.lvl)
    actors_block = _lvl.top_block(doc2, 'actors')
    first_actor = next(c for c in actors_block.children if isinstance(c, _lvl.Block))
    first_actor.children.append(_lvl.KV('totallyFakeFieldXYZ', '1'))
    mutated2 = wire.Level(level.stem, doc2, secs=[])
    warn = checks.prop_check(mutated2)
    assert any('totallyFakeFieldXYZ' in w for w in warn)
    # never fatal: prop_check only ever returns strings, it must not raise
    assert isinstance(warn, list)


# ----------------------------------------------------------------------------
# 4. vmharness: the World from the Level's own actor table, main() driven for
# a bounded tick count, no Python exception, and at least one interesting call.
# ----------------------------------------------------------------------------
def test_vmharness_bounded_run(lvl_corpus):
    lib = os.path.join(lvl_corpus, 'world', 'global.dante')
    if not os.path.exists(lib):
        pytest.skip('no global.dante under %s/world' % lvl_corpus)
    src = wire.DirSource(lvl_corpus)
    level = wire.open_level(src, 'boss_sp_side', load_set=False)
    assert level.script is not None

    pool = vmharness.read_pool_level(level)
    assert pool, 'boss_sp_side has no dormant spawn pool at all'

    interesting = ('spawnCharacter', 'setMusic', 'defineCheckpoint',
                  'saveCheckpoint', 'setCurrentObjective')
    world = vmharness.ScenarioWorld({}, interesting, pool=pool, verbose=False)
    vm = vmharness.build_level(level, [lib], world)
    vm.call('void setupLevel()')          # must not raise
    vm.start('void main()')
    ok = vmharness.run_bounded(vm, 1500, timeout_s=90)
    assert ok, 'boss_sp_side main() never reached idle() within the tick bound'
    kinds = {e[1] for e in world.events}
    assert 'spawnCharacter' in kinds or 'setMusic' in kinds, (
        'expected at least one spawnCharacter or setMusic call; saw %s' % kinds)


# ----------------------------------------------------------------------------
# 5. placement: one prop from SMB_CORPUS, inside a shipped section's bbox
# passes, well outside it fails.
# ----------------------------------------------------------------------------
def test_placement_in_section(smb_corpus, bst_corpus):
    smb_dir = os.path.join(smb_corpus, 'models')
    model = 'graveyard\\spike_bottom'
    if placement.smb_meta_dir(smb_dir, model) is None:
        pytest.skip('%s.smb not under %s' % (model, smb_dir))
    bst_path = os.path.join(bst_corpus, 'cemetery1.bst')
    if not os.path.exists(bst_path):
        pytest.skip('cemetery1.bst not under %s' % bst_corpus)
    m = bstmod.parse(open(bst_path, 'rb').read())

    def volume(s):
        lo = struct.unpack_from('<3f', s['bbox'], 0)
        hi = struct.unpack_from('<3f', s['bbox'], 12)
        return (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])

    section = max(m['sections'], key=volume)
    lo = struct.unpack_from('<3f', section['bbox'], 0)
    hi = struct.unpack_from('<3f', section['bbox'], 12)
    cx, cy, cz = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0

    assert placement.in_section(smb_dir, model, cx, cy, cz, 0.0, section['bbox']) is True
    assert placement.in_section(smb_dir, model, cx + 5000.0, cy, cz, 0.0,
                                section['bbox']) is False

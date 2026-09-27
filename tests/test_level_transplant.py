"""Transplant: the reference gate on every shipped set, the cemetery1 -> abyss
experiment replayed to its md5, a pair the experiment never tried, two appends."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import glob
import hashlib
import os
import struct

import pytest

from _run import gbtvgr, ok
from gbtvgr.level import transplant as tp
from gbtvgr.sets import bst, geom

EXPERIMENT_MD5 = 'e98e11c81ccca571bee7a6fa409cd234'


def _read(path):
    with open(path, 'rb') as fh:
        return fh.read()


def _corpus_set(bst_corpus, name):
    path = os.path.join(bst_corpus, name + '.bst')
    if not os.path.isfile(path):
        pytest.skip('needs %s in the .bst corpus' % name)
    return path


def _centre(s):
    b = struct.unpack('<6f', s['bbox'])
    return tuple((b[k] + b[k + 3]) / 2.0 for k in range(3))


def _gates(path):
    """The two shipped CLI gates plus the two module gates, on a written set."""
    ok(gbtvgr('bst', 'verify', path))
    out = ok(gbtvgr('bst-geom', 'check', path))
    assert 'FAIL' not in out, out


def test_check_refs_clean_on_corpus(bst_corpus):
    files = sorted(glob.glob(os.path.join(bst_corpus, '**', '*.bst'), recursive=True))
    assert files
    bad = {}
    for f in files:
        problems = tp.check_refs(bst.parse(_read(f)))
        if problems:
            bad[os.path.basename(f)] = problems[:5]
    assert not bad, bad


def test_records_roundtrip(bst_corpus):
    m = bst.parse(_read(_corpus_set(bst_corpus, 'abyss')))
    assert all(tp.pack_probe(tp.unpack_probe(p)) == p for p in m['portals'])
    assert all(tp.pack_light(tp.unpack_light(l)) == l for l in m['lights'])
    slot = m['sections'][0]['names3'][:100]
    renamed = tp.set_slot_name(slot, 'lightmap\\x\\9_0.tga')
    assert tp.slot_name(renamed) == 'lightmap\\x\\9_0.tga'
    assert renamed[:8] == slot[:8] and renamed[0x48:] == slot[0x48:]


def test_experiment_replay(bst_corpus, tmp_path):
    host_path = _corpus_set(bst_corpus, 'abyss')
    donor = bst.parse(_read(_corpus_set(bst_corpus, 'cemetery1')))
    before = bst.parse(_read(host_path))
    host, staged = tp.append_section(bst.parse(_read(host_path)), donor, 42,
                                     (-400.0, 30.0, 150.0), lightmap_dir='abyss',
                                     split=(0, -280.0), unembed=False)
    out = bst.build(host)
    assert hashlib.md5(out).hexdigest() == EXPERIMENT_MD5
    assert len(staged) == 14 and staged[0] == ('art\\lightmap\\cemetery1\\42_0.tex',
                                               'art\\lightmap\\abyss\\2_0.tex')
    after = bst.parse(out)
    assert tp.check_refs(after) == []
    assert tp.check_additive(before, after) == []
    path = str(tmp_path / 'abyss_plus.bst')
    with open(path, 'wb') as fh:
        fh.write(out)
    _gates(path)
    # Defaults differ only in the host's own directory spelling and the bisected plane.
    auto, _ = tp.append_section(bst.parse(_read(host_path)), donor, 42, (-400.0, 30.0, 150.0), unembed=False)
    out2 = bst.build(auto)
    assert len(out2) == len(out)
    diff = [i for i in range(len(out)) if out[i] != out2[i]]
    assert len(diff) == 19
    assert tp.slot_name(auto['sections'][2]['names3'][:100]) == 'lightmap\\Abyss\\2_0.tga'
    assert geom.bsp_nodes(auto)[0][0] == (-1.0, 0.0, 0.0, 277.55413818359375)


def test_second_pair(bst_corpus, tmp_path):
    """library1b Rm_Kids: 30 materials with 11 embedded, 18 probes, 9 shared lights."""
    host_path = _corpus_set(bst_corpus, 'boss_sp_side')
    donor = bst.parse(_read(_corpus_set(bst_corpus, 'library1b')))
    before = bst.parse(_read(host_path))
    host, staged = tp.append_section(bst.parse(_read(host_path)), donor, 8, (0.0, 0.0, -250.0), unembed=False)
    out = bst.build(host)
    after = bst.parse(out)
    assert len(after['sections']) == 16
    assert tp.check_refs(after) == []
    assert tp.check_additive(before, after) == []
    assert len(staged) == 4 + 2 * 18
    assert geom.bsp_nodes(after)[0][0] == (0.0, 0.0, 1.0, -219.75)
    new = after['sections'][15]
    assert tp._u32s(new['portidx']) == list(range(16, 34))
    assert all(tp.unpack_light(after['lights'][li])['sections'] == [15]
               for li in tp._u32s(new['arr818']))
    assert all(after['materials'][tp._h70(me)] == donor['materials'][tp._h70(dme)]
               for me, dme in zip(new['meshes'], donor['sections'][8]['meshes']))
    path = str(tmp_path / 'boss_plus.bst')
    with open(path, 'wb') as fh:
        fh.write(out)
    _gates(path)


def test_two_appends_keep_the_bsp(bst_corpus):
    host_path = _corpus_set(bst_corpus, 'boss_sp_side')
    donor = bst.parse(_read(_corpus_set(bst_corpus, 'library1b')))
    before = bst.parse(_read(host_path))
    host = bst.parse(_read(host_path))
    host, _ = tp.append_section(host, donor, 8, (0.0, 0.0, -250.0))
    host, _ = tp.append_section(host, donor, 4, (-300.0, 0.0, -250.0))
    after = bst.parse(bst.build(host))
    assert tp.check_refs(after) == []
    assert tp.check_additive(before, after) == []
    old, new = geom.bsp_nodes(before), geom.bsp_nodes(after)
    assert len(new) == len(old) + 4
    for i, s in enumerate(before['sections']):
        c = _centre(s)
        assert new[tp.bsp_walk(after, c)][1] == old[tp.bsp_walk(before, c)][1], i
    for i in (15, 16):
        assert new[tp.bsp_walk(after, _centre(after['sections'][i]))][1] == i
    assert sorted(n[1] for n in new if n[1] >= 0) == \
        sorted(n[1] for n in old if n[1] >= 0) + [15, 16]


def test_translate_in_place(bst_corpus):
    m = geom.capture(bst.parse(_read(_corpus_set(bst_corpus, 'cemetery1'))))
    moved = tp.translate_section(m, 42, (-400.0, 30.0, 150.0))
    assert (moved['meshes'], moved['probes'], moved['nav_nodes']) == (91, 5, 34)
    assert moved['skipped_lights'] == [59, 69]
    after = bst.parse(bst.build(geom.relayout(m)))
    assert tp.check_refs(after) == []
    bb = struct.unpack('<6f', after['sections'][42]['bbox'])
    assert tuple(round(x, 2) for x in bb[:3]) == (-365.02, -4.66, -24.06)


def test_cli_check_additive(bst_corpus, tmp_path):
    host_path = _corpus_set(bst_corpus, 'abyss')
    same = str(tmp_path / 'same.bst')
    with open(same, 'wb') as fh:
        fh.write(_read(host_path))
    assert tp.main(['check-additive', host_path, same]) == 0
    assert tp.main(['check-refs', host_path]) == 0

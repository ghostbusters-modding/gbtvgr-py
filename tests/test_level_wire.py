"""Whole levels: .lvl + every .sec + .bst open and write back byte-identical, in engine order."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import collections
import os
import tempfile

import pytest

from gbtvgr.level import wire


@pytest.fixture(scope="session")
def corpus(lvl_corpus, bst_corpus):
    """world/ from the .lvl corpus, sets/ from the .bst corpus."""
    return wire.DirSource(lvl_corpus, sets=bst_corpus)


def _source_bytes(src, rel):
    """The corpus file behind a written path, matched without regard to case."""
    return src.read(rel.replace(os.sep, '\\'))


def test_whole_level_roundtrip(corpus):
    """open_level + write_level reproduces every shipped file byte for byte."""
    stems = corpus.level_stems()
    assert len(stems) == 20, stems
    written = collections.Counter()
    bad = []
    for stem in stems:
        lv = wire.open_level(corpus, stem, load_set=True)
        with tempfile.TemporaryDirectory(prefix="lvlwire_") as td:
            for p in wire.write_level(lv, td):
                rel = os.path.relpath(p, td)
                with open(p, 'rb') as fh:
                    data = fh.read()
                orig = _source_bytes(corpus, rel)
                if orig != data:
                    bad.append(rel)
                written[os.path.splitext(rel)[1]] += 1
        del lv
    assert not bad, bad
    assert written['.lvl'] == 20 and written['.sec'] == 67 and written['.bst'] == 20, written


def test_section_recursion_matches_engine(corpus):
    """Every section-list entry resolves, in order, and hotel1a's layer counts match the RE."""
    total = 0
    for stem in corpus.level_stems():
        lv = wire.open_level(corpus, stem, load_set=False)
        names = wire.sec_names(lv.lvl)
        assert [n for n, _ in lv.secs] == names, stem
        for n, _ in lv.secs:
            assert corpus.has('world\\%s.sec' % n), (stem, n)
        total += len(lv.secs)
    assert total == 67

    lv = wire.open_level(corpus, 'hotel1a', load_set=False)
    per_index = collections.Counter(idx for idx, _ in wire.actors(lv))
    assert per_index[0] == 592
    assert sum(n for idx, n in per_index.items() if idx > 0) == 2006
    assert len(lv.secs) == 8


def test_library1b_doors_live_in_layers(corpus):
    """All 27 CAniModel actors of library1b sit in .sec layers, none in the .lvl."""
    lv = wire.open_level(corpus, 'library1b', load_set=False)
    classes = wire.actor_classes(lv)
    doors = [(idx, a.tag) for idx, a in wire.actors(lv)
             if classes[idx].get(a.tag) == 'CAniModel']
    assert len(doors) == 27, doors
    assert all(idx > 0 for idx, _ in doors), doors


def test_library_matches_corpus(game_dir, corpus):
    """The POD chain and the extracted corpus give the same bytes for two levels."""
    from gbtvgr.level.library import Library
    with Library(game_dir).open() as lib:
        for stem in ('hotel1a', 'library1b'):
            lv = wire.open_level(lib, stem, load_set=True)
            assert 'en' in lv.lang, stem
            for name, data in wire.serialize_level(lv):
                if name.lower().endswith('.txt'):
                    continue  # the corpus carries no lang tables
                assert corpus.read(name) == data, name

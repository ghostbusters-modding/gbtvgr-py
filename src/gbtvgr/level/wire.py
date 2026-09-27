"""One level the way ghost.exe reads it: the .lvl, every .sec it names, the .bst, script, lang tables.
Glue over gbtvgr.lvl and gbtvgr.sets.bst; the whole-level byte round trip lives here."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import argparse
import collections
import contextlib
import io
import os
import re
import sys
import tempfile

from .. import lvl as _lvl
from ..sets import bst as _bst


class LevelError(Exception):
    pass


class Level:
    """secs is in engine order: depth-first through every <section-list>, so index N = levelSection N."""
    __slots__ = ('stem', 'set_name', 'bst', 'lvl', 'secs', 'script', 'lang')

    def __init__(self, stem, lvl, secs=None, set_name=None, bst=None, script=None, lang=None):
        self.stem = stem
        self.lvl = lvl
        self.secs = list(secs or [])
        self.set_name = set_name
        self.bst = bst
        self.script = script
        self.lang = dict(lang or {})

    def docs(self):
        """[(name, Block)] with the .lvl at index 0, then the .sec layers in engine order."""
        return [(self.stem, self.lvl)] + list(self.secs)


# ============================================================================
# .lvl header helpers
# ============================================================================

def sec_names(block):
    """The <section-list> values in file order; the engine ignores the Section_N numbers."""
    sl = _lvl.top_block(block, 'section-list')
    if sl is None:
        return []
    return [c.value for c in sl.children if isinstance(c, _lvl.KV)]


def set_stem(block):
    """set-filename minus its extension: the engine swaps .pst for .bst itself."""
    kv = _lvl.top_kv(block, 'set-filename')
    if kv is None:
        return None
    return os.path.splitext(kv.value.strip())[0]


# ============================================================================
# sources: anything with read(archive_path) and names(prefix=, ext=)
# ============================================================================

def _key(name):
    return name.lower().replace('/', '\\')


class DirSource:
    """Loose files under root/world and root/sets, or two dirs given apart. Lookups ignore case."""

    def __init__(self, root=None, world=None, sets=None):
        if root is None and world is None:
            raise LevelError('DirSource needs a root or a world dir')
        self.world = world or os.path.join(root, 'world')
        self.sets = sets or (os.path.join(root, 'sets') if root else None)
        self._map = None

    def _index(self):
        """lowercased archive path -> (archive path as spelled on disk, file path)."""
        if self._map is None:
            self._map = {}
            for top, d in (('world', self.world), ('sets', self.sets)):
                if not d or not os.path.isdir(d):
                    continue
                for dirpath, _, files in os.walk(d):
                    rel = os.path.relpath(dirpath, d)
                    for f in files:
                        parts = [top] + ([] if rel == '.' else rel.split(os.sep)) + [f]
                        name = '\\'.join(parts)
                        self._map.setdefault(_key(name), (name, os.path.join(dirpath, f)))
        return self._map

    def path(self, name):
        hit = self._index().get(_key(name))
        return hit[1] if hit else None

    def has(self, name):
        return _key(name) in self._index()

    def read(self, name):
        p = self.path(name)
        if p is None:
            return None
        with open(p, 'rb') as fh:
            return fh.read()

    def names(self, category=None, ext=None, prefix=None, contains=None):
        pre = _key(prefix) if prefix else None
        needle = contains.lower() if contains else None
        out = []
        for k, (name, _) in self._index().items():
            if pre and not k.startswith(pre):
                continue
            if ext and not k.endswith(ext.lower()):
                continue
            if needle and needle not in k:
                continue
            out.append(name)
        return sorted(out, key=str.lower)

    def level_stems(self):
        return [n.split('\\')[-1][:-4] for n in self.names(prefix='world\\', ext='.lvl')
                if n.count('\\') == 1]

    def set_stems(self):
        return [n.split('\\')[-1][:-4] for n in self.names(prefix='sets\\', ext='.bst')]


def open_source(source, sets=None):
    """A str is a game dir when it holds COMMON.POD, else a corpus dir; objects pass through."""
    if not isinstance(source, str):
        return source
    if os.path.isfile(os.path.join(source, 'COMMON.POD')):
        from .library import Library
        return Library(source).open()
    return DirSource(source, sets=sets)


# ============================================================================
# open / write
# ============================================================================

def _parse_doc(name, data):
    try:
        return _lvl.parse_lvl(data)
    except _lvl.LvlError as e:
        raise LevelError('%s: %s' % (name, e))


def _load_secs(src, block, out):
    for name in sec_names(block):
        data = src.read('world\\%s.sec' % name)
        if data is None:
            raise LevelError("Can't open section %s.sec" % name)
        sec = _parse_doc(name + '.sec', data)
        out.append((name, sec))
        _load_secs(src, sec, out)


def open_level(source, stem, load_set=True):
    """Everything the engine opens for one level; a missing .sec or .bst is fatal, as in the engine."""
    src = open_source(source)
    data = src.read('world\\%s.lvl' % stem)
    if data is None:
        raise LevelError('no such level: world\\%s.lvl' % stem)
    doc = _parse_doc(stem + '.lvl', data)
    secs = []
    _load_secs(src, doc, secs)
    # A .sec's own set-filename overwrites the level's in the engine; the last one parsed wins.
    set_name = set_stem(doc)
    for _, sec in secs:
        set_name = set_stem(sec) or set_name
    parsed = None
    if load_set:
        if set_name is None:
            raise LevelError('%s.lvl has no set-filename' % stem)
        raw = src.read('sets\\%s.bst' % set_name)
        if raw is None:
            raise LevelError("Can't open set sets\\%s.bst" % set_name)
        parsed = _bst.parse(raw)
    script = src.read('world\\%s.dante' % stem)
    lang = collections.OrderedDict()
    pat = re.compile(r'^world\\([^\\]+)\\%s\.txt$' % re.escape(stem.lower()))
    for n in src.names(prefix='world\\', ext='.txt'):
        m = pat.match(_key(n))
        if m:
            lang[m.group(1)] = src.read(n)
    return Level(stem, doc, secs, set_name, parsed, script, lang)


def serialize_level(level, with_set=True):
    """[(archive path, bytes)] for every file write_level would write."""
    out = [('world\\%s.lvl' % level.stem, _lvl.serialize_lvl(level.lvl))]
    for name, sec in level.secs:
        out.append(('world\\%s.sec' % name, _lvl.serialize_lvl(sec)))
    if with_set and level.bst is not None:
        out.append(('sets\\%s.bst' % level.set_name, _bst.build(level.bst)))
    if level.script is not None:
        out.append(('world\\%s.dante' % level.stem, level.script))
    for lang, data in level.lang.items():
        out.append(('world\\%s\\%s.txt' % (lang, level.stem), data))
    return out


def write_level(level, outdir):
    """Writes the level as loose files under outdir; returns the paths written."""
    paths = []
    for name, data in serialize_level(level):
        p = os.path.join(outdir, *name.split('\\'))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'wb') as fh:
            fh.write(data)
        paths.append(p)
    return paths


# ============================================================================
# actors
# ============================================================================

def actors(level):
    """(section_index, actor Block): 0 is the .lvl, N the N-th .sec, matching the engine's levelSection."""
    for idx, (_, doc) in enumerate(level.docs()):
        ab = _lvl.top_block(doc, 'actors')
        if ab is None:
            continue
        for a in ab.children:
            if isinstance(a, _lvl.Block):
                yield idx, a


def actor_classes(level):
    """Per file index, the <actor-list> name -> class map of that file."""
    return [_lvl.actor_class_map(doc) for _, doc in level.docs()]


def actor_count(doc):
    ab = _lvl.top_block(doc, 'actors')
    return sum(1 for a in ab.children if isinstance(a, _lvl.Block)) if ab else 0


# ============================================================================
# validate
# ============================================================================

def validate(level, dante_api=None):
    """gbtvgr.lvl's validate over the .lvl and every .sec: [(file name, [problem])]."""
    # cmd_validate reads a path and prints, so each doc goes through a temp file.
    results = []
    with tempfile.TemporaryDirectory(prefix='gbtvgr_level_') as td:
        for name, data in serialize_level(level, with_set=False):
            if not name.lower().endswith(('.lvl', '.sec')):
                continue
            path = os.path.join(td, name.split('\\')[-1])
            with open(path, 'wb') as fh:
                fh.write(data)
            args = argparse.Namespace(file=path, verbose=False,
                                      dante_api=dante_api or _lvl.DEFAULT_DANTE_API)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _lvl.cmd_validate(args)
            problems = [ln.strip()[len('FAIL: '):] for ln in buf.getvalue().splitlines()
                        if ln.strip().startswith('FAIL: ')]
            results.append((name.split('\\')[-1], problems))
    return results


# ============================================================================
# CLI
# ============================================================================

def _first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def cmd_info(a):
    src = open_source(a.source, a.sets)
    lv = open_level(src, a.stem, load_set=not a.no_set)
    print('%s: set %s, %d sections' % (lv.stem, lv.set_name, len(lv.secs)))
    for idx, (name, doc) in enumerate(lv.docs()):
        ext = '.lvl' if idx == 0 else '.sec'
        print('  [%d] %-28s %5d actors' % (idx, name + ext, actor_count(doc)))
    if lv.script is not None:
        print('  script          %d bytes' % len(lv.script))
    if lv.lang:
        print('  lang            %s' % ', '.join(lv.lang))
    m = lv.bst
    if m is not None:
        print('  set sections    %d' % len(m['sections']))
        print('  materials       %d' % len(m['materials']))
        print('  probes          %d' % len(m['portals']))
        print('  lights          %d' % len(m['lights']))
        print('  nav nodes       %s' % ('empty' if m['nav']['empty'] else
                                        '%d (%d verts)' % (m['nav']['nnodes'], m['nav']['nverts'])))
        print('  bsp nodes       %d' % (len(m['bsp']) // 0x30))
    return 0


def roundtrip_level(src, stem, with_set=True):
    """[(archive path, ok, message)] for one level: open, rebuild, compare to the source bytes."""
    lv = open_level(src, stem, load_set=with_set)
    out = []
    for name, data in serialize_level(lv):
        orig = src.read(name)
        if orig is None:
            out.append((name, False, 'source file missing'))
        elif orig == data:
            out.append((name, True, '%d bytes' % len(data)))
        else:
            out.append((name, False, 'byte mismatch at offset %d (orig %d bytes, rebuilt %d bytes)'
                        % (_first_diff(orig, data), len(orig), len(data))))
    return out


def cmd_roundtrip(a):
    src = open_source(a.source, a.sets)
    stems = a.stems or src.level_stems()
    ok = fail = 0
    for stem in stems:
        try:
            rows = roundtrip_level(src, stem, with_set=not a.no_set)
        except (LevelError, ValueError) as e:
            print('FAIL  %s: %s' % (stem, e))
            fail += 1
            continue
        for name, good, msg in rows:
            print('%s  %s  (%s)' % ('OK  ' if good else 'FAIL', name, msg))
            if good:
                ok += 1
            else:
                fail += 1
    print('roundtrip: %d/%d files passed' % (ok, ok + fail))
    return 0 if not fail else 1


def cmd_validate(a):
    src = open_source(a.source, a.sets)
    lv = open_level(src, a.stem, load_set=False)
    bad = 0
    for name, problems in validate(lv, a.dante_api):
        if problems:
            bad += 1
            print('FAIL  %s' % name)
            for p in problems:
                print('        %s' % p)
        else:
            print('OK    %s' % name)
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Whole levels: the .lvl, its .sec layers and its .bst, read and written together.')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('source', help='a dir holding world/ and sets/, or the game directory')
        p.add_argument('--sets', help='a separate dir of .bst files, when sets/ is not under source')

    p = sub.add_parser('info', help='file, actor and set counts for one level')
    common(p)
    p.add_argument('stem')
    p.add_argument('--no-set', action='store_true', help='skip the .bst')
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser('roundtrip-test', help='open and rebuild every level, byte-compare every file')
    common(p)
    p.add_argument('stems', nargs='*', help='level stems (default: every .lvl in world/)')
    p.add_argument('--no-set', action='store_true', help='skip the .bst')
    p.set_defaults(fn=cmd_roundtrip)

    p = sub.add_parser('validate', help="gbtvgr lvl validate over the .lvl and each .sec")
    common(p)
    p.add_argument('stem')
    p.add_argument('--dante-api', default=_lvl.DEFAULT_DANTE_API)
    p.set_defaults(fn=cmd_validate)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except LevelError as e:
        print('error: %s' % e, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

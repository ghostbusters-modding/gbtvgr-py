"""The game library: archives mounted the way the engine mounts them, indexed
in place, read by archive path."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import json
import os
import sys
import threading
import time
import zlib

from ..archive import pod
from ..archive.assets import ArtIndex
from ..sets import bst
from .materials import (MaterialInfo, MODEL_BIT, SET_BIT, LAYER_TYPES, cstr, tex_path,
                        parse as parse_material, info as material_info)

# CPod::mountDefaultPods, in order; each archive's chain field mounts after it
MOUNT_ORDER = ('PATCH.POD', 'W64ENSND.POD', 'W64MUSND.POD', 'W64ART.POD',
               'W64SOUND.POD', 'W64SET.POD', 'W64MODEL.POD', 'LANGUAGE.POD',
               'COMMON.POD')

CATEGORIES = {
    'meshes': ('models\\', '.smb'),
    'materials': ('materials\\', '.mtb'),
    'textures': ('art\\', '.tex'),
    'lightmaps': ('art\\lightmap\\', '.tex'),
    'lightprobes': ('art\\lightprobe\\', '.tex'),
    'sets': ('sets\\', '.bst'),
    'levels': ('world\\', '.lvl'),
    'scripts': ('world\\', '.dante'),
    'characters': ('data\\', '.cib'),
    'skeletal': ('skeletal\\', '.dfm'),
    'effects': ('fx\\', '.tfa'),
}


def _key(name):
    return name.lower().replace('/', '\\')


def default_game_dir():
    """The game directory from the environment, or the usual Steam locations."""
    env = os.environ.get('GBTVGR_GAME') or os.environ.get('GAME_DIR')
    if env and os.path.isfile(os.path.join(env, 'COMMON.POD')):
        return env
    name = 'Ghostbusters The Video Game Remastered'
    roots = ['C:\\Program Files (x86)\\Steam\\steamapps\\common',
             'C:\\Program Files\\Steam\\steamapps\\common']
    for drive in 'CDEFGHXYZ':
        roots.append('%s:\\SteamLibrary\\steamapps\\common' % drive)
        roots.append('/mnt/%s/SteamLibrary/steamapps/common' % drive.lower())
    for r in roots:
        p = os.path.join(r, name)
        if os.path.isfile(os.path.join(p, 'COMMON.POD')):
            return p
    return None


class Entry:
    __slots__ = ('name', 'pod', 'entry', 'size')

    def __init__(self, name, p, e):
        self.name = name
        self.pod = p
        self.entry = e
        self.size = e['usize']

    @property
    def archive(self):
        """The archive's file name, or the loose file's directory for an extra dir."""
        return os.path.basename(self.pod.path) if self.pod else self.entry['dir']

    @property
    def ext(self):
        return os.path.splitext(self.name)[1].lower()

    @property
    def stem(self):
        return os.path.splitext(self.name.split('\\')[-1])[0]


class Library:
    """The mounted POD chain. The material scan caches to disk only when cache_dir is given."""

    def __init__(self, game_dir, extra_dirs=(), cache_dir=None):
        self.game_dir = game_dir
        self.extra_dirs = [d for d in extra_dirs if d and os.path.isdir(d)]
        self.cache_dir = cache_dir
        self.pods = []
        self.map = {}
        self.mount_log = []
        self._materials = None
        self._mat_lock = threading.Lock()
        self._art = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- mounting ---------------------------------------------------------------
    def open(self):
        self.close()
        seen = set()
        for name in MOUNT_ORDER:
            self._mount_chain(os.path.join(self.game_dir, name), seen)
        for d in self.extra_dirs:
            self._mount_dir(d)
        return self

    def _mount_dir(self, d):
        """A mod's loose tree (sets/, world/, art/, ...) listed like an archive; read()
        already prefers it, so a mod's set and level show up beside the shipped ones."""
        n = 0
        for root, _dirs, files in os.walk(d):
            for f in files:
                full = os.path.join(root, f)
                name = os.path.relpath(full, d).replace(os.sep, '\\')
                if name.startswith('.') or name.lower().endswith(('.py', '.md', '.ini', '.sh', '.dn')):
                    continue
                self.map[_key(name)] = Entry(name, None, {'usize': os.path.getsize(full), 'dir': d})
                n += 1
        self.mount_log.append('%s: %d loose files' % (d, n))

    def _mount_chain(self, path, seen):
        while path and os.path.isfile(path):
            k = os.path.basename(path).lower()
            if k in seen:
                break
            seen.add(k)
            try:
                p = pod.Pod(path)
            except (OSError, ValueError) as exc:
                self.mount_log.append('%s: %s' % (os.path.basename(path), exc))
                break
            self.pods.append(p)
            for e in p.entries:
                self.map.setdefault(_key(e['name']), Entry(e['name'], p, e))
            self.mount_log.append('%s: %d entries' % (os.path.basename(path), len(p.entries)))
            path = os.path.join(os.path.dirname(path), p.next_pod) if p.next_pod else None

    def close(self):
        for p in self.pods:
            p.close()
        self.pods = []
        self.map = {}
        if self._art is not None:
            self._art.close()
            self._art = None

    def __bool__(self):
        return bool(self.pods)

    # -- lookup -------------------------------------------------------------------
    def entry(self, name):
        return self.map.get(_key(name))

    def has(self, name):
        return _key(name) in self.map

    def read(self, name):
        """The bytes of an asset by archive path, or None. Extra dirs win."""
        k = _key(name)
        for d in self.extra_dirs:
            p = os.path.join(d, *k.split('\\'))
            if os.path.isfile(p):
                with open(p, 'rb') as fh:
                    return fh.read()
        e = self.map.get(k)
        if e is None or e.pod is None:
            return None
        try:
            return e.pod.read(e.entry)
        except (OSError, ValueError, zlib.error):
            return None

    def art(self):
        """An ArtIndex over the same install, for the codec helpers that want one."""
        if self._art is None:
            self._art = ArtIndex(self.game_dir, self.extra_dirs)
        return self._art

    def names(self, category=None, ext=None, prefix=None, contains=None):
        if category:
            prefix = prefix or CATEGORIES[category][0]
            ext = ext or CATEGORIES[category][1]
        pre = _key(prefix) if prefix else None
        needle = contains.lower() if contains else None
        out = []
        for k, e in self.map.items():
            if pre and not k.startswith(pre):
                continue
            if ext and not k.endswith(ext):
                continue
            if needle and needle not in k:
                continue
            out.append(e.name)
        out.sort(key=str.lower)
        return out

    def tree(self, category):
        """{dir: [names]} for the asset browser."""
        out = {}
        for n in self.names(category):
            d, _, base = n.rpartition('\\')
            out.setdefault(d, []).append(base)
        return out

    # -- typed reads --------------------------------------------------------------
    def model_ref_path(self, ref):
        """A .lvl `modelInstance` / component mesh ref -> archive path."""
        ref = ref.replace('/', '\\')
        if not ref.lower().startswith('models\\'):
            ref = 'models\\' + ref
        if not ref.lower().endswith('.smb'):
            ref = os.path.splitext(ref)[0] + '.smb'
        return ref

    def material_ref_path(self, ref):
        ref = ref.replace('/', '\\')
        if not ref.lower().startswith('materials\\'):
            ref = 'materials\\' + ref
        if not ref.lower().endswith('.mtb'):
            ref += '.mtb'
        return ref

    def read_model(self, ref):
        from . import meshes
        blob = self.read(self.model_ref_path(ref))
        return meshes.load_model(blob, ref) if blob else None

    def read_texture(self, ref, max_size=None):
        from . import textures
        blob = self.read(ref)
        return textures.decode(blob, max_size) if blob else None

    def read_material(self, ref):
        blob = self.read(self.material_ref_path(ref))
        return parse_material(blob) if blob else None

    def material_info(self, ref):
        cached = self.materials(block=False)
        key = _key(ref)
        if cached and key in cached:
            return cached[key]
        rec = self.read_material(ref)
        return material_info(ref, rec) if rec else None

    def read_set(self, stem):
        """The parsed .bst dict (gbtvgr.sets.bst.parse), or None."""
        blob = self.read('sets\\%s.bst' % stem)
        return bst.parse(blob) if blob else None

    def read_set_data(self, stem):
        """The set as a SetData (numpy views over sections/lights/probes), or None."""
        from . import setdata
        blob = self.read('sets\\%s.bst' % stem)
        return setdata.load_set(blob, stem) if blob else None

    def read_level(self, stem):
        return self.read('world\\%s.lvl' % stem)

    def read_section(self, name):
        return self.read('world\\%s.sec' % name)

    def set_stems(self):
        return [os.path.splitext(n.split('\\')[-1])[0] for n in self.names('sets')]

    def level_stems(self):
        return [os.path.splitext(n.split('\\')[-1])[0] for n in self.names('levels')]

    # -- the material scan ----------------------------------------------------------
    def _cache_key(self):
        parts = []
        for p in self.pods:
            st = os.stat(p.path)
            parts.append('%s:%d:%d' % (os.path.basename(p.path).lower(), st.st_size,
                                       int(st.st_mtime)))
        # a mod's loose materials are few; their names and sizes join the key
        for e in self.map.values():
            if e.pod is None and e.ext == '.mtb':
                parts.append('%s:%d' % (_key(e.name), e.size))
        return '|'.join(parts)

    def _cache_path(self):
        return os.path.join(self.cache_dir, 'materials.json') if self.cache_dir else None

    def materials(self, block=True, progress=None):
        """{material ref (lower): MaterialInfo} over every shipped .mtb. The first
        scan reads 3000 archives entries and is cached on disk afterwards."""
        with self._mat_lock:
            if self._materials is not None:
                return self._materials
            if not block:
                return None
            cache = self._cache_path()
            key = self._cache_key()
            if cache:
                try:
                    with open(cache, 'r', encoding='utf-8') as fh:
                        d = json.load(fh)
                    if d.get('key') == key:
                        self._materials = {k: MaterialInfo.from_dict(v['name'], v)
                                           for k, v in d['materials'].items()}
                        return self._materials
                except (OSError, ValueError, KeyError):
                    pass
            out = {}
            names = self.names('materials')
            t0 = time.time()
            for i, n in enumerate(names):
                blob = self.read(n)
                if not blob:
                    continue
                ref = n[len('materials\\'):-len('.mtb')]
                try:
                    out[_key(ref)] = material_info(ref, blob)
                except (ValueError, KeyError, IndexError):
                    continue
                if progress and i % 100 == 0:
                    progress(i, len(names))
            self._materials = out
            if cache:
                try:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    with open(cache, 'w', encoding='utf-8') as fh:
                        json.dump({'key': key, 'seconds': time.time() - t0,
                                   'materials': {k: dict(v.to_dict(), name=v.name)
                                                 for k, v in out.items()}}, fh)
                except OSError:
                    pass
            return out

    def scan_materials_async(self, done=None):
        def run():
            m = self.materials(block=True)
            if done:
                done(m)
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t

    def set_materials(self):
        return sorted(k for k, v in self.materials().items() if v.set_ok)

    def model_materials(self):
        return sorted(k for k, v in self.materials().items() if v.model_ok)

    # -- extraction -------------------------------------------------------------
    def extract(self, out_dir, names, progress=None):
        """Write the named assets as loose files under out_dir, game separators kept."""
        n = 0
        for i, name in enumerate(names):
            blob = self.read(name)
            if blob is None:
                continue
            dst = os.path.join(out_dir, *name.replace('/', '\\').split('\\'))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, 'wb') as fh:
                fh.write(blob)
            n += 1
            if progress and i % 50 == 0:
                progress(i, len(names))
        return n


def open_library(game_dir=None, extra_dirs=(), cache_dir=None):
    game_dir = game_dir or default_game_dir()
    if not game_dir:
        raise FileNotFoundError('no game directory: set $GBTVGR_GAME or pass --game')
    return Library(game_dir, extra_dirs, cache_dir).open()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description='The mounted POD chain: list, read and extract by archive path.')
    ap.add_argument('--game', help='game directory (default: $GBTVGR_GAME, $GAME_DIR, Steam)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('mounts', help='the archives in mount order')
    p = sub.add_parser('names', help='archive paths, filtered')
    p.add_argument('--category', choices=sorted(CATEGORIES))
    p.add_argument('--prefix')
    p.add_argument('--ext')
    p.add_argument('--contains')
    p = sub.add_parser('extract', help='write assets as loose files, game separators kept')
    p.add_argument('-o', '--out', required=True)
    p.add_argument('names', nargs='+')
    a = ap.parse_args(argv)
    lib = open_library(a.game)
    try:
        if a.cmd == 'mounts':
            for line in lib.mount_log:
                print(line)
        elif a.cmd == 'names':
            for n in lib.names(a.category, a.ext, a.prefix, a.contains):
                print(n)
        elif a.cmd == 'extract':
            n = lib.extract(a.out, a.names)
            print('%d written to %s' % (n, a.out))
            return 0 if n == len(a.names) else 1
    finally:
        lib.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())

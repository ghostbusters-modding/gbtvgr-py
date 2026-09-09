"""Name -> bytes across the mounted archives, with extracted directories on top."""
import os
import zlib

from .pod import Pod


# PATCH first: it is what the game itself loads first, so a modded texture wins
POD_ORDER = ('PATCH.POD', 'W64ART02.POD', 'W64ART.POD', 'W64MODEL.POD',
             'COMMON.POD')

class ArtIndex(object):
    """Name -> bytes over the game's archives, extracted dirs winning.
    Only the indices are read, so nothing is ever unpacked to disk."""

    def __init__(self, gamedir=None, dirs=()):
        self.pods = []
        self.map = {}
        self.dirs = [d for d in dirs if d and os.path.isdir(d)]
        if gamedir and os.path.isdir(gamedir):
            for name in POD_ORDER:
                path = os.path.join(gamedir, name)
                if not os.path.isfile(path):
                    continue
                try:
                    p = Pod(path)
                except (OSError, ValueError):
                    continue
                pi = len(self.pods)
                self.pods.append(p)
                for e in p.entries:
                    self.map.setdefault(e['name'].lower().replace('/', '\\'),
                                        (pi, e))

    def __bool__(self):
        return bool(self.pods or self.dirs)

    __nonzero__ = __bool__

    def close(self):
        for p in self.pods:
            p.close()
        self.pods = []

    def read(self, name):
        """name like 'art\\graveyard\\wall_diff.tex'; None when absent."""
        key = name.lower().replace('/', '\\')
        for d in self.dirs:
            path = os.path.join(d, *key.split('\\'))
            if os.path.isfile(path):
                with open(path, 'rb') as fh:
                    return fh.read()
        hit = self.map.get(key)
        if hit is None:
            return None
        pi, e = hit
        try:
            return self.pods[pi].read(e)
        except (OSError, ValueError, zlib.error):
            return None

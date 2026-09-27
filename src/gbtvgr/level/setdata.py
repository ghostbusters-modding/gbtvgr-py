"""`.bst` sets decoded for the viewport and for use as templates.

Section geometry decodes on demand: the biggest shipped set carries 140k
meshes, and the outliner should not pay for them before a section is shown."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import struct

import numpy as np

from gbtvgr.mesh.bvt import bvt_collect
from gbtvgr.sets import bst, geom

from .mathutil import F32
from . import meshes as mlib
from .materials import cstr

INSTANCED = 0x20000


def _slot_desc(raw):
    """One 100-byte STexSlot -> (name, fmt, w, h); name '' when empty."""
    if raw[8] == 0:
        return ('', 0, 0, 0)
    name = cstr(raw[8:8 + 0x40])
    fmt, w, h = struct.unpack_from('<3I', raw, 0x48)
    return (name, fmt, w, h)


class SetMesh:
    __slots__ = ('index', 'name', 'material', 'flags', 'geom_type', 'model_index',
                 'positions', 'normals', 'uv0', 'uv1', 'lmuv', 'colors', 'indices',
                 'bbox', 'raw')

    def __init__(self, index, raw):
        self.index = index
        self.raw = raw
        self.name = cstr(raw['name'])
        self.material = struct.unpack('<H', raw['h70'])[0]
        self.flags = raw['flags']
        self.geom_type = struct.unpack('<I', raw['f6c'])[0]
        self.model_index = struct.unpack('<I', raw['f90'])[0] if 'f90' in raw else None
        self.positions = self.normals = self.uv0 = self.uv1 = self.lmuv = None
        self.colors = {}
        self.indices = None
        self.bbox = None
        if 'pkt' in raw:
            bb = struct.unpack('<6f', raw['pkt']['bbox'])
            self.bbox = (np.array(bb[:3], F32), np.array(bb[3:], F32))
            attrs = mlib.decode_packet(raw['pkt'])
            vec = attrs.get('vec', {})
            uv = attrs.get('uv', {})
            self.positions = vec.get(0)
            self.normals = vec.get(1)
            self.uv0 = uv.get(2)
            self.uv1 = uv.get(3)
            self.lmuv = uv.get(12)
            self.colors = attrs.get('color', {})
            self.indices = attrs['indices']

    @property
    def instanced(self):
        return bool(self.flags & INSTANCED)

    def lit_colors(self):
        """The per-vertex colour the game would show unlit: rnm0 when baked, else col0."""
        for slot in (19, 11):
            if slot in self.colors:
                return self.colors[slot]
        return None


class SetSection:
    def __init__(self, index, raw):
        self.index = index
        self.raw = raw
        self.name = cstr(raw['name'])
        bb = struct.unpack('<6f', raw['bbox'])
        self.bbox = (np.array(bb[:3], F32), np.array(bb[3:], F32))
        self.light_indices = list(np.frombuffer(raw['arr818'], dtype='<u4'))
        self.probe_indices = list(np.frombuffer(raw['portidx'], dtype='<u4'))
        self.lightmaps = [_slot_desc(raw['names3'][i * 100:(i + 1) * 100]) for i in range(3)]
        self.surfaces = []
        for l in raw['lrefs']:
            a, mask = struct.unpack_from('<2I', l['v'], 0)
            sb = struct.unpack_from('<6f', l['v'], 8)
            self.surfaces.append((cstr(l['name']), a, mask, sb))
        self.mesh_count = len(raw['meshes'])
        self._meshes = None
        self._collision = None

    @property
    def meshes(self):
        if self._meshes is None:
            self._meshes = [SetMesh(i, m) for i, m in enumerate(self.raw['meshes'])]
        return self._meshes

    @property
    def loaded(self):
        return self._meshes is not None

    @property
    def collision(self):
        """(verts, tris, surface index, flags) numpy arrays, or empty ones."""
        if self._collision is None:
            if self.raw.get('bvt') is None:
                self._collision = (np.zeros((0, 3), F32), np.zeros((0, 3), np.int32),
                                   np.zeros(0, np.int32), np.zeros(0, np.int32))
            else:
                self._collision = mlib.hull_from_bvt(self.raw['bvt'])
        return self._collision

    @property
    def arena(self):
        return self.raw['bvtflag']

    @property
    def lit(self):
        return bool(self.lightmaps[0][0])

    def stats(self):
        return geom.section_stats(self.raw)


class SetLight:
    __slots__ = ('index', 'name', 'pos', 'color', 'radius', 'sections', 'raw', 'params')

    def __init__(self, index, raw):
        self.index = index
        self.raw = raw
        self.name = cstr(raw['a'])
        self.pos = np.array(struct.unpack('<3f', raw['pos']), F32)
        self.params = list(struct.unpack('<12I', raw['scal']))
        c = self.params[2]
        self.color = ((c >> 16) & 255, (c >> 8) & 255, c & 255)
        self.radius = struct.unpack('<f', struct.pack('<I', self.params[4]))[0]
        self.sections = list(np.frombuffer(raw['idx'], dtype='<u4'))


class SetProbe:
    __slots__ = ('index', 'pos', 'aux', 'direction', 'ambient', 'env', 'raw')

    def __init__(self, index, raw):
        self.index = index
        self.raw = raw
        self.pos = np.array(struct.unpack_from('<3f', raw, 0), F32)
        self.aux = struct.unpack_from('<3f', raw, 12)
        self.direction = np.array(struct.unpack_from('<3f', raw, 24), F32)
        self.ambient = _slot_desc(raw[0x24:0x24 + 100])
        self.env = _slot_desc(raw[0x88:0x88 + 100])


class SetData:
    """A parsed set with numpy views. `raw` is the codec's dict, captured so a
    relayout can write it back unchanged."""

    def __init__(self, name, raw):
        self.name = name
        self.raw = geom.capture(raw)
        self.materials = [geom.material_name(e, i) for i, e in enumerate(raw['materials'])]
        self.sections = [SetSection(i, s) for i, s in enumerate(raw['sections'])]
        self.lights = [SetLight(i, l) for i, l in enumerate(raw['lights'])]
        self.probes = [SetProbe(i, p) for i, p in enumerate(raw['portals'])]
        self.fog_a = struct.unpack('<3f', raw['fogA'])
        self.fog_b = struct.unpack('<3f', raw['fogB'])
        self.sun = struct.unpack('<3f', raw['sky']['fogvec'])
        self.fog_rgb = struct.unpack('<3I', raw['sky']['fog3'])

    @property
    def sky_layers(self):
        return [l for l in self.raw['sky']['layers'] if l['type']]

    @property
    def bounds(self):
        los = [s.bbox[0] for s in self.sections]
        his = [s.bbox[1] for s in self.sections]
        if not los:
            return np.zeros(3, F32), np.zeros(3, F32)
        return np.min(los, axis=0), np.max(his, axis=0)

    def nav(self):
        verts, polys = geom.nav_polys(self.raw['nav'])
        return np.array(verts, F32).reshape(-1, 3), polys

    def bsp(self):
        return geom.bsp_nodes(self.raw)

    def material_ref(self, index):
        e = self.raw['materials'][index]
        return cstr(e['ref'][0]) if e['ref'][0] else None

    def total_meshes(self):
        return sum(s.mesh_count for s in self.sections)

    def section_by_name(self, name):
        for s in self.sections:
            if s.name.lower() == name.lower():
                return s
        return None


def load_set(blob, name=''):
    return SetData(name, bst.parse(blob))


def rebuild(setdata, tight=False):
    """Write the set back: relayout the captured dict, byte-identical when untouched."""
    return bst.build(geom.relayout(setdata.raw, tight=tight))

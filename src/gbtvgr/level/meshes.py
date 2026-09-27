"""`.smb` models as numpy arrays, for the viewport and for component authoring."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import struct

import numpy as np

from gbtvgr.mesh import smb
from gbtvgr.mesh.bvt import bvt_collect
from gbtvgr.mesh.smb import decl_stride

from .mathutil import F32

NOSLOT = 0xFFFFFFFF
S_POS, S_NORMAL, S_UV, S_UV1, S_COLOR, S_LMUV = 0, 1, 2, 3, 11, 12
RNM_SLOTS = (19, 20, 21)


def cstr(raw):
    e = raw.find(b'\0')
    return raw[:e if e >= 0 else len(raw)].decode('latin1')


def decl_map(decl):
    return {i: (o, t) for i, (o, t, _u) in enumerate(decl) if o != NOSLOT}


def view_f3(vdata, n, stride, off):
    a = np.ndarray((n, 3), dtype='<f4', buffer=vdata, offset=off, strides=(stride, 4))
    out = np.array(a, dtype=F32)
    out[~np.isfinite(out)] = 0.0
    return out


def view_half2(vdata, n, stride, off):
    a = np.ndarray((n, 2), dtype='<f2', buffer=vdata, offset=off, strides=(stride, 2))
    out = np.array(a, dtype=F32)
    out[~np.isfinite(out)] = 0.0
    return out


def view_color(vdata, n, stride, off):
    """Packed D3DCOLOR (B, G, R, A in memory) -> (n, 4) RGBA uint8."""
    a = np.ndarray((n, 4), dtype=np.uint8, buffer=vdata, offset=off, strides=(stride, 1))
    return np.array(a[:, [2, 1, 0, 3]], dtype=np.uint8)


def decode_packet(pkt):
    """A render packet dict -> dict of numpy attribute arrays."""
    decl = pkt['decl']
    n = pkt['nverts']
    st = decl_stride(decl)
    vd = pkt['vdata']
    sl = decl_map(decl)
    out = {'nverts': n, 'stride': st}
    for slot, (off, t) in sl.items():
        if t == 3:
            out.setdefault('vec', {})[slot] = view_f3(vd, n, st, off)
        elif t == 5:
            out.setdefault('uv', {})[slot] = view_half2(vd, n, st, off)
        elif t == 9:
            out.setdefault('color', {})[slot] = view_color(vd, n, st, off)
    idx = np.frombuffer(pkt['idata'], dtype='<u2', count=pkt['nprims'] * 3)
    out['indices'] = np.array(idx, dtype=np.uint32).reshape(-1, 3)
    return out


class Part:
    __slots__ = ('name', 'mat_index', 'positions', 'normals', 'uvs', 'colors',
                 'indices', 'bbox')

    def __init__(self, name, mat_index, attrs, bbox):
        self.name = name
        self.mat_index = mat_index
        vec = attrs.get('vec', {})
        self.positions = vec.get(S_POS, np.zeros((attrs['nverts'], 3), F32))
        self.normals = vec.get(S_NORMAL)
        self.uvs = attrs.get('uv', {}).get(S_UV)
        self.colors = attrs.get('color', {}).get(S_COLOR)
        self.indices = attrs['indices']
        self.bbox = bbox


class Hull:
    __slots__ = ('name', 'verts', 'tris', 'flags', 'arena')

    def __init__(self, name, verts, tris, flags, arena):
        self.name = name
        self.verts = verts
        self.tris = tris
        self.flags = flags
        self.arena = arena


class ModelData:
    def __init__(self, name, raw):
        self.name = name
        self.raw = raw
        self.materials = [cstr(e['ref']) or 'embedded:%d' % i
                          for i, e in enumerate(raw['materials'])]
        self.embedded = [e['embedded'] for e in raw['materials']]
        self.parts = []
        for p in raw['parts']:
            attrs = decode_packet(p)
            bb = struct.unpack('<6f', p['bbox'])
            self.parts.append(Part(cstr(p['name']), p['mat_idx'], attrs,
                                   (np.array(bb[:3], F32), np.array(bb[3:], F32))))
        self.hulls = []
        for c in raw['collisions']:
            nv, nt = len(c['verts']) // 12, len(c['tris']) // 6
            verts = np.frombuffer(c['verts'], dtype='<f4', count=nv * 3).reshape(-1, 3).astype(F32)
            tris = np.frombuffer(c['tris'], dtype='<u2', count=nt * 3).reshape(-1, 3).astype(np.int32)
            flags = np.frombuffer(c['triflags'], dtype=np.uint8, count=nt).copy()
            self.hulls.append(Hull(cstr(c['name']), verts, tris, flags, c.get('bvt_arena', 0)))
        bb = struct.unpack('<6f', raw['bbox'])
        self.bbox = (np.array(bb[:3], F32), np.array(bb[3:], F32))
        self.lod = raw['lod']

    @property
    def collision_part_count(self):
        return len(self.hulls)

    @property
    def vertex_count(self):
        return sum(len(p.positions) for p in self.parts)

    @property
    def triangle_count(self):
        return sum(len(p.indices) for p in self.parts)

    def slot_names(self):
        """The material-slot contract a component binds to."""
        return list(self.materials)

    def collision_soup(self):
        """Every hull as one (verts, tris) pair in model space."""
        vs, ts, base = [], [], 0
        for h in self.hulls:
            vs.append(h.verts)
            ts.append(h.tris + base)
            base += len(h.verts)
        if not vs:
            return np.zeros((0, 3), F32), np.zeros((0, 3), np.int32)
        return np.concatenate(vs), np.concatenate(ts)


def load_model(blob, name=''):
    return ModelData(name, smb.parse(blob))


def hull_from_bvt(node):
    verts, tris, surf, tflags = bvt_collect(node)
    return (np.array(verts, F32).reshape(-1, 3), np.array(tris, np.int32).reshape(-1, 3),
            np.array(surf, np.int32), np.array(tflags, np.int32))

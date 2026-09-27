"""`.mtb` materials as the editor sees them: a name, a geometry mask, a layer
list, and the two surgical edits the format allows (retexture a layer, embed
the record in a mesh)."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import os
import struct

from gbtvgr.mesh.mtb import read_material, write_material
from gbtvgr.wire import R, W

MODEL_BIT = 0x1       # a model loads the material only with this bit set
SET_BIT = 0x4         # set geometry needs this one instead

LAYER_TYPES = {0: 'diffuse', 1: 'bump', 6: 'spec', 10: 'glow', 12: 'diffuse2',
               13: 'bump2', 20: 'spec2', 21: 'diffuse3', 22: 'bump3', 23: 'spec3'}


def cstr(raw):
    e = raw.find(b'\0')
    return raw[:e if e >= 0 else len(raw)].decode('latin1')


def tex_path(name):
    """A layer's texture name -> its archive path (`art\\` prefix, `.tex`)."""
    name = name.replace('/', '\\').lstrip('\\')
    if name.lower().startswith('art\\'):
        name = name[4:]
    return 'art\\' + os.path.splitext(name)[0] + '.tex'


class MaterialInfo:
    __slots__ = ('name', 'mask', 'layers', 'flags')

    def __init__(self, name, mask, layers, flags=0):
        self.name = name
        self.mask = mask
        self.layers = layers          # [(type, texture name)] in layer order
        self.flags = flags

    @property
    def diffuse(self):
        """Best-guess colour layer: type alone does not distinguish diff from
        bump/spec (both are often type 0), so the name decides instead."""
        if not self.layers:
            return None
        bad = ('_bump', '_spec', '_norm', '_glow', '_nrm')

        def score(entry):
            base = os.path.splitext(entry[1])[0].lower()
            return 0 if base.endswith('_diff') else 2 if base.endswith(bad) else 1
        return min(self.layers, key=score)[1]

    @property
    def set_ok(self):
        return bool(self.mask & SET_BIT)

    @property
    def model_ok(self):
        return bool(self.mask & MODEL_BIT)

    def layer_of_type(self, typ):
        for i, (t, _n) in enumerate(self.layers):
            if t == typ:
                return i
        return None

    def to_dict(self):
        return {'mask': self.mask, 'flags': self.flags, 'layers': self.layers}

    @classmethod
    def from_dict(cls, name, d):
        return cls(name, d['mask'], [tuple(x) for x in d['layers']], d.get('flags', 0))


def parse(blob):
    return read_material(R(blob))


def build(record, base_offset=0):
    """Record -> bytes. The dependency pad aligns to the absolute file offset,
    so it is recomputed for where the record will land."""
    pos = base_offset + 4 + 16
    for name, _h in record['deps']:
        pos += len(name) + 1 + 16
    pos += 1
    record['deppad'] = b'\0' * ((-pos) % 4)
    w = W()
    write_material(w, record)
    return w.data()


def info(name, blob_or_record):
    rec = blob_or_record if isinstance(blob_or_record, dict) else parse(blob_or_record)
    mask = struct.unpack('<I', rec['fc'])[0]
    flags = struct.unpack('<I', rec['f10'])[0]
    layers = []
    for l in rec['layers']:
        n = cstr(l['name'])
        if n:
            layers.append((struct.unpack('<I', l['fac'])[0], n))
    return MaterialInfo(name, mask, layers, flags)


def _same_texture(a, b):
    return os.path.splitext(tex_path(a))[0].lower() == os.path.splitext(tex_path(b))[0].lower()


def retexture(record, layer_index, new_name):
    """A copy of the record with one layer's 64-byte texture path rewritten and
    its dependency line moved with it. The 16-byte hash is never validated."""
    import copy
    rec = copy.deepcopy(record)
    layer = rec['layers'][layer_index]
    old = cstr(layer['name'])
    name = new_name.replace('/', '\\').encode('latin1')
    if not name.lower().endswith(b'.tga'):
        name = os.path.splitext(name)[0] + b'.tga'
    if len(name) >= 0x40:
        raise ValueError('texture path too long for a material layer: %r' % new_name)
    layer['name'] = name + b'\0' * (0x40 - len(name))
    dep_name = b'art\\' + name if not name.lower().startswith(b'art\\') else name
    deps, hit = [], False
    for dn, dh in rec['deps']:
        if not hit and _same_texture(dn.decode('latin1'), old):
            deps.append((dep_name, dh))
            hit = True
        else:
            deps.append((dn, dh))
    if not hit:
        deps.append((dep_name, b'\0' * 16))
    rec['deps'] = deps
    return rec


def clone_name(base, suffix):
    """A derived material's archive name, kept short enough for a .smb ref."""
    base = base.replace('/', '\\')
    return '%s_%s' % (base, suffix)

"""The skybox block: composed from a donor set's layers, with the texture and
material of chosen layers rewritten. Layer bodies are fixed-size and only the
common prologue is resolved, so every extra byte is carried from the donor.
"""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import struct

import numpy as np

from gbtvgr.sets import bst as gbst

NAME_LEN = 0x40
MAT_LEN = 0x100
PROLOGUE = 0x148
LAYER_TYPES = {1: 'dome', 2: 'billboard', 3: 'billboard2', 4: 'starfield', 5: 'model'}


def layer_texture(layer):
    raw = layer['raw']
    return raw[:NAME_LEN].split(b'\0')[0].decode('latin1')


def layer_material(layer):
    raw = layer['raw']
    return raw[NAME_LEN:NAME_LEN + MAT_LEN].split(b'\0')[0].decode('latin1')


def layer_model(layer):
    """Dome and model layers name a model after the prologue."""
    raw = layer['raw']
    if layer['type'] == 1:
        off = PROLOGUE + 0xC
    elif layer['type'] == 5:
        off = PROLOGUE
    else:
        return ''
    return raw[off:off + 0x80].split(b'\0')[0].decode('latin1')


def describe(layer):
    return {'type': layer['type'], 'kind': LAYER_TYPES.get(layer['type'], '?'),
            'texture': layer_texture(layer), 'material': layer_material(layer),
            'model': layer_model(layer)}


def _fixed(raw, off, width, text):
    b = text.replace('/', '\\').encode('latin1')
    if len(b) >= width:
        raise ValueError('name too long for a %d-byte field: %r' % (width, text))
    return raw[:off] + b + b'\0' + raw[off + len(b) + 1:off + width] + raw[off + width:]


def rewrite_layer(layer, texture=None, material=None, model=None):
    raw = layer['raw']
    if texture is not None:
        raw = _fixed(raw, 0, NAME_LEN, texture)
    if material is not None:
        raw = _fixed(raw, NAME_LEN, MAT_LEN, material)
    if model is not None and layer['type'] in (1, 5):
        off = PROLOGUE + (0xC if layer['type'] == 1 else 0)
        raw = _fixed(raw, off, 0x80, model)
    assert len(raw) == gbst.SKY_SIZES[layer['type']]
    return {'type': layer['type'], 'raw': raw}


def empty_sky(sun=(-0.57735, -0.57735, -0.57735), fog_rgb=(0, 0, 0)):
    return {'layers': [], 'fogvec': struct.pack('<3f', *sun),
            'fog3': struct.pack('<3I', *fog_rgb), 'order': b'\0' * 200}


def build_sky(skynode, donor_raw=None):
    """skynode.props: donor (set stem the caller resolved to donor_raw), layers as
    [{index, enabled, texture, material, model}], sun, fog_rgb."""
    p = skynode.props
    sun = np.array(p.get('sun', (-0.57735, -0.57735, -0.57735)), dtype=np.float64)
    n = np.linalg.norm(sun)
    sun = sun / n if n > 1e-9 else np.array([-0.57735, -0.57735, -0.57735])
    fog_rgb = [int(v) for v in p.get('fog_rgb', (0, 0, 0))]
    if donor_raw is None:
        return empty_sky(sun, fog_rgb)
    donor = donor_raw['sky']
    layers = [dict(l) for l in donor['layers']]
    edits = {int(e.get('index', -1)): e for e in p.get('layers', [])}
    out = []
    for i, l in enumerate(layers):
        e = edits.get(i)
        if e is not None and not e.get('enabled', True):
            out.append({'type': 0})
            continue
        if e is None or not l['type']:
            out.append(l)
            continue
        out.append(rewrite_layer(l, e.get('texture'), e.get('material'), e.get('model')))
    return {'layers': out, 'fogvec': struct.pack('<3f', *sun),
            'fog3': struct.pack('<3I', *fog_rgb), 'order': donor['order']}


def donor_layers(donor_raw):
    return [dict(describe(l), index=i) for i, l in enumerate(donor_raw['sky']['layers']) if l['type']]

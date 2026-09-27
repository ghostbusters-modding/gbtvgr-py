"""Light and light-probe records for an authored set.

Only colour (word 2) and radius (word 4) of a light's 12 parameter words are
resolved; the rest is abyss Omni1's block, the configuration proven in-game.
"""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import struct

import numpy as np

LIGHT_PARAMS = (0x41E00000, 1, 0x00AED5FB, 0, 0x43A3EF9B, 1, 1, 0, 0, 1, 1, 0x3F800000)
OMNI_FALLOFF = b'lights\\default_omni_falloff.tga'
DEFAULT_SPOT = b'lights\\default_spot.tga'
PROBE_AMBIENT_DESC = (24, 16, 16)
PROBE_ENV_DESC = (24, 64, 64)


def tex_slot_empty():
    """The shipped "no lightmap" STexSlot: a 0x00 at slot+8, never all 0xCD."""
    return b'\xCD' * 8 + b'\x00' + b'\xCD' * 91


def tex_slot(path, fmt, w, h, filetag=b'\0' * 8):
    name = path.encode('latin1') + b'\0'
    if len(name) > 0x40:
        raise ValueError('texture path too long for an STexSlot: %r' % path)
    s = b'\xCD' * 8 + name + b'\xCD' * (0x40 - len(name))
    s += struct.pack('<5I', fmt, w, h, 0, 0) + filetag
    assert len(s) == 100
    return s


def sphere_hits_box(pos, radius, lo, hi):
    p = np.asarray(pos, dtype=np.float64)
    d = np.maximum(np.maximum(np.asarray(lo) - p, 0.0), p - np.asarray(hi))
    return float(np.sqrt((d * d).sum())) < radius


def light_sections(light_nodes, sections):
    """Per light: the sections its radius reaches (at least one), and per section:
    the lights that reach it."""
    per_light, per_sec = [], [[] for _ in sections]
    for li, l in enumerate(light_nodes):
        pos = l.world_pos()
        hit = [si for si, s in enumerate(sections) if sphere_hits_box(pos, l.radius, s.lo, s.hi)]
        if not hit and sections:
            c = [np.linalg.norm(pos - s.center()) for s in sections]
            hit = [int(np.argmin(c))]
        per_light.append(hit)
        for si in hit:
            per_sec[si].append(li)
    return per_light, per_sec


def light_record(node, section_indices):
    r, g, b = node.color
    par = list(LIGHT_PARAMS)
    par[2] = (r << 16) | (g << 8) | b
    par[4] = struct.unpack('<I', struct.pack('<f', node.radius))[0]
    name = node.name.encode('latin1')[:0x1F]
    pos = node.world_pos()
    return {
        'a': name + b'\0' + b'\xCD' * (0x20 - len(name) - 1),
        'pos': struct.pack('<3f', *pos),
        'b': struct.pack('<3f', 0.0, 0.0, 0.0),
        'scal': struct.pack('<12I', *par),
        'vec': struct.pack('<3f', node.radius, node.radius, node.radius),
        'f48': struct.pack('<f', 1.0),
        'idx': b''.join(struct.pack('<I', s) for s in sorted(set(section_indices))),
        'name1': OMNI_FALLOFF.ljust(0x40, b'\0'),
        'name2': DEFAULT_SPOT.ljust(0x40, b'\0'),
        'name3': b'\0' * 0x40,
    }


def probe_record(node, index, set_name):
    """One 0xEC record naming lightprobe\\<set>\\{ambient,env}_<index>.tga."""
    pos = node.world_pos()
    d = np.array(node.props.get('direction', (0.0, -1.0, 0.0)), dtype=np.float64)
    n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else np.array([0.0, -1.0, 0.0])
    rec = struct.pack('<3f', *pos) + struct.pack('<3f', 0.0, 0.0, 0.0) + struct.pack('<3f', *d)
    rec += tex_slot('lightprobe\\%s\\ambient_%d.tga' % (set_name, index), *PROBE_AMBIENT_DESC)
    rec += tex_slot('lightprobe\\%s\\env_%d.tga' % (set_name, index), *PROBE_ENV_DESC)
    assert len(rec) == 0xEC
    return rec


def probe_sections(probe_nodes, sections):
    """A probe belongs to the section holding it, else the nearest one."""
    per_sec = [[] for _ in sections]
    for pi, p in enumerate(probe_nodes):
        pos = p.world_pos()
        hit = [si for si, s in enumerate(sections) if s.contains(pos)]
        if not hit and sections:
            c = [np.linalg.norm(pos - s.center()) for s in sections]
            hit = [int(np.argmin(c))]
        for si in hit:
            per_sec[si].append(pi)
    return per_sec


def probe_deps(count, set_name):
    deps = []
    for i in range(count):
        deps.append((('art\\lightprobe\\%s\\ambient_%d.tga' % (set_name, i)).encode('latin1'), b'\0' * 16))
        deps.append((('art\\lightprobe\\%s\\env_%d.tga' % (set_name, i)).encode('latin1'), b'\0' * 16))
    return deps

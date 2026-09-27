"""A seed set: one shipped section carried out of its donor as a whole set, so a graft
can grow from a real room instead of a borrowed host."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import copy
import struct

from ..sets import geom
from . import transplant as tp

HEADER_KEYS = ('lightver', 'hash', 'deppad', 'datasize', 'boolA', 'fogA', 'fogB', 'boolB',
               'f2f08', 'f62d0', 'hdr2a', 'vec6310', 'hdr2b', 'f6358', 's6360', 'hdr2c',
               'bspglob')


def _dep_name(pod_path):
    """The build-dep list names the .tga the loader rewrites to .tex."""
    return pod_path[:-4] + '.tga'


def _rename_slots(sec, dirname, staged):
    names3 = bytearray(sec['names3'])
    lit = []
    for k in range(3):
        slot = bytes(names3[k * tp.SLOT:(k + 1) * tp.SLOT])
        old = tp.slot_name(slot)
        if not old:
            continue
        new = 'lightmap\\%s\\0_%d.tga' % (dirname, k)
        names3[k * tp.SLOT:(k + 1) * tp.SLOT] = tp.set_slot_name(slot, new)
        staged.append((tp.pod_path(old), tp.pod_path(new)))
        lit.append(old)
    sec['names3'] = bytes(names3)
    if lit:
        staged.append((tp.pod_path(lit[0].rsplit('_', 1)[0] + '_3.tga'),
                       tp.pod_path('lightmap\\%s\\0_3.tga' % dirname)))


def _probes(donor, sec, dirname, staged):
    out = []
    for n, p in enumerate(tp._u32s(sec['portidx'])):
        rec = bytearray(donor['portals'][p])
        for off, kind in tp.PROBE_SLOTS:
            slot = bytes(rec[off:off + tp.SLOT])
            old = tp.slot_name(slot)
            if not old:
                continue
            new = 'lightprobe\\%s\\%s_%d.tga' % (dirname, kind, n)
            rec[off:off + tp.SLOT] = tp.set_slot_name(slot, new)
            staged.append((tp.pod_path(old), tp.pod_path(new)))
        out.append(bytes(rec))
    sec['portidx'] = tp._pack_u32s(list(range(len(out))))
    return out


def regroup(nav):
    """Component ids from the links as they stand: a donor group cut from its outside
    links can fall into islands, and every shipped set keeps id == component."""
    nbs = [tp.nav_neighbours(d) for d in nav['nodes']]
    comp = [-1] * len(nbs)
    n = 0
    for s in range(len(nbs)):
        if comp[s] >= 0:
            continue
        comp[s] = n
        stack = [s]
        while stack:
            x = stack.pop()
            for j in nbs[x]:
                if j >= 0 and comp[j] < 0:
                    comp[j] = n
                    stack.append(j)
        n += 1
    for d, g in zip(nav['nodes'], comp):
        h = tp.nav_header(d)
        h['group'] = g
        d['hdr'] = tp.pack_nav_header(h)
    nav['f'] = n
    return n


def new_set_from_section(donor_m, donor_index, lightmap_dir, unembed=True):
    """The donor's header, skybox and scalars around one section at index 0, with only
    the materials, probes, lights and nav it uses. Returns (set, staged) like append_section."""
    if 'origdatasize' not in donor_m:
        geom.capture(donor_m)
    sec = copy.deepcopy(donor_m['sections'][donor_index])
    m = {k: copy.deepcopy(donor_m[k]) for k in HEADER_KEYS}
    m['sky'] = copy.deepcopy(donor_m['sky'])
    m['materials'], m['mat16'] = [], b''
    remap = {}
    for i in sorted(set(tp._h70(me) for me in sec['meshes'])):
        remap[i] = len(m['materials'])
        mat = copy.deepcopy(donor_m['materials'][i])
        mat, shipped = (tp.unembed_material(mat, '%s\\emb_0_%d' % (lightmap_dir, i))
                        if unembed else (mat, None))
        m['materials'].append(mat)
        if shipped:
            m.setdefault('mtb_staged', []).append(shipped)
        m['mat16'] += donor_m['mat16'][i * 2:i * 2 + 2]
    for me in sec['meshes']:
        me['h70'] = struct.pack('<H', remap[tp._h70(me)])
    staged = []
    _rename_slots(sec, lightmap_dir, staged)
    m['portals'] = _probes(donor_m, sec, lightmap_dir, staged)
    m['lights'] = []
    for li in tp._u32s(sec['arr818']):
        L = copy.deepcopy(donor_m['lights'][li])
        L['idx'] = tp._pack_u32s([0])
        m['lights'].append(L)
    sec['arr818'] = tp._pack_u32s(list(range(len(m['lights']))))
    sec['arr4f4'] = b''
    sec['arr470'] = b''
    m['sections'] = [sec]
    m['watervis'] = donor_m['watervis'][donor_index:donor_index + 1]
    # The loader skips this list. It is trimmed so the file names only what it ships.
    renamed = {_dep_name(src).encode('latin1'): _dep_name(dst).encode('latin1')
               for src, dst in staged}
    m['deps'] = [(renamed[name], h) for name, h in donor_m['deps'] if name in renamed]
    m['fx'] = b''
    m['breakers'], m['wblobs'], m['wmesh'] = [], [], []
    m['bsp'] = tp._pack_bsp((0.0, 0.0, 0.0, 0.0), 0, -1, -1,
                            struct.unpack('<6f', sec['bbox']))
    m['nav'] = {'empty': True, 'f': 0, 'nverts': 0, 'nnodes': 0, 'nextra': 0, 'nparts': 0}
    if tp.append_nav(m, donor_m, donor_index, (0.0, 0.0, 0.0)):
        regroup(m['nav'])
    else:
        m['nav'] = {'empty': True, 'f': 0, 'nverts': 0, 'nnodes': 0, 'nextra': 0, 'nparts': 0}
    m['origdatasize'] = donor_m['datasize']
    geom.relayout(m)
    return m, staged

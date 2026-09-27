"""Light for generated sections: flat lightmap tiles sampled from the rooms they join,
and a probe listed for every section that names none."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import argparse
import hashlib
import os
import struct
import sys

import numpy as np

from .. import tex as gtex
from ..mesh.bvt import union_bbox
from ..mesh.smb import decl_stride
from ..sets import bst, geom
from . import lights as glights
from . import textures
from . import transplant as tp
from .sections import DECL_LM64, GEOM_LIGHTMAP

AX = {'x': 0, 'y': 1, 'z': 2}
TILE_SIZE = (64, 32)          # the smallest retail tile: a size the engine has drawn
MID_GREY = (128.0, 128.0, 128.0)   # neutral: the lightmap shader doubles what it samples
LM_SLOTS = (0, 1, 2, 5, 8)    # what a mesh must already carry to become lightmapped


def _render_meshes(sec):
    return [me for me in sec['meshes'] if not (me['flags'] & 0x20000)]


def _bbox_centre(sec):
    b = struct.unpack('<6f', sec['bbox'])
    return tuple((b[k] + b[k + 3]) / 2.0 for k in range(3))


def _usable_probes(m):
    """Probes with both cube names set. The loader parks the others at 99999 ft."""
    out = []
    for i, rec in enumerate(m['portals']):
        p = tp.unpack_probe(rec)
        if tp.slot_name(p['ambient']) and tp.slot_name(p['env']):
            out.append(i)
    return out


def _probe_pos(m, i):
    return tp.unpack_probe(m['portals'][i])['pos']


def _nearest(m, candidates, point):
    best = None
    for i in candidates:
        d = sum((a - b) ** 2 for a, b in zip(_probe_pos(m, i), point))
        if best is None or d < best[0]:
            best = (d, i)
    return None if best is None else best[1]


def _read(library, slot_path):
    if library is None or not slot_path:
        return None
    return library.read(tp.pod_path(slot_path))


# --- sampling ------------------------------------------------------------------------
def _tile_mean(blob):
    """Mean over the lit texels only: the empty chart area is pure black."""
    _w, _h, img = textures.decode(blob)
    rgb = img[:, :, :3].reshape(-1, 3).astype(np.float64)
    lit = rgb.max(axis=1) > 0
    return rgb[lit].mean(axis=0) if lit.any() else None


def _cube_mean(blob):
    faces = textures.decode_cube(blob)
    return np.mean([f[:, :, :3].reshape(-1, 3).mean(axis=0) for f in faces], axis=0)


def room_ambient(m, section_index, library):
    """(r, g, b) in 0..255: the section's lightmap tiles, else its probes' ambient
    cubes, else mid grey."""
    s = m['sections'][section_index]
    means = []
    for k in range(3):
        blob = _read(library, tp.slot_name(s['names3'][k * tp.SLOT:(k + 1) * tp.SLOT]))
        if blob:
            mu = _tile_mean(blob)
            if mu is not None:
                means.append(mu)
    if means:
        return tuple(float(v) for v in np.mean(means, axis=0))
    cubes = []
    for p in tp._u32s(s['portidx']):
        blob = _read(library, tp.slot_name(tp.unpack_probe(m['portals'][p])['ambient']))
        if blob:
            cubes.append(_cube_mean(blob))
    if cubes:
        return tuple(float(v) for v in np.mean(cubes, axis=0))
    return MID_GREY


# --- tiles ----------------------------------------------------------------------------
def _pair(colour):
    c = np.asarray(colour, np.float64)
    if c.shape == (3,):
        return c, c
    if c.shape == (2, 3):
        return c[0], c[1]
    raise ValueError('colour must be (r, g, b) or ((r, g, b), (r, g, b))')


def flat_tile(colour, size=TILE_SIZE, name=''):
    """fmt-3 .tex bytes: colour[0] at u=0 running to colour[1] at u=1, flat in v.
    The header hash is unresolved, so a digest of name and pixels stands in."""
    a, b = _pair(colour)
    w, h = size
    t = np.arange(w, dtype=np.float64) / max(w - 1, 1)
    rgb = a[None, :] * (1.0 - t[:, None]) + b[None, :] * t[:, None]
    img = np.empty((h, w, 4), np.uint8)
    img[:, :, :3] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)[None, :, :]
    img[:, :, 3] = 255
    payload = np.ascontiguousarray(img[:, :, [2, 1, 0, 3]]).tobytes()
    digest = hashlib.md5(name.lower().encode('latin1') + payload).digest()
    return gtex.build_header(7, digest, 0, textures.FMT_BGRA8, w, h, 0, 1, 0,
                             gtex.DEFAULT_D) + payload


def _filetag(m):
    """The per-file STexSlot stamp, copied from any populated slot the set carries."""
    for s in m['sections']:
        for k in range(3):
            slot = s['names3'][k * tp.SLOT:(k + 1) * tp.SLOT]
            if tp.slot_name(slot):
                return slot[0x5C:0x64]
    for rec in m['portals']:
        slot = rec[0x24:0x24 + tp.SLOT]
        if tp.slot_name(slot):
            return slot[0x5C:0x64]
    return b'\0' * 8


# --- meshes ---------------------------------------------------------------------------
def _lightmapped(g, lmuv, colours):
    """Geometry dict -> the {11, 12} family with the given lightmap uvs and colours."""
    sl = geom.decl_slots(g['decl'])
    if 12 in sl:
        g['uv'][12] = lmuv
        if 11 in sl:
            g['color'][11] = colours
        return geom.encode_mesh(g)
    missing = [s for s in LM_SLOTS if s not in sl]
    if missing:
        raise ValueError('mesh lacks slots %r needed for the lightmapped layout' % missing)
    out = {'decl': DECL_LM64, 'nverts': g['nverts'], 'tris': g['tris'],
           'vec': {k: g['vec'][k] for k in (0, 1, 5, 8)},
           'vecraw': {k: v for k, v in g['vecraw'].items() if k in (0, 1, 5, 8)},
           'uv': {2: g['uv'][2], 3: g['uv'].get(3, g['uv'][2]), 12: lmuv},
           'uvraw': {k: v for k, v in g['uvraw'].items() if k in (2, 3)},
           'color': {11: colours}, 'raw': {}, 'pad': None,
           'f18': g['f18'], 'morphs': g['morphs'], 'mdata': g['mdata']}
    if 3 not in sl and 2 in g['uvraw']:
        out['uvraw'][3] = g['uvraw'][2]
    return geom.encode_mesh(out)


def _mesh_bbox(meshes):
    return union_bbox([me['pkt']['bbox'] for me in meshes])


def light_section(m, section_index, colour, out_dir, lightmap_dir, axis=None, size=TILE_SIZE):
    """Three flat tiles named in names3, meshes made lightmapped with uvs 0..1 along `axis`.
    A mesh born lightmapped (the loft's (t, i/n)) keeps its uvs and is graded along its u.
    colour (r, g, b) or a (low, high) pair grades along it. Returns [(tile bytes, pod path)]."""
    if 'origdatasize' not in m:
        geom.capture(m)
    s = m['sections'][section_index]
    meshes = _render_meshes(s)
    if not meshes:
        raise ValueError('section %d has no render mesh to light' % section_index)
    mb = _mesh_bbox(meshes)
    if axis is None:
        axis = 'x' if mb[3] - mb[0] >= mb[5] - mb[2] else 'z'
    ax = AX[axis]
    ca = 2 if ax == 0 else 0
    span = max(mb[ax + 3] - mb[ax], 1e-6)
    cross = max(mb[ca + 3] - mb[ca], 1e-6)
    a, b = _pair(colour)
    for me in meshes:
        g = geom.decode_mesh(me['pkt'])
        if 12 in g['uv']:
            lmuv = [(float(u), float(w)) for u, w in g['uv'][12]]
            t = np.clip(np.array([u for u, _w in lmuv], np.float64), 0.0, 1.0)
        else:
            pos = np.asarray(g['vec'][0], np.float64)
            t = np.clip((pos[:, ax] - mb[ax]) / span, 0.0, 1.0)
            v = np.clip((pos[:, ca] - mb[ca]) / cross, 0.0, 1.0)
            lmuv = [(float(u), float(w)) for u, w in zip(t, v)]
        rgb = np.clip(np.round(a[None, :] * (1.0 - t[:, None]) + b[None, :] * t[:, None]),
                      0, 255).astype(int)
        # vertex colours sit on disk as B, G, R, A (sections._packed_colors)
        colours = [(int(bb), int(gg), int(r), 255) for r, gg, bb in rgb]
        me['pkt'] = _lightmapped(g, lmuv, colours)
        me['f6c'] = struct.pack('<I', GEOM_LIGHTMAP)
    w, h = size
    tag = _filetag(m)
    staged, names3 = [], b''
    for k in range(3):
        slot_path = 'lightmap\\%s\\%d_%d.tga' % (lightmap_dir, section_index, k)
        dst = tp.pod_path(slot_path)
        blob = flat_tile((a, b), size, dst)
        names3 += glights.tex_slot(slot_path, textures.FMT_BGRA8, w, h, tag)
        staged.append((blob, dst))
        if out_dir:
            path = os.path.join(out_dir, *dst.split('\\'))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as fh:
                fh.write(blob)
    s['names3'] = names3
    geom.relayout(m)
    return staged


# --- probes ---------------------------------------------------------------------------
def ensure_probes(m):
    """Every section with an empty portidx lists the usable probe nearest its bbox
    centre. Returns [(section index, probe index)]. Lights are left alone."""
    usable = _usable_probes(m)
    if not usable:
        return []
    if 'origdatasize' not in m:
        geom.capture(m)
    changes = []
    for i, s in enumerate(m['sections']):
        if s['portidx']:
            continue
        p = _nearest(m, usable, _bbox_centre(s))
        s['portidx'] = struct.pack('<I', p)
        changes.append((i, p))
    if changes:
        geom.relayout(m)
    return changes


def light_connector(m, index, a_index, b_index, axis, library=None, out_dir=None,
                    lightmap_dir=None, size=TILE_SIZE):
    """A corridor between rooms a and b: tiles graded from a's ambient to b's along
    `axis`, and the nearer of the two rooms' probes listed. Returns light_section's list."""
    ca = room_ambient(m, a_index, library)
    cb = room_ambient(m, b_index, library)
    ax = AX[axis]
    pair = (ca, cb) if _bbox_centre(m['sections'][a_index])[ax] <= \
        _bbox_centre(m['sections'][b_index])[ax] else (cb, ca)
    if lightmap_dir is None:
        try:
            lightmap_dir = tp._host_dir(m, index)
        except ValueError:
            lightmap_dir = 'connector'
    s = m['sections'][index]
    if not s['portidx']:
        usable = set(_usable_probes(m))
        local = [p for j in (a_index, b_index) for p in tp._u32s(m['sections'][j]['portidx'])
                 if p in usable]
        centre = _bbox_centre(s)
        p = _nearest(m, local or sorted(usable), centre)
        if p is not None:
            s['portidx'] = struct.pack('<I', p)
    return light_section(m, index, pair, out_dir, lightmap_dir, axis=axis, size=size)


# --- the gate -------------------------------------------------------------------------
def _flat_white(meshes):
    """True when every colour slot of every mesh is 255 white: the fullbright look."""
    for me in meshes:
        p = me['pkt']
        decl = p['decl']
        st = decl_stride(decl)
        n = p['nverts']
        if n == 0:
            continue
        buf = np.frombuffer(p['vdata'], np.uint8, count=n * st).reshape(n, st)
        for slot, (off, t) in geom.decl_slots(decl).items():
            if slot in geom.COLOR_SLOTS and t == 9:
                if not np.all(buf[:, off:off + 3] == 255):
                    return False
    return True


def check_lighting(m, probes=True):
    """Sections that would render fullbright (no lightmap, white vertices), and with
    probes=True those listing no probe while the set has some. [] = clean."""
    out = []
    n_usable = len(_usable_probes(m))
    for i, s in enumerate(m['sections']):
        name = geom.name_str(s['name'])
        meshes = _render_meshes(s)
        lit = bool(tp.slot_name(s['names3'][:tp.SLOT]))
        if meshes and not lit and _flat_white(meshes):
            out.append('section %d (%s): no lightmap and flat white vertex colours, '
                       'renders fullbright' % (i, name))
        if probes and n_usable and not s['portidx']:
            out.append('section %d (%s): lists no probe while the set has %d, a prop there '
                       'renders black' % (i, name, n_usable))
    return out


# --- CLI ------------------------------------------------------------------------------
def _load(path):
    with open(path, 'rb') as fh:
        return bst.parse(fh.read())


def cmd_check(a):
    rc = 0
    for f in a.files:
        bad = check_lighting(_load(f), probes=not a.no_probes)
        print('%s %s: %d problems' % ('OK  ' if not bad else 'FAIL', f, len(bad)))
        for b in bad:
            print('    ' + b)
        rc |= bool(bad)
    return rc


def cmd_probes(a):
    m = _load(a.file)
    changes = ensure_probes(m)
    out = bst.build(m)
    with open(a.out, 'wb') as fh:
        fh.write(out)
    print('wrote %s: %d sections given a probe' % (a.out, len(changes)))
    for i, p in changes:
        print('    section %d (%s) -> probe %d' % (i, geom.name_str(m['sections'][i]['name']), p))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('check', help='fullbright and probe-less sections of a set')
    p.add_argument('files', nargs='+')
    p.add_argument('--no-probes', action='store_true', help='report only fullbright sections')
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser('probes', help='list the nearest probe in every probe-less section')
    p.add_argument('file')
    p.add_argument('-o', '--out', required=True)
    p.set_defaults(fn=cmd_probes)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())

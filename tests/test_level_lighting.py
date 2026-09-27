"""Flat lightmap tiles for generated sections, a probe for every section, the gate."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import glob
import os
import struct

import numpy as np
import pytest

from _run import gbtvgr, ok
from gbtvgr.level import connector as conn
from gbtvgr.level import lighting as L
from gbtvgr.level import textures
from gbtvgr.level import transplant as tp
from gbtvgr.sets import bst, geom

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRANSPLANT_DEMO = os.path.join(ROOT, 'mods', 'TransplantDemo')
GRAFT2 = os.path.join(TRANSPLANT_DEMO, 'sets', 'graft2.bst')
HARBOR1A = os.path.join(ROOT, 'mods', 'TheHarbor', 'sets', 'harbor1a.bst')
A, B, C = 0, 1, 2                  # graft2: Room25, Room26, the connector

# shipped sections with an empty portidx in a set that has probes. The engine
# tolerates it (50 of them hold nav), so the probe rule is a composed-set invariant.
SHIPPED_PROBELESS = 236


class _DirReader:
    """library.read over a mod tree, for the tiles a graft staged beside its set."""

    def __init__(self, root):
        self.root = root

    def read(self, name):
        p = os.path.join(self.root, *name.replace('/', '\\').split('\\'))
        if not os.path.isfile(p):
            return None
        with open(p, 'rb') as fh:
            return fh.read()


def _read(path):
    with open(path, 'rb') as fh:
        return fh.read()


def _need(path):
    if not os.path.isfile(path):
        pytest.skip('needs %s' % path)
    return _read(path)


def _gates(path):
    ok(gbtvgr('bst', 'verify', path))
    out = ok(gbtvgr('bst-geom', 'check', path))
    assert 'FAIL' not in out, out


def _names3(sec):
    return [tp.slot_name(sec['names3'][k * tp.SLOT:(k + 1) * tp.SLOT]) for k in range(3)]


def _desc(sec, k):
    return struct.unpack_from('<3I', sec['names3'], k * tp.SLOT + 0x48)


def _colours(sec):
    out = []
    for me in sec['meshes']:
        g = geom.decode_mesh(me['pkt'])
        for slot in sorted(g['color']):
            out.extend(g['color'][slot])
    return out


def test_room_ambient_samples_the_lit_texels():
    """Room26's tiles are 69% empty chart (black). The mean must be of the lit rest."""
    m = bst.parse(_need(GRAFT2))
    lib = _DirReader(TRANSPLANT_DEMO)
    ca = L.room_ambient(m, A, lib)
    cb = L.room_ambient(m, B, lib)
    for c in (ca, cb):
        assert len(c) == 3 and all(125.0 <= v <= 145.0 for v in c), c
    assert L.room_ambient(m, C, None) == L.MID_GREY


def test_light_section_graft2(tmp_path, bst_corpus):
    """The heaven corridor: after light_section it names three tiles that exist, the
    tile is the sampled colour, the mesh is lightmapped, and every gate is clean."""
    # the shipped graft2 is lit now; the unlit box corridor is rebuilt from cemetery1
    from test_level_caps import _box
    raw = _box(bst_corpus)['out']
    before = bst.parse(raw)
    m = bst.parse(raw)
    lib = _DirReader(TRANSPLANT_DEMO)
    assert any('section %d' % C in p for p in L.check_lighting(m, probes=False))
    white = _colours(before['sections'][C])
    assert white and all(c[:3] == (255, 255, 255) for c in white)

    ca, cb = L.room_ambient(m, A, lib), L.room_ambient(m, B, lib)
    staged = L.light_connector(m, C, A, B, 'z', library=lib, out_dir=str(tmp_path),
                               lightmap_dir='graft2')
    sec = m['sections'][C]
    assert _names3(sec) == ['lightmap\\graft2\\2_%d.tga' % k for k in range(3)]
    assert [d for _b, d in staged] == ['art\\lightmap\\graft2\\2_%d.tex' % k for k in range(3)]
    for blob, dst in staged:
        path = os.path.join(str(tmp_path), *dst.split('\\'))
        assert os.path.isfile(path) and _read(path) == blob
        w, h, img = textures.decode(blob)
        assert (w, h) == L.TILE_SIZE
        assert tuple(img[0, 0, :3]) == tuple(int(round(v)) for v in ca)
        assert tuple(img[-1, -1, :3]) == tuple(int(round(v)) for v in cb)
        assert (img[:, :, 3] == 255).all()
    for k in range(3):
        assert _desc(sec, k) == (textures.FMT_BGRA8,) + L.TILE_SIZE
    assert sec['names3'][0x5C:0x64] == before['sections'][A]['names3'][0x5C:0x64]

    for me in sec['meshes']:
        assert struct.unpack('<I', me['f6c'])[0] == 1
        g = geom.decode_mesh(me['pkt'])
        assert 12 in g['uv'] and 11 in g['color']
        u = np.array([uv[0] for uv in g['uv'][12]])
        assert u.min() == 0.0 and u.max() == 1.0
        z = np.array([p[2] for p in g['vec'][0]])
        assert np.all(np.diff(u[np.argsort(z)]) >= 0), 'lightmap u must grow along the corridor'
        assert all(c[3] == 255 for c in g['color'][11])
    lit = _colours(sec)
    assert lit != white and all(120 <= c[0] <= 145 for c in lit)
    assert tp._u32s(sec['portidx']) == [0]

    out = bst.build(m)
    after = bst.parse(out)
    assert bst.build(after) == out
    assert tp.check_refs(after) == []
    assert L.check_lighting(after) == []
    assert tp.check_additive(before, after) == ['section 2 header', 'section 2 blob']
    for i in (A, B):
        assert _names3(after['sections'][i]) == _names3(before['sections'][i])
    path = str(tmp_path / 'graft2_lit.bst')
    with open(path, 'wb') as fh:
        fh.write(out)
    _gates(path)


def test_light_section_keeps_a_lofts_own_uvs(bst_corpus, tmp_path):
    """The lit loft is born with (t along the corridor, i/n around the ring) in slot 12;
    light_section keeps them, grades the colour along t and names the tiles."""
    from test_level_caps import _loft, _box
    r = _loft(bst_corpus, light=True)
    m = r['parsed']
    sec = m['sections'][C]
    assert _names3(sec) == ['lightmap\\graft2\\2_%d.tga' % k for k in range(3)]
    assert L.check_lighting(m) == []
    assert tp.check_refs(m) == [] and tp.check_additive(r['seed'], m) == \
        tp.check_additive(_box(bst_corpus)['seed'], _box(bst_corpus)['parsed'])
    me = sec['meshes'][0]
    assert struct.unpack('<I', me['f6c'])[0] == 1
    g = geom.decode_mesh(me['pkt'])
    lm = np.array(g['uv'][12])
    pos = np.array(g['vec'][0])
    n = g['nverts'] // 6 - 1
    assert n == 96
    assert lm[:, 0].min() == 0.0 and lm[:, 0].max() == 1.0
    assert np.all(np.diff(lm[np.argsort(pos[:, 2]), 0]) >= -1e-3), 'u grows along the corridor'
    ring_v = np.round(lm[:, 1] * n).astype(int)
    assert set(ring_v) == set(range(n + 1)) and np.abs(lm[:, 1] * n - ring_v).max() < 0.02
    # the end: the ring and its flange, v stepping by 1/n around each, both seam copies there
    end = np.abs(pos[:, 2] - pos[:, 2].max()) < 1e-3
    assert sorted(ring_v[end]) == sorted(list(range(n + 1)) * 2)
    # the same uvs survive a second light_section: nothing is recomputed from positions
    m2 = bst.parse(r['out'])
    L.light_section(m2, C, ((10.0, 20.0, 30.0), (200.0, 210.0, 220.0)), None, 'graft2', axis='z')
    g2 = geom.decode_mesh(m2['sections'][C]['meshes'][0]['pkt'])
    assert g2['uv'][12] == g['uv'][12]
    cols = np.array(g2['color'][11])
    assert tuple(cols[np.argmin(lm[:, 0])]) == (30, 20, 10, 255)
    assert tuple(cols[np.argmax(lm[:, 0])]) == (220, 210, 200, 255)
    path = str(tmp_path / 'graft2_loft_lit.bst')
    with open(path, 'wb') as fh:
        fh.write(r['out'])
    _gates(path)


def test_light_section_is_deterministic():
    """Two runs give the same bytes: the build scripts byte-compare against the ship."""
    raw = _need(GRAFT2)
    outs = []
    for _ in range(2):
        m = bst.parse(raw)
        staged = L.light_section(m, C, ((100.0, 110.0, 120.0), (60.0, 50.0, 40.0)), None,
                                 'graft2', axis='z')
        outs.append((bst.build(m), [b for b, _d in staged]))
    assert outs[0] == outs[1]
    w, h, img = textures.decode(outs[0][1][0])
    assert tuple(img[0, 0, :3]) == (100, 110, 120) and tuple(img[0, -1, :3]) == (60, 50, 40)
    assert len(set(b[4:20] for b in outs[0][1])) == 3, 'each tile carries its own header hash'
    m = bst.parse(outs[0][0])
    g = geom.decode_mesh(m['sections'][C]['meshes'][0]['pkt'])
    low = min(g['color'][11], key=lambda c: c[0])
    assert low == (40, 50, 60, 255), 'vertex colours land as B, G, R, A like the editor packs them'


def test_make_connector_light_keyword(bst_corpus):
    """abyss Room02 -> room01 with light=True and no library: mid grey tiles, a probe."""
    path = os.path.join(bst_corpus, 'abyss.bst')
    if not os.path.isfile(path):
        pytest.skip('needs abyss in the .bst corpus')
    host = bst.parse(_read(path))
    host = conn.make_connector(host, 1, ('x', 1), 0, ('x', -1), 10.0, 10.0,
                               'graveyard\\pavementwet', light=True)
    idx = len(host['sections']) - 1
    sec = host['sections'][idx]
    staged = host['light_staged']
    assert [d for _b, d in staged] == ['art\\lightmap\\Abyss\\%d_%d.tex' % (idx, k) for k in range(3)]
    assert _names3(sec) == ['lightmap\\Abyss\\%d_%d.tga' % (idx, k) for k in range(3)]
    w, h, img = textures.decode(staged[0][0])
    assert (img[:, :, :3] == 128).all()
    assert len(tp._u32s(sec['portidx'])) == 1
    out = bst.build(host)
    after = bst.parse(out)
    assert bst.build(after) == out
    assert tp.check_refs(after) == []
    assert not [p for p in L.check_lighting(after) if 'section %d ' % idx in p]


def test_ensure_probes_harbor1a():
    """The street (Room64, section 1) listed no probe and the Ecto-1 there rendered
    black. ensure_probes gives it the nearest one and the gate goes from N to 0.
    The shipped set is rebuilt often, so the street's probe is stripped here first."""
    m = bst.parse(_need(HARBOR1A))
    street = 1
    if len(m['sections']) <= street or geom.name_str(m['sections'][street]['name']) != 'Room64':
        pytest.skip('harbor1a.bst no longer holds Room64 at section 1')
    geom.capture(m)
    m['sections'][street]['portidx'] = b''
    before = L.check_lighting(m)
    assert len(before) > 0 and any(p.startswith('section %d ' % street) for p in before)
    unlit = [i for i, s in enumerate(m['sections'])
             if not tp.slot_name(s['names3'][:tp.SLOT]) and L._render_meshes(s)]

    changes = L.ensure_probes(m)
    got = dict(changes)
    assert street in got
    c = L._bbox_centre(m['sections'][street])
    dists = [sum((a - b) ** 2 for a, b in zip(L._probe_pos(m, i), c))
             for i in range(len(m['portals']))]
    assert got[street] == int(np.argmin(dists))
    assert all(m['sections'][i]['portidx'] for i in range(len(m['sections'])))
    for i in unlit:
        L.light_section(m, i, L.MID_GREY, None, 'harbor1a')
    assert L.check_lighting(m) == []
    assert L.ensure_probes(m) == []
    out = bst.build(m)
    after = bst.parse(out)
    assert bst.build(after) == out
    assert tp.check_refs(after) == []
    assert L.check_lighting(after) == []


def test_shipped_sets(bst_corpus):
    """No shipped section renders fullbright. The probe rule trips a pinned 236 shipped
    sections: the engine has some fallback there that a composed set has lost."""
    files = sorted(glob.glob(os.path.join(bst_corpus, '*.bst')))
    if not files:
        pytest.skip('empty .bst corpus')
    probeless = 0
    for f in files:
        m = bst.parse(_read(f))
        assert L.check_lighting(m, probes=False) == [], f
        probeless += len(L.check_lighting(m))
    assert probeless == SHIPPED_PROBELESS

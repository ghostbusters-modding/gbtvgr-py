"""Host: a seed set carved out of one shipped section parses, checks clean, keeps the
room's bytes, and grows like any host when a section is appended to it."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import os
import struct

import pytest

from _run import gbtvgr, ok
from gbtvgr.level import host
from gbtvgr.level import link as lk
from gbtvgr.level import transplant as tp
from gbtvgr.sets import bst, geom

CAVE, TUNNEL, PIT = 42, 43, 41      # cemetery1's Room25, Room26 and the nav-less Room21


def _read(path):
    with open(path, 'rb') as fh:
        return fh.read()


@pytest.fixture(scope='module')
def cemetery1(bst_corpus):
    path = os.path.join(bst_corpus, 'cemetery1.bst')
    if not os.path.isfile(path):
        pytest.skip('needs cemetery1 in the .bst corpus')
    return bst.parse(_read(path))


def _roundtrip(m):
    data = bst.build(m)
    back = bst.parse(data)
    assert bst.build(back) == data
    return back, data


def test_seed_from_room25(cemetery1, tmp_path):
    donor = cemetery1
    seed, staged = host.new_set_from_section(donor, CAVE, 'seedtest', unembed=False)
    m, data = _roundtrip(seed)
    assert tp.check_refs(m) == [] and lk.check_links(m) == []
    assert len(m['sections']) == 1 and len(m['watervis']) == 1
    sec, src = m['sections'][0], donor['sections'][CAVE]
    # the room's own bytes ride through: mesh blob, collision, bbox, name
    assert sec['blob'] == src['blob'] and sec['bbox'] == src['bbox'] and sec['name'] == src['name']
    assert tp._section_bytes(sec) != tp._section_bytes(src)      # only the tables moved
    used = sorted(set(tp._h70(me) for me in src['meshes']))
    assert len(m['materials']) == len(used) == len(m['mat16']) // 2
    for me, orig in zip(sec['meshes'], src['meshes']):
        assert m['materials'][tp._h70(me)] == donor['materials'][tp._h70(orig)]
    assert tp._u32s(sec['portidx']) == list(range(len(m['portals']))) == [0, 1, 2, 3, 4]
    assert tp._u32s(sec['arr818']) == [0, 1] and all(tp._u32s(L['idx']) == [0] for L in m['lights'])
    assert sec['arr4f4'] == b'' and sec['arr470'] == b'' and m['fx'] == b''
    for k in range(3):
        assert tp.slot_name(sec['names3'][k * tp.SLOT:(k + 1) * tp.SLOT]) == \
            'lightmap\\seedtest\\0_%d.tga' % k
    assert len(staged) == 14 and all(dst.startswith('art\\light') and '\\seedtest\\' in dst
                                     for _src, dst in staged)
    # the build-dep list names the staged files under their new names with the old hashes
    tga = lambda pod: (pod[:-4] + '.tga').encode('latin1')
    src_of = {tga(dst): tga(src) for src, dst in staged if not dst.endswith('\\0_3.tex')}
    assert sorted(n for n, _h in m['deps']) == sorted(src_of)
    donor_hash = dict(donor['deps'])
    assert all(h == donor_hash[src_of[n]] for n, h in m['deps'])
    # the header and skybox are the donor's
    assert tp._sky_bytes(m) == tp._sky_bytes(donor)
    for k in host.HEADER_KEYS:
        assert m[k] == donor[k], k
    # one leaf, the room's bbox, and every point walks to it
    nodes = geom.bsp_nodes(m)
    assert len(nodes) == 1 and nodes[0][1] == 0 and nodes[0][2:4] == (-1, -1)
    assert nodes[0][4] == struct.unpack('<6f', sec['bbox'])
    assert tp.bsp_walk(m, (0.0, 0.0, 0.0)) == 0 and lk.bsp_leaves(m) == {0}
    # the room's nav island, regrouped from 0
    nav = m['nav']
    assert not nav['empty'] and nav['nnodes'] == 34 and nav['f'] == 1
    assert {tp.nav_header(d)['group'] for d in nav['nodes']} == {0}
    assert lk.section_nav_nodes(m, 0) == list(range(34))
    path = str(tmp_path / 'seed.bst')
    with open(path, 'wb') as fh:
        fh.write(data)
    ok(gbtvgr('bst', 'verify', path))
    assert 'FAIL' not in ok(gbtvgr('bst-geom', 'check', path))


def test_seed_grows_like_a_host(cemetery1):
    seed, _ = host.new_set_from_section(cemetery1, CAVE, 'seedtest')
    before, data = _roundtrip(seed)
    m = bst.parse(data)
    m, staged = tp.append_section(m, cemetery1, TUNNEL, (0.0, 0.0, 30.0), lightmap_dir='seedtest')
    assert len(staged) == 6 and tp.append_nav(m, cemetery1, TUNNEL, (0.0, 0.0, 30.0)) == [34, 35, 36, 37]
    geom.relayout(m)            # append_nav grows the fixed part and leaves the offsets stale
    after, _ = _roundtrip(m)
    assert tp.check_refs(after) == [] and tp.check_additive(before, after) == []
    assert lk.bsp_leaves(after) == {0, 1} and len(geom.bsp_nodes(after)) == 3
    assert tp.bsp_walk(after, (53.0, -15.0, -100.0)) != tp.bsp_walk(after, (53.0, -15.0, -50.0))
    assert after['nav']['f'] == 2 and len(after['portals']) == 6 and len(after['lights']) == 4


def test_seed_without_nav(cemetery1):
    seed, staged = host.new_set_from_section(cemetery1, PIT, 'seedtest')
    m, _ = _roundtrip(seed)
    assert m['nav']['empty'] and m['nav']['f'] == 0 and m['nav']['nnodes'] == 0
    assert tp.check_refs(m) == [] and lk.check_links(m) == []
    assert len(m['portals']) == 6 and len(staged) == 16


def test_regroup_splits_islands():
    def node(nb, group=7):
        return {'hdr': tp.pack_nav_header({'group': group, 'centroid': (0.0, 0.0, 0.0),
                                           'hmin': 0.0, 'hmax': 1.0, 'n1': (0.0, 1.0, 0.0),
                                           'n2': (0.0, 1.0, 0.0)}),
                'v': b'', 'nb': struct.pack('<%di' % len(nb), *nb), 'd0': b'', 'f15c': b'\0' * 4}
    nav = {'empty': False, 'f': 1, 'nodes': [node([1, -1]), node([0, -1]), node([-1, -1])]}
    assert host.regroup(nav) == 2 and nav['f'] == 2
    assert [tp.nav_header(d)['group'] for d in nav['nodes']] == [0, 0, 1]
    assert lk.check_nav(dict(nav, nverts=0, nnodes=3, nextra=0, nparts=0, verts=b'',
                             extra=[], parts=[])) == []

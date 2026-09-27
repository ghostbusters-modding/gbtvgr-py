"""Move a baked section between sets at the wire layer: dict edits on bst.parse(),
never a re-encode of anything but vertex positions."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import argparse
import copy
import os
import struct
import sys

from ..mesh.bvt import bvt_bbox, bvt_collect, union_bbox
from ..sets import bst, geom
from ..mesh import mtb
from ..wire import W

# Inline per-section capacities in Section::load; the reader never bound-checks them.
CAP = {'portidx': 66, 'arr4f4': 200, 'arr818': 128, 'arr470': 32}
LIGHT_FLAGS = 2000        # stack buffer in FUN_1402E94F0, indexed by light id
BVT_SLACK = 0.0101        # shipped roots overshoot their section bbox by 0.01, never more
BSP_NODE = 0x30
PROBE_SIZE = 0xEC
SLOT = 100
PROBE_SLOTS = ((0x24, 'ambient'), (0x88, 'env'))


def _u32s(b):
    return list(struct.unpack('<%dI' % (len(b) // 4), b))


def _i32s(b):
    return list(struct.unpack('<%di' % (len(b) // 4), b))


def _pack_u32s(vals):
    return struct.pack('<%dI' % len(vals), *vals)


def _h70(me):
    return struct.unpack('<H', me['h70'])[0]


# --- STexSlot and record helpers (codec additions, parked here) ------------------
def slot_name(slot):
    s = slot[8:8 + 0x40]
    e = s.find(b'\0')
    return s[:e if e >= 0 else 0x40].decode('latin1')


def set_slot_name(slot, name):
    """Bytes past the NUL stay as shipped: the desc and filetag live there."""
    b = name.encode('latin1')
    if len(b) >= 0x40:
        raise ValueError('slot name too long: %r' % name)
    out = bytearray(slot)
    out[8:8 + len(b) + 1] = b + b'\0'
    return bytes(out)


def pod_path(slot_path):
    """`lightmap\\x\\1_0.tga` -> `art\\lightmap\\x\\1_0.tex`, the loader's own rewrite."""
    return 'art\\' + os.path.splitext(slot_path)[0] + '.tex'


def unpack_probe(raw):
    """0xEC probe record -> fields. The +0x0C vec3 is unresolved, kept as 'vec'."""
    f = struct.unpack_from('<9f', raw, 0)
    return {'pos': f[0:3], 'vec': f[3:6], 'dir': f[6:9],
            'ambient': raw[0x24:0x88], 'env': raw[0x88:PROBE_SIZE]}


def pack_probe(p):
    return (struct.pack('<9f', *(p['pos'] + p['vec'] + p['dir']))
            + p['ambient'] + p['env'])


def unpack_light(l):
    """bst light dict -> the fields a transplant edits; everything else rides in 'raw'."""
    return {'name': geom.name_str(l['a']), 'pos': struct.unpack('<3f', l['pos']),
            'sections': _u32s(l['idx']), 'raw': l}


def pack_light(d):
    l = dict(d['raw'])
    l['pos'] = struct.pack('<3f', *d['pos'])
    l['idx'] = _pack_u32s(d['sections'])
    return l


def unembed_material(mat, name):
    """An embedded material as a named ref plus the .mtb bytes to ship under materials\\.
    A lone embedded material takes a load path retail never exercises alone (RE_EMBEDDED_MATERIAL)."""
    if mat.get('embedded') is None:
        return mat, None
    w = W()
    mtb.write_material(w, mat['embedded'])
    raw = name.encode('latin1')
    ref = {'ref': (raw, b'\0' * ((-(len(raw) + 1)) % 4)), 'embedded': None}
    return ref, (w.data(), 'materials\\%s.mtb' % name)


def unique_name(raw, taken, width):
    """A name field that no other record in `taken` carries; suffix _2, _3 ... when it does."""
    base = geom.name_str(raw)
    used = set(geom.name_str(t) for t in taken)
    if base not in used:
        return raw
    k = 2
    while '%s_%d' % (base, k) in used:
        k += 1
    return geom.name_bytes('%s_%d' % (base, k), width)


def duplicate_names(m):
    """Section and light names the VM would refuse to export twice."""
    out = []
    for kind, key, recs in (('section', 'name', m['sections']), ('light', 'a', m['lights'])):
        seen = set()
        for r in recs:
            n = geom.name_str(r[key])
            if n in seen:
                out.append('%s name %r appears twice' % (kind, n))
            seen.add(n)
    return out


# --- reference check ------------------------------------------------------------------
def check_refs(m):
    """Every index a set carries, checked against the table it lands in. [] = clean."""
    bad = list(duplicate_names(m))
    ns, nm, npb, nl = (len(m['sections']), len(m['materials']), len(m['portals']),
                       len(m['lights']))
    if len(m['watervis']) != ns:
        bad.append('watervis %d bytes for %d sections' % (len(m['watervis']), ns))
    if len(m['mat16']) != nm * 2:
        bad.append('mat16 %d bytes for %d materials' % (len(m['mat16']), nm))
    if nl > LIGHT_FLAGS:
        bad.append('%d lights > %d flag slots' % (nl, LIGHT_FLAGS))
    for i, s in enumerate(m['sections']):
        tag = 'section %d' % i
        for me in s['meshes']:
            if _h70(me) >= nm:
                bad.append('%s mesh %r h70 %d >= %d materials'
                           % (tag, geom.name_str(me['name']), _h70(me), nm))
        for key, cap in CAP.items():
            if len(s[key]) // 4 > cap:
                bad.append('%s %s %d entries > %d' % (tag, key, len(s[key]) // 4, cap))
        bad += ['%s portidx %d >= %d probes' % (tag, p, npb)
                for p in _u32s(s['portidx']) if p >= npb]
        bad += ['%s arr818 light %d >= %d lights' % (tag, l, nl)
                for l in _u32s(s['arr818']) if l >= nl]
        # PVS entries are s32; both readers skip negatives (FUN_1402E94F0, FUN_1402E4AB0).
        bad += ['%s arr4f4 pvs %d >= %d sections' % (tag, v, ns)
                for v in _i32s(s['arr4f4']) if v >= ns]
        bad += ['%s arr470 neighbour %d >= %d sections' % (tag, v, ns)
                for v in _u32s(s['arr470']) if v >= ns]
        if s['bvt'] is not None:
            _cv, _ct, cs, _cf = bvt_collect(s['bvt'])
            bad += ['%s bvt surf %d >= %d lrefs' % (tag, c, len(s['lrefs']))
                    for c in sorted(set(cs)) if c >= len(s['lrefs'])]
            bb = struct.unpack('<6f', s['bbox'])
            root = bvt_bbox(s['bvt'])
            exc = max([bb[k] - root[k] for k in range(3)]
                      + [root[k] - bb[k] for k in range(3, 6)])
            if exc > BVT_SLACK:
                bad.append('%s BVT root leaves section bbox by %.3f' % (tag, exc))
    for i, l in enumerate(m['lights']):
        bad += ['light %d idx section %d >= %d' % (i, v, ns)
                for v in _u32s(l['idx']) if v >= ns]
    nodes = geom.bsp_nodes(m)
    leaves = set()
    for i, (_plane, sec, front, back, _bb) in enumerate(nodes):
        if sec >= 0:
            leaves.add(sec)
            if sec >= ns:
                bad.append('bsp leaf %d section %d >= %d' % (i, sec, ns))
            if (front, back) != (-1, -1):
                bad.append('bsp leaf %d has children %d %d' % (i, front, back))
        elif not (0 <= front < len(nodes) and 0 <= back < len(nodes)):
            bad.append('bsp node %d children %d %d outside %d nodes'
                       % (i, front, back, len(nodes)))
    bad += ['section %d has no bsp leaf' % i for i in range(ns) if i not in leaves]
    nav = m['nav']
    if not nav['empty']:
        for i, d in enumerate(nav['nodes']):
            bad += ['nav node %d vert %d >= %d' % (i, v, nav['nverts'])
                    for v in _u32s(d['v']) if v >= nav['nverts']]
            bad += ['nav node %d neighbour %d outside %d nodes' % (i, v, nav['nnodes'])
                    for v in _i32s(d['nb']) if v != -1 and not 0 <= v < nav['nnodes']]
            e = struct.unpack('<I', d['f15c'])[0]
            if e >= nav['nextra']:
                bad.append('nav node %d extra %d >= %d' % (i, e, nav['nextra']))
            g = struct.unpack_from('<I', d['hdr'], 0)[0]
            if g >= nav['f']:
                bad.append('nav node %d group %d >= %d' % (i, g, nav['f']))
    return bad


# --- rigid translate ---------------------------------------------------------------------
def _tr_vec(b, off):
    x, y, z = struct.unpack('<3f', b)
    return struct.pack('<3f', x + off[0], y + off[1], z + off[2])


def _tr_bbox(b, off):
    v = struct.unpack('<6f', b)
    return struct.pack('<6f', *[v[k] + off[k % 3] for k in range(6)])


def _tr_bvt(n, off):
    n['bbox'] = _tr_bbox(n['bbox'], off)
    if 'kids' in n:
        for k in n['kids']:
            _tr_bvt(k, off)
    else:
        n['verts'] = b''.join(_tr_vec(n['verts'][i * 12:(i + 1) * 12], off)
                              for i in range(len(n['verts']) // 12))


def _tr_mesh(me, off):
    p = me['pkt']
    g = geom.decode_mesh(p)
    g['vec'][geom.S_POS] = [(x + off[0], y + off[1], z + off[2])
                            for x, y, z in g['vec'][geom.S_POS]]
    q = geom.encode_mesh(g)
    st = len(p['vdata']) // p['nverts'] if p['nverts'] else 0
    # Only the 12 position bytes of each vertex may change; anything else is a codec bug.
    for v in range(p['nverts']):
        a, b = p['vdata'][v * st + 12:(v + 1) * st], q['vdata'][v * st + 12:(v + 1) * st]
        if a != b:
            raise ValueError('mesh %r: vertex %d changed outside the position slot'
                             % (geom.name_str(me['name']), v))
    if q['idata'] != p['idata'] or q['nverts'] != p['nverts']:
        raise ValueError('mesh %r: index data changed' % geom.name_str(me['name']))
    p['vdata'] = q['vdata']
    p['bbox'] = _tr_bbox(p['bbox'], off)


def _tr_nav(nav, bb, off):
    """Verts inside bb move; nodes with every vert inside get centroid and heights moved."""
    inside = set()
    verts = bytearray(nav['verts'])
    for i in range(nav['nverts']):
        p = struct.unpack_from('<3f', verts, i * 12)
        if all(bb[k] <= p[k] <= bb[k + 3] for k in range(3)):
            inside.add(i)
            verts[i * 12:(i + 1) * 12] = _tr_vec(bytes(verts[i * 12:(i + 1) * 12]), off)
    nav['verts'] = bytes(verts)
    moved = straddle = 0
    for d in nav['nodes']:
        ring = _u32s(d['v'])
        hit = sum(1 for v in ring if v in inside)
        if hit == 0:
            continue
        if hit < len(ring):
            straddle += 1
            continue
        h = list(struct.unpack('<I11f', d['hdr']))
        h[1] += off[0]; h[2] += off[1]; h[3] += off[2]
        h[4] += off[1]; h[5] += off[1]
        d['hdr'] = struct.pack('<I11f', *h)
        moved += 1
    return {'nav_verts': len(inside), 'nav_nodes': moved, 'nav_straddle': straddle}


def translate_section(m, index, offset, nav=True):
    """Rigidly move one section and what only it owns. Returns counts of what moved;
    a probe another section lists, or a light whose section list is not just this one, stays."""
    s = m['sections'][index]
    old_bb = struct.unpack('<6f', s['bbox'])
    moved = {'meshes': 0, 'probes': 0, 'lights': 0, 'skipped_probes': [], 'skipped_lights': []}
    for me in s['meshes']:
        if me['flags'] & 0x20000:
            continue
        _tr_mesh(me, offset)
        moved['meshes'] += 1
    s['bbox'] = _tr_bbox(s['bbox'], offset)
    if s['bvt'] is not None:
        _tr_bvt(s['bvt'], offset)
    for l in s['lrefs']:
        l['v'] = l['v'][:8] + _tr_bbox(l['v'][8:], offset)
    others = set()
    for j, t in enumerate(m['sections']):
        if j != index:
            others.update(_u32s(t['portidx']))
    for p in _u32s(s['portidx']):
        if p in others:
            moved['skipped_probes'].append(p)
            continue
        rec = m['portals'][p]
        m['portals'][p] = _tr_vec(rec[:12], offset) + rec[12:]
        moved['probes'] += 1
    for li in _u32s(s['arr818']):
        L = m['lights'][li]
        if _u32s(L['idx']) != [index]:
            moved['skipped_lights'].append(li)
            continue
        L['pos'] = _tr_vec(L['pos'], offset)
        moved['lights'] += 1
    if nav and not m['nav']['empty']:
        moved.update(_tr_nav(m['nav'], old_bb, offset))
    return moved


# --- nav records ------------------------------------------------------------------------------
def nav_header(d):
    """Node header -> dict: group id, centroid, the two absolute heights, two unit normals."""
    h = struct.unpack('<I11f', d['hdr'])
    return {'group': h[0], 'centroid': h[1:4], 'hmin': h[4], 'hmax': h[5],
            'n1': h[6:9], 'n2': h[9:12]}


def pack_nav_header(h):
    return struct.pack('<I11f', h['group'], *h['centroid'], h['hmin'], h['hmax'],
                       *h['n1'], *h['n2'])


def nav_ring(d):
    return _u32s(d['v'])


def nav_neighbours(d):
    return _i32s(d['nb'])


def append_nav(host, donor, donor_index, offset):
    """Copy the donor nav polys lying inside the donor section into the host, translated.
    Returns the new host node indices; links leaving the copied set become walls (-1)."""
    dn, hn = donor['nav'], host['nav']
    if dn['empty']:
        return []
    if hn['empty']:
        hn.update({'empty': False, 'nverts': 0, 'nnodes': 0, 'nextra': 0, 'nparts': 0,
                   'f': 0, 'verts': b'', 'nodes': [], 'extra': [], 'parts': []})
    bb = struct.unpack('<6f', donor['sections'][donor_index]['bbox'])
    dv = [struct.unpack_from('<3f', dn['verts'], i * 12) for i in range(dn['nverts'])]
    inside = set(i for i, p in enumerate(dv)
                 if all(bb[k] <= p[k] <= bb[k + 3] for k in range(3)))
    # all vertices inside: a threshold cell straddling the doorway plane is dropped
    # from both rooms and link.py's stretch covers it (a centroid rule orphans it)
    sel = [ni for ni, d in enumerate(dn['nodes']) if all(v in inside for v in nav_ring(d))]
    node_map = {ni: hn['nnodes'] + k for k, ni in enumerate(sel)}
    vert_map, extra_map, group_map = {}, {}, {}
    verts = bytearray(hn['verts'])
    for ni in sel:
        d = copy.deepcopy(dn['nodes'][ni])
        ring = nav_ring(d)
        for v in ring:
            if v not in vert_map:
                vert_map[v] = len(verts) // 12
                verts += _tr_vec(dn['verts'][v * 12:(v + 1) * 12], offset)
        d['v'] = _pack_u32s([vert_map[v] for v in ring])
        d['nb'] = struct.pack('<%di' % (len(d['nb']) // 4),
                              *[node_map.get(x, -1) for x in nav_neighbours(d)])
        h = nav_header(d)
        if h['group'] not in group_map:
            group_map[h['group']] = hn['f'] + len(group_map)
        h['group'] = group_map[h['group']]
        h['centroid'] = tuple(h['centroid'][k] + offset[k] for k in range(3))
        h['hmin'] += offset[1]
        h['hmax'] += offset[1]
        d['hdr'] = pack_nav_header(h)
        e = struct.unpack('<I', d['f15c'])[0]
        if e not in extra_map:
            extra_map[e] = len(hn['extra'])
            hn['extra'].append(copy.deepcopy(dn['extra'][e]))
        d['f15c'] = struct.pack('<I', extra_map[e])
        hn['nodes'].append(d)
    hn['verts'] = bytes(verts)
    hn['nverts'] = len(verts) // 12
    hn['nnodes'] = len(hn['nodes'])
    hn['nextra'] = len(hn['extra'])
    hn['f'] += len(group_map)
    return [node_map[ni] for ni in sel]


# --- BSP leaf insertion ---------------------------------------------------------------------
def _pack_bsp(plane, sec, front, back, bb):
    return struct.pack('<4f4h6f', *(tuple(plane) + (sec, front, back, 0) + tuple(bb)))


def insert_bsp_leaf(m, index, split=None):
    """Wrap the tree in a new root: one axis plane, the new leaf in front, old root behind.
    split = (axis, value); None bisects the widest gap between the section and the tree."""
    old = m['bsp']
    n = len(old) // BSP_NODE
    sec_bb = struct.unpack('<6f', m['sections'][index]['bbox'])
    if n == 0:
        m['bsp'] = _pack_bsp((0.0, 0.0, 0.0, 0.0), index, -1, -1, sec_bb)
        return None
    root_bb = geom.bsp_nodes(m)[0][4]
    host_bb = union_bbox([root_bb] + [s['bbox'] for j, s in enumerate(m['sections'])
                                      if j != index])
    if split is None:
        gaps = [(host_bb[ax] - sec_bb[ax + 3], ax, -1) for ax in range(3)]
        gaps += [(sec_bb[ax] - host_bb[ax + 3], ax, 1) for ax in range(3)]
        gap, ax, side = max(gaps)
        if gap <= 0:
            raise ValueError('section %d overlaps the host volume on every axis' % index)
        value = (host_bb[ax] + sec_bb[ax + 3]) / 2.0 if side < 0 \
            else (sec_bb[ax] + host_bb[ax + 3]) / 2.0
    else:
        ax, value = split
        if sec_bb[ax + 3] <= value <= host_bb[ax]:
            side = -1
        elif host_bb[ax + 3] <= value <= sec_bb[ax]:
            side = 1
        else:
            raise ValueError('split %r does not separate section %d from the host'
                             % (split, index))
    top = union_bbox([root_bb, sec_bb])
    normal = [0.0, 0.0, 0.0]
    normal[ax] = float(side)
    leaf_bb = list(top)
    # Front is n.p >= d (reader @0x14033B450): the new leaf takes the section's side.
    leaf_bb[ax + 3 if side < 0 else ax] = value
    out = bytearray(_pack_bsp(normal + [side * value], -1, n + 1, 1, top))
    for i in range(n):
        rec = bytearray(old[i * BSP_NODE:(i + 1) * BSP_NODE])
        sec, f, b, pad = struct.unpack_from('<4h', rec, 16)
        struct.pack_into('<4h', rec, 16, sec, f + 1 if f >= 0 else f,
                         b + 1 if b >= 0 else b, pad)
        out += rec
    out += _pack_bsp((0.0, 0.0, 0.0, 0.0), index, -1, -1, leaf_bb)
    m['bsp'] = bytes(out)
    return (ax, value)


def bsp_walk(m, p):
    """Leaf index for a point, descending exactly as FUN_14033D500 does."""
    nodes = geom.bsp_nodes(m)
    i = 0
    while nodes[i][1] < 0:
        n = nodes[i][0]
        i = nodes[i][2] if n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - n[3] > 0 else nodes[i][3]
    return i


# --- the transplant -------------------------------------------------------------------------
def _host_dir(m, index):
    """Directory component the host names its own tiles and probes under."""
    for s in m['sections'][:index]:
        for k in range(3):
            name = slot_name(s['names3'][k * SLOT:(k + 1) * SLOT])
            if name:
                return name.split('\\')[1]
    for rec in m['portals']:
        name = slot_name(rec[0x24:0x24 + SLOT])
        if name:
            return name.split('\\')[1]
    raise ValueError('host names no lightmap or probe; pass lightmap_dir')


def append_section(host, donor, donor_index, offset, lightmap_dir=None, split=None, unembed=True):
    """Copy donor section into host as its last section, translated by offset.
    Returns (host, staged): the (pod_path, new_pod_path) tile and cube copies it needs."""
    for m in (host, donor):
        if 'origdatasize' not in m:
            geom.capture(m)
    sec = copy.deepcopy(donor['sections'][donor_index])
    idx = len(host['sections'])
    # the loader exports every section as a VM global "CRoom <name>": a repeat is fatal
    sec['name'] = unique_name(sec['name'], [x['name'] for x in host['sections']], 0x40)
    host['sections'].append(sec)
    host['watervis'] += donor['watervis'][donor_index:donor_index + 1]
    remap = {}
    mtb_dir = lightmap_dir or _host_dir(host, idx)
    for i in sorted(set(_h70(me) for me in sec['meshes'])):
        remap[i] = len(host['materials'])
        mat = copy.deepcopy(donor['materials'][i])
        mat, shipped = (unembed_material(mat, '%s\\emb_%d_%d' % (mtb_dir, idx, i))
                        if unembed else (mat, None))
        host['materials'].append(mat)
        if shipped:
            host.setdefault('mtb_staged', []).append(shipped)
        host['mat16'] += donor['mat16'][i * 2:i * 2 + 2]
    for me in sec['meshes']:
        me['h70'] = struct.pack('<H', remap[_h70(me)])
    dirname = lightmap_dir or _host_dir(host, idx)
    staged = []
    names3 = bytearray(sec['names3'])
    lit = []
    for k in range(3):
        slot = bytes(names3[k * SLOT:(k + 1) * SLOT])
        old = slot_name(slot)
        if not old:
            continue
        new = 'lightmap\\%s\\%d_%d.tga' % (dirname, idx, k)
        names3[k * SLOT:(k + 1) * SLOT] = set_slot_name(slot, new)
        staged.append((pod_path(old), pod_path(new)))
        lit.append(old)
    sec['names3'] = bytes(names3)
    if lit:
        # The never-read _3 tile ships beside every lit section; staged so the dir looks retail.
        staged.append((pod_path(lit[0].rsplit('_', 1)[0] + '_3.tga'),
                       pod_path('lightmap\\%s\\%d_3.tga' % (dirname, idx))))
    new_probes = []
    for p in _u32s(sec['portidx']):
        rec = bytearray(donor['portals'][p])
        n = len(host['portals'])
        for off, kind in PROBE_SLOTS:
            slot = bytes(rec[off:off + SLOT])
            old = slot_name(slot)
            if not old:
                continue
            new = 'lightprobe\\%s\\%s_%d.tga' % (dirname, kind, n)
            rec[off:off + SLOT] = set_slot_name(slot, new)
            staged.append((pod_path(old), pod_path(new)))
        host['portals'].append(bytes(rec))
        new_probes.append(n)
    sec['portidx'] = _pack_u32s(new_probes)
    new_lights = []
    for li in _u32s(sec['arr818']):
        L = copy.deepcopy(donor['lights'][li])
        L['idx'] = _pack_u32s([idx])
        # lights are exported as "CPortalLight <name>" the same way
        L['a'] = unique_name(L['a'], [x['a'] for x in host['lights']], 0x20)
        new_lights.append(len(host['lights']))
        host['lights'].append(L)
    sec['arr818'] = _pack_u32s(new_lights)
    # Both lists index the donor set. Empty means no culling: abyss ships every section so.
    sec['arr4f4'] = b''
    sec['arr470'] = b''
    translate_section(host, idx, offset, nav=False)
    insert_bsp_leaf(host, idx, split)
    geom.relayout(host)
    return host, staged


# --- additivity -------------------------------------------------------------------------------
_SCALARS = ('hash', 'deps', 'boolA', 'fogA', 'fogB', 'boolB', 'f2f08', 'f62d0', 'hdr2a',
            'vec6310', 'hdr2b', 'f6358', 's6360', 'hdr2c', 'fx', 'breakers', 'wblobs',
            'wmesh', 'bspglob', 'lightver')


def _section_bytes(s):
    w = W()
    t = dict(s)
    t['dataofs'] = 0
    bst.write_section(w, t)
    return w.data()


def _sky_bytes(m):
    w = W()
    bst.write_skybox(w, m['sky'])
    return w.data()


def check_additive(before, after):
    """Every host byte a transplant must leave alone, compared. [] = nothing touched."""
    diffs = []
    ns = len(before['sections'])
    if len(after['sections']) < ns:
        return ['sections %d -> %d' % (ns, len(after['sections']))]
    for i, s in enumerate(before['sections']):
        t = after['sections'][i]
        if _section_bytes(s) != _section_bytes(t):
            diffs.append('section %d header' % i)
        if s['blob'] != t['blob']:
            diffs.append('section %d blob' % i)
    if _sky_bytes(before) != _sky_bytes(after):
        diffs.append('skybox')
    diffs += [k for k in _SCALARS if before[k] != after[k]]
    if after['datasize'] < before['datasize']:
        diffs.append('datasize shrank')
    nm = len(before['materials'])
    if before['materials'] != after['materials'][:nm]:
        diffs.append('materials[:%d]' % nm)
    if before['mat16'] != after['mat16'][:nm * 2]:
        diffs.append('mat16[:%d]' % nm)
    for key in ('lights', 'portals', 'watervis'):
        n = len(before[key])
        if before[key] != after[key][:n]:
            diffs.append('%s[:%d]' % (key, n))
    hn, rn = before['nav'], after['nav']
    if hn['empty'] or rn['empty']:
        if hn != rn:
            diffs.append('nav')
    else:
        for key in ('verts', 'nodes', 'extra', 'parts'):
            if hn[key] != rn[key][:len(hn[key])]:
                diffs.append('nav %s[:old]' % key)
        if rn['f'] < hn['f']:
            diffs.append('nav group count shrank')
    if not _bsp_kept(before['bsp'], after['bsp']):
        diffs.append('bsp %d -> %d nodes' % (len(before['bsp']) // BSP_NODE,
                                             len(after['bsp']) // BSP_NODE))
    return diffs


def _bsp_kept(old, new):
    """Old nodes must sit as one shifted block: a wrap prepends 1 node, a wedge split 2."""
    n, grown = len(old) // BSP_NODE, (len(new) - len(old)) // BSP_NODE
    if grown < 0:
        return False
    for k in range(grown + 1):
        for i in range(n):
            a = bytearray(old[i * BSP_NODE:(i + 1) * BSP_NODE])
            sec, f, b, pad = struct.unpack_from('<4h', a, 16)
            struct.pack_into('<4h', a, 16, sec, f + k if f >= 0 else f,
                             b + k if b >= 0 else b, pad)
            if bytes(a) != new[(i + k) * BSP_NODE:(i + k + 1) * BSP_NODE]:
                break
        else:
            return True
    return False


# --- CLI --------------------------------------------------------------------------
def _load(path):
    with open(path, 'rb') as fh:
        return bst.parse(fh.read())


def cmd_check_refs(a):
    rc = 0
    for f in a.files:
        bad = check_refs(_load(f))
        print('%s %s: %d problems' % ('OK  ' if not bad else 'FAIL', f, len(bad)))
        for b in bad:
            print('    ' + b)
        rc |= bool(bad)
    return rc


def cmd_append(a):
    host, donor = _load(a.host), _load(a.donor)
    before = _load(a.host)
    split = (a.split[0], a.split[1]) if a.split else None
    host, staged = append_section(host, donor, a.donor_index, tuple(a.offset),
                                  a.dir, split)
    out = bst.build(host)
    with open(a.out, 'wb') as fh:
        fh.write(out)
    print('wrote %s (%d bytes): section %d, %d files to stage'
          % (a.out, len(out), len(host['sections']) - 1, len(staged)))
    for src, dst in staged:
        print('    %s -> %s' % (src, dst))
    bad = check_refs(bst.parse(out))
    diffs = check_additive(before, bst.parse(out))
    print('check-refs: %d problems, check-additive: %d differences' % (len(bad), len(diffs)))
    for line in bad + diffs:
        print('    ' + line)
    if a.stage:
        from ..archive.assets import ArtIndex
        art = ArtIndex(a.game)
        if not art:
            print('no archives under %r; nothing staged' % a.game)
            return 1
        missing = 0
        for src, dst in staged:
            data = art.read(src)
            if data is None:
                print('    missing in the archives: %s' % src)
                missing += 1
                continue
            path = os.path.join(a.stage, *dst.split('\\'))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as fh:
                fh.write(data)
        art.close()
        print('staged %d files under %s, %d missing' % (len(staged) - missing, a.stage, missing))
        return int(bool(missing))
    return int(bool(bad or diffs))


def cmd_check_additive(a):
    diffs = check_additive(_load(a.before), _load(a.after))
    print('%s %s -> %s: %d differences' % ('OK  ' if not diffs else 'FAIL', a.before,
                                           a.after, len(diffs)))
    for d in diffs:
        print('    ' + d)
    return int(bool(diffs))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('check-refs', help='dangling index report for a set')
    p.add_argument('files', nargs='+')
    p.set_defaults(fn=cmd_check_refs)
    p = sub.add_parser('append', help='lift one donor section into a host set')
    p.add_argument('host'); p.add_argument('donor'); p.add_argument('donor_index', type=int)
    p.add_argument('--offset', type=float, nargs=3, required=True, metavar=('X', 'Y', 'Z'))
    p.add_argument('-o', '--out', required=True)
    p.add_argument('--dir', help='directory component for the new tile and probe names')
    p.add_argument('--split', type=float, nargs=2, metavar=('AXIS', 'VALUE'),
                   help='BSP root plane; default bisects the widest gap')
    p.add_argument('--stage', metavar='DIR', help='copy the tiles and cubes to DIR/art/...')
    p.add_argument('--game', default=os.environ.get('GAME_DIR'),
                   help='game directory holding the POD chain (default $GAME_DIR)')
    p.set_defaults(fn=cmd_append)
    p = sub.add_parser('check-additive', help='every host byte a transplant must keep')
    p.add_argument('before'); p.add_argument('after')
    p.set_defaults(fn=cmd_check_additive)
    a = ap.parse_args(argv)
    if a.cmd == 'append' and a.split:
        a.split = (int(a.split[0]), a.split[1])
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())

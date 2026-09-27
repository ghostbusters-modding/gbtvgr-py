"""The shopping list of shipped content: sections, actor "beats" and audio
strings across the whole corpus, indexed by code so a director can browse it."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import argparse
import collections
import csv
import glob
import json
import os
import re
import struct
import subprocess
import sys

from .. import lvl as _lvl
from ..sets import bst, geom
from . import transplant, wire

FX_REC_SIZE = 0x118  # RE_SECTION_LINK.md 2.1: the doorway record in the fx table
NAMES3_SLOT = 100     # STexSlot width, lightmap_format.md #4

# A section's real openings, but only once gbtvgr.level.link exists and this
# holds; until then the fx table is read directly (see read_doorways below).
EXTERIOR_SETS = {'cemetery2', 'timessquare1', 'timessquare1b', 'timessquare2', 'lost_island'}
# Sections with no BVT to test geometrically; named by hand from the corpus survey.
EXTERIOR_SECTIONS = {
    'hotel1a': {'room_exterior1', 'room_exterior2', 'room_exterior4',
                'room_exterior5', 'room_exterior6', 'room_loungeExterior3'},
}
CEIL_RATIO_EXTERIOR = 0.2  # ceiling area below this fraction of floor area -> open to the sky

# Corpus-observed material folders (level_census-style survey of the 20 .bst material
# tables); open-ended on purpose, a section's own materials decide the actual tag.
STYLE_VOCAB = ('graveyard', 'museum', 'hotel', 'industrial', 'street', 'sewer',
               'firehouse', 'library', 'timessquare', 'lost_island', 'skyline')

BEAT_CLASSES = {
    'CCameraPathActor': 'camera_paths', 'CAniModel': 'animodels',
    'CTrigger': 'triggers', 'CSpawn': 'spawns',
    'CPKESource': 'pkesources', 'CBarrier': 'barriers',
}

HANDLER_RE = re.compile(r'^void\s+(\w+)\(')
FUNC_HEADER_RE = re.compile(r'^(?:void|bool|int|float|String)\s+(\w+)\([^\n]*\)\s*$')
AUDIO_CALL_RE = re.compile(r'\b(setMusic|playStinger|loadMixSnapshot)\(\s*"([^"]*)"')


# ============================================================================
# small shared helpers
# ============================================================================

def _f6(b):
    return struct.unpack('<6f', b)


def _bbox_size(bbox):
    return (bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])


def _h70(mesh):
    return struct.unpack('<H', mesh['h70'])[0]


def _material_ref(entry):
    """Bare `folder/name` for one material table entry, no index prefix."""
    ref = geom.name_str(entry['ref'][0])
    return ref.replace('\\', '/') if ref else 'embedded'


def _lit(section):
    """All 3 lightmap tile slots filled, or none: level_components.md "Lightmap tiles"."""
    slots = [section['names3'][i * NAMES3_SLOT:(i + 1) * NAMES3_SLOT] for i in range(3)]
    return all(s[0] != 0 for s in slots)


# ============================================================================
# doorways: the fx table (RE_SECTION_LINK.md 2.1)
# ============================================================================

def _fx_records(m):
    """[{'index','a','b','verts'}] parsed straight from the fx blob.
    verts is a 1..8 vec3 list (stacked quad copies, not deduplicated here)."""
    fx = m['fx']
    out = []
    for i in range(len(fx) // FX_REC_SIZE):
        rec = fx[i * FX_REC_SIZE:(i + 1) * FX_REC_SIZE]
        nverts = min(8, struct.unpack_from('<I', rec, 0x7C)[0])
        verts = [struct.unpack_from('<3f', rec, 0x80 + k * 12) for k in range(nverts)]
        a, b = struct.unpack_from('<2i', rec, 0x108)
        out.append({'index': i, 'a': a, 'b': b, 'verts': verts})
    return out


def read_doorways(m):
    """Prefers gbtvgr.level.link's reader once it exists; else the fx blob directly
    (this file was written before link.py landed, so that path is unverified)."""
    try:
        from . import link as _link  # noqa: local import, may not exist yet
        recs = _link.read_doorways(m)
        out = []
        for i, r in enumerate(recs):
            if isinstance(r, dict):
                a = r.get('a', r.get('A'))
                b = r.get('b', r.get('B'))
                verts = list(r.get('verts', r.get('quad', ())))
            else:
                a, b, verts = r[0], r[1], list(r[2])
            out.append({'index': i, 'a': a, 'b': b, 'verts': verts})
        return out
    except Exception:
        return _fx_records(m)


def openings_for_section(m, section_index):
    """[{'doorway','other_section','axis','sign','centre','width','height'}] for
    every doorway naming this section, in world space (RE_SECTION_LINK.md 2.1/2.3)."""
    out = []
    sbb = _f6(m['sections'][section_index]['bbox'])
    scentre = ((sbb[0] + sbb[3]) / 2.0, (sbb[1] + sbb[4]) / 2.0, (sbb[2] + sbb[5]) / 2.0)
    for rec in read_doorways(m):
        if rec['a'] != section_index and rec['b'] != section_index:
            continue
        verts = rec['verts']
        if not verts:
            continue
        xs = [p[0] for p in verts]; ys = [p[1] for p in verts]; zs = [p[2] for p in verts]
        ext = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        axis_i = min(range(3), key=lambda k: ext[k])  # the flat (wall-normal) axis
        centre = (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))
        if axis_i == 1:  # a floor/ceiling hatch: no vertical "height" to report
            width, height = max(ext[0], ext[2]), 0.0
        else:
            height = ext[1]
            other = 0 if axis_i == 2 else 2
            width = ext[other]
        sign = 1 if centre[axis_i] >= scentre[axis_i] else -1
        other_section = rec['b'] if rec['a'] == section_index else rec['a']
        out.append({'doorway': rec['index'], 'other_section': other_section,
                    'axis': 'xyz'[axis_i], 'sign': sign, 'centre': centre,
                    'width': width, 'height': height})
    return out


# ============================================================================
# interior/exterior guess and style tag
# ============================================================================

def _bvt_floor_ceiling_areas(bvt):
    """Summed triangle area facing mostly up (floor) or mostly down (ceiling)."""
    if bvt is None:
        return 0.0, 0.0
    verts, tris, _surf, _flags = geom.bvt_collect(bvt)
    floor = ceil_ = 0.0
    for a, b, c in tris:
        pa, pb, pc = verts[a], verts[b], verts[c]
        ux, uy, uz = pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]
        vx, vy, vz = pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        length = (nx * nx + ny * ny + nz * nz) ** 0.5
        if length == 0:
            continue
        area = length / 2.0
        ny_n = ny / length
        if ny_n > 0.5:
            floor += area
        elif ny_n < -0.5:
            ceil_ += area
    return floor, ceil_


def guess_interior(set_name, section_name, has_bvt, floor_area, ceil_area):
    """Skybox-facing guess: little ceiling over the floor means open to the sky.
    Falls back to a named list only when there's no BVT to measure."""
    if has_bvt and floor_area > 0:
        return 'exterior' if (ceil_area / floor_area) < CEIL_RATIO_EXTERIOR else 'interior'
    if section_name in EXTERIOR_SECTIONS.get(set_name, ()):
        return 'exterior'
    if set_name in EXTERIOR_SETS:
        return 'exterior'
    return 'unknown'


def _segments(mat_ref):
    if mat_ref == 'embedded' or not mat_ref:
        return []
    return [s.lower() for s in mat_ref.split('/')]


def _vocab_counts(mat_refs):
    c = collections.Counter()
    for ref in mat_refs:
        for seg in _segments(ref):
            for word in STYLE_VOCAB:
                if word in seg:
                    c[word] += 1
                    break
    return c


def style_tag(mat_refs, home):
    """Comma-joined vocabulary words this section's materials touch, home style
    (the set's own dominant folder) dropped unless nothing else matches."""
    c = _vocab_counts(mat_refs)
    if home in c and len(c) > 1:
        del c[home]
    if c:
        return ','.join(sorted(c))
    return home or ''


# ============================================================================
# nav: per-section node count
# ============================================================================

def _nav_centroids(nav):
    if nav.get('empty'):
        return []
    return [struct.unpack_from('<3f', node['hdr'], 4) for node in nav['nodes']]


def _count_nav_in_bbox(centroids, bbox, pad=2.0):
    x0, y0, z0, x1, y1, z1 = bbox
    return sum(1 for (x, y, z) in centroids
               if x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad
               and z0 - pad <= z <= z1 + pad)


# ============================================================================
# sections
# ============================================================================

def _section_mesh_stats(s):
    inline = [me for me in s['meshes'] if not (me['flags'] & 0x20000)]
    instanced = [me for me in s['meshes'] if me['flags'] & 0x20000]
    tris = sum(me['pkt']['nprims'] for me in inline)
    return len(s['meshes']), len(inline), len(instanced), tris


def section_entry(set_name, m, index, nav_centroids, home):
    s = m['sections'][index]
    name = geom.name_str(s['name'])
    bbox = _f6(s['bbox'])
    mesh_count, inline_n, instanced_n, tris = _section_mesh_stats(s)
    mat_idx = sorted(set(_h70(me) for me in s['meshes']))
    mat_refs = [_material_ref(m['materials'][i]) for i in mat_idx if i < len(m['materials'])]
    has_bvt = s['bvt'] is not None
    floor_a, ceil_a = _bvt_floor_ceiling_areas(s['bvt'])
    return {
        'set': set_name, 'index': index, 'name': name,
        'bbox': list(bbox), 'size': list(_bbox_size(bbox)),
        'mesh_count': mesh_count, 'inline_meshes': inline_n, 'instanced_meshes': instanced_n,
        'triangle_count': tris,
        'lit': _lit(s), 'materials': mat_refs,
        'probe_count': len(s['portidx']) // 4, 'light_count': len(s['arr818']) // 4,
        'has_bvt': has_bvt, 'nav_node_count': _count_nav_in_bbox(nav_centroids, bbox),
        'openings': openings_for_section(m, index),
        'interior': guess_interior(set_name, name, has_bvt, floor_a, ceil_a),
        'style': style_tag(mat_refs, home),
    }


def _cache_key(path):
    st = os.stat(path)
    return {'mtime': st.st_mtime, 'size': st.st_size}


def build_sections(bst_dir, cache_dir=None):
    """Decoding a .bst (parse + BVT areas + nav) is the slow part; cache_dir keeps
    one JSON per set, keyed by the source file's mtime/size, across repeat runs."""
    out = []
    for path in sorted(glob.glob(os.path.join(bst_dir, '*.bst'))):
        set_name = os.path.splitext(os.path.basename(path))[0]
        key = _cache_key(path)
        cpath = os.path.join(cache_dir, set_name + '.json') if cache_dir else None
        if cpath and os.path.isfile(cpath):
            with open(cpath) as fh:
                cached = json.load(fh)
            if cached.get('source') == key:
                out += cached['sections']
                continue
        with open(path, 'rb') as fh:
            m = bst.parse(fh.read())
        all_refs = [_material_ref(e) for e in m['materials']]
        home_c = _vocab_counts(all_refs)
        home = home_c.most_common(1)[0][0] if home_c else ''
        nav_centroids = _nav_centroids(m['nav'])
        rows = [section_entry(set_name, m, i, nav_centroids, home)
                for i in range(len(m['sections']))]
        if cpath:
            os.makedirs(cache_dir, exist_ok=True)
            _write_json(cpath, {'source': key, 'sections': rows})
        out += rows
    return out


# ============================================================================
# beats: dante decompiles and the driver-function index
# ============================================================================

def _functions(dn_text):
    """[(name, body)] for every top-level function: header line then a lone
    '{' (control-flow braces sit on the same line as their condition)."""
    lines = dn_text.split('\n')
    out = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].rstrip('\r')
        if i + 1 < n and lines[i + 1].rstrip('\r') == '{':
            m = FUNC_HEADER_RE.match(line)
            if m:
                start = i + 2
                j = start
                while j < n and lines[j].rstrip('\r') != '}':
                    j += 1
                out.append((m.group(1), '\n'.join(lines[start:j])))
                i = j + 1
                continue
        i += 1
    return out


def _driver_index(dn_text):
    """token -> sorted [function names] that reference it, built once per module."""
    idx = collections.defaultdict(set)
    for name, body in _functions(dn_text):
        for tok in set(re.findall(r'[A-Za-z_]\w*', body)):
            idx[tok].add(name)
    return idx


def _get_or_decompile_dn(world_dir, stem, out_root, precomputed_root=None):
    if precomputed_root:
        p = os.path.join(precomputed_root, stem, stem + '.dn')
        if os.path.isfile(p):
            with open(p, encoding='latin1') as fh:
                return fh.read()
    out_dir = os.path.join(out_root, stem)
    dn_path = os.path.join(out_dir, stem + '.dn')
    if not os.path.isfile(dn_path):
        src = os.path.join(world_dir, stem + '.dante')
        if not os.path.isfile(src):
            return None
        os.makedirs(out_dir, exist_ok=True)
        r = subprocess.run([sys.executable, '-m', 'dante', 'decompile', src, '-o', out_dir],
                            capture_output=True, text=True)
        if r.returncode != 0 or not os.path.isfile(dn_path):
            return None
    with open(dn_path, encoding='latin1') as fh:
        return fh.read()


def _parse_vec(s):
    try:
        return tuple(float(x) for x in s.split(','))
    except ValueError:
        return None


def _iter_kvs(node):
    for c in node.children:
        if isinstance(c, _lvl.KV):
            yield c.key, c.value
        elif isinstance(c, _lvl.Block):
            yield from _iter_kvs(c)


def handler_names(actor_block):
    """Every `void name(...)` reference anywhere in this actor's own fields."""
    names = []
    for _key, val in _iter_kvs(actor_block):
        m = HANDLER_RE.match(val)
        if m:
            names.append(m.group(1))
    return names


def _actor_section(bst_m, pos):
    if bst_m is None or pos is None:
        return None
    try:
        leaf = transplant.bsp_walk(bst_m, pos)
        return geom.bsp_nodes(bst_m)[leaf][1]
    except (IndexError, ValueError, ZeroDivisionError):
        return None


def cinematics_for_level(lv):
    """(cinemat-list names, [{'name','duration','num_events'}] from <cinemats>)."""
    names_list, defs = [], []
    for _n, doc in lv.docs():
        cl = _lvl.top_block(doc, 'cinemat-list')
        if cl:
            names_list += [c.value for c in cl.children
                           if isinstance(c, _lvl.KV) and c.key == 'cinemat']
        cm = _lvl.top_block(doc, 'cinemats')
        if cm:
            for c in cm.children:
                if not isinstance(c, _lvl.Block):
                    continue
                props = _lvl.top_block(c, 'properties')
                fields = _lvl.actor_fields(props) if props else {}
                dur = fields.get('duration')
                nev = fields.get('numEvents')
                defs.append({'name': c.tag,
                            'duration': float(dur) if dur else None,
                            'num_events': int(nev) if nev else None})
    return names_list, defs


def _camera_path_stats(actor):
    keys = _lvl.top_block(actor, 'Keys')
    if keys is None:
        return 0, 0.0
    duration = 0.0
    for kb in keys.children:
        if isinstance(kb, _lvl.Block):
            d = _lvl.actor_fields(kb).get('duration')
            if d:
                duration += float(d)
    return len(keys.children), duration


def build_beats_for_level(lvl_dir, bst_dir, stem, dn_out_root, dn_precomputed_root=None):
    src = wire.DirSource(root=lvl_dir, sets=bst_dir)
    lv = wire.open_level(src, stem, load_set=True)
    world_dir = os.path.join(lvl_dir, 'world')
    dn_text = _get_or_decompile_dn(world_dir, stem, dn_out_root, dn_precomputed_root)
    driver_idx = _driver_index(dn_text) if dn_text else {}

    beats = {v: [] for v in BEAT_CLASSES.values()}
    class_maps = wire.actor_classes(lv)
    for sec_idx, actor in wire.actors(lv):
        cls = class_maps[sec_idx].get(actor.tag, '?')
        kind = BEAT_CLASSES.get(cls)
        if kind is None:
            continue
        fields = _lvl.actor_fields(actor)
        pos = _parse_vec(fields['pos']) if 'pos' in fields else None
        section = _actor_section(lv.bst, pos)
        drivers = sorted(driver_idx.get(actor.tag, ()))
        row = {'name': actor.tag, 'pos': pos, 'section': section, 'drivers': drivers}
        if cls == 'CCameraPathActor':
            row['key_count'], row['duration'] = _camera_path_stats(actor)
        elif cls == 'CAniModel':
            row['model'] = fields.get('modelInstance', '')
            row['handlers'] = handler_names(actor)
        elif cls == 'CTrigger':
            row['handlers'] = handler_names(actor)
        beats[kind].append(row)

    cinemat_list, cinemats = cinematics_for_level(lv)
    return {'level': stem, 'cinemat_list': cinemat_list, 'cinemats': cinemats, **beats}


def build_beats(lvl_dir, bst_dir, dn_out_root, dn_precomputed_root=None, levels=None):
    src = wire.DirSource(root=lvl_dir, sets=bst_dir)
    stems = levels or src.level_stems()
    return [build_beats_for_level(lvl_dir, bst_dir, stem, dn_out_root, dn_precomputed_root)
            for stem in sorted(stems)]


# ============================================================================
# audio: setMusic / playStinger / loadMixSnapshot across every script
# ============================================================================

def audio_events(dn_text):
    out = []
    for name, body in _functions(dn_text):
        for m in AUDIO_CALL_RE.finditer(body):
            out.append({'function': name, 'call': m.group(1), 'arg': m.group(2)})
    return out


def build_audio(lvl_dir, dn_out_root, dn_precomputed_root=None):
    world_dir = os.path.join(lvl_dir, 'world')
    stems = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(world_dir, '*.dante')))
    out = []
    for stem in stems:
        dn_text = _get_or_decompile_dn(world_dir, stem, dn_out_root, dn_precomputed_root)
        if dn_text is None:
            continue
        for ev in audio_events(dn_text):
            ev = dict(ev)
            ev['level'] = stem
            out.append(ev)
    return out


# ============================================================================
# build / cache / CLI
# ============================================================================

DN_PRECOMPUTED_DEFAULT = ('inventory', 'census', 'dante_dn')  # under $GB_BUILD, from B3_COMMON


def _default_dn_precomputed(gb_build):
    if not gb_build:
        return None
    p = os.path.join(gb_build, *DN_PRECOMPUTED_DEFAULT)
    return p if os.path.isdir(p) else None


def _write_json(path, data):
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=1)


def _write_sections_csv(path, rows):
    cols = ['set', 'index', 'name', 'bbox_min_x', 'bbox_min_y', 'bbox_min_z',
            'bbox_max_x', 'bbox_max_y', 'bbox_max_z', 'size_x', 'size_y', 'size_z',
            'mesh_count', 'triangle_count', 'lit', 'materials', 'probe_count',
            'light_count', 'has_bvt', 'nav_node_count', 'opening_count',
            'openings', 'interior', 'style']
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            op_summary = ';'.join('%s%+d->%s' % (o['axis'], o['sign'], o['other_section'])
                                  for o in r['openings'])
            w.writerow([r['set'], r['index'], r['name']] + r['bbox'] + r['size'] +
                       [r['mesh_count'], r['triangle_count'], r['lit'],
                        ';'.join(r['materials']), r['probe_count'], r['light_count'],
                        r['has_bvt'], r['nav_node_count'], len(r['openings']),
                        op_summary, r['interior'], r['style']])


def _write_beats_csv(path, levels):
    cols = ['level', 'kind', 'name', 'pos_x', 'pos_y', 'pos_z', 'section',
            'key_count', 'duration', 'model', 'handlers', 'drivers']
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for lv in levels:
            for kind in BEAT_CLASSES.values():
                for row in lv[kind]:
                    pos = row['pos'] or (None, None, None)
                    w.writerow([lv['level'], kind, row['name'], pos[0], pos[1], pos[2],
                               row['section'], row.get('key_count', ''),
                               row.get('duration', ''), row.get('model', ''),
                               ';'.join(row.get('handlers', [])),
                               ';'.join(row['drivers'])])


def build_all(bst_dir, lvl_dir, out_dir, dn_precomputed_root=None):
    """Rebuilds sections.json/csv, beats.json/csv and audio.json under out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    dn_out_root = os.path.join(out_dir, 'dn')
    os.makedirs(dn_out_root, exist_ok=True)

    sections = build_sections(bst_dir, cache_dir=os.path.join(out_dir, 'cache'))
    _write_json(os.path.join(out_dir, 'sections.json'), sections)
    _write_sections_csv(os.path.join(out_dir, 'sections.csv'), sections)

    beats = build_beats(lvl_dir, bst_dir, dn_out_root, dn_precomputed_root)
    _write_json(os.path.join(out_dir, 'beats.json'), beats)
    _write_beats_csv(os.path.join(out_dir, 'beats.csv'), beats)

    audio = build_audio(lvl_dir, dn_out_root, dn_precomputed_root)
    _write_json(os.path.join(out_dir, 'audio.json'), audio)

    return sections, beats, audio


def _load_or_build_sections(a):
    path = os.path.join(a.out, 'sections.json')
    if os.path.isfile(path) and not getattr(a, 'rebuild', False):
        with open(path) as fh:
            return json.load(fh)
    if not a.bst_corpus:
        raise SystemExit('no cached sections.json and no --bst-corpus given')
    os.makedirs(a.out, exist_ok=True)
    sections = build_sections(a.bst_corpus, cache_dir=os.path.join(a.out, 'cache'))
    _write_json(path, sections)
    _write_sections_csv(os.path.join(a.out, 'sections.csv'), sections)
    return sections


def _load_or_build_beats(a):
    path = os.path.join(a.out, 'beats.json')
    if os.path.isfile(path) and not getattr(a, 'rebuild', False):
        with open(path) as fh:
            return json.load(fh)
    if not (a.bst_corpus and a.lvl_corpus):
        raise SystemExit('no cached beats.json and no --bst-corpus/--lvl-corpus given')
    os.makedirs(a.out, exist_ok=True)
    dn_out_root = os.path.join(a.out, 'dn')
    precomputed = a.dn_precomputed or _default_dn_precomputed(os.environ.get('GB_BUILD'))
    beats = build_beats(a.lvl_corpus, a.bst_corpus, dn_out_root, precomputed)
    _write_json(path, beats)
    _write_beats_csv(os.path.join(a.out, 'beats.csv'), beats)
    return beats


def cmd_build(a):
    precomputed = a.dn_precomputed or _default_dn_precomputed(os.environ.get('GB_BUILD'))
    if not (a.bst_corpus and a.lvl_corpus):
        raise SystemExit('build needs --bst-corpus and --lvl-corpus')
    sections, beats, audio = build_all(a.bst_corpus, a.lvl_corpus, a.out, precomputed)
    print('wrote %d sections, %d levels, %d audio events under %s'
          % (len(sections), len(beats), len(audio), a.out))
    return 0


def cmd_sections(a):
    rows = _load_or_build_sections(a)
    if a.set:
        rows = [r for r in rows if r['set'] == a.set]
    print('%-14s %3s %-28s %-16s %6s %5s %-8s %-5s %s'
          % ('set', 'idx', 'name', 'size', 'tris', 'lit', 'interior', 'bvt', 'style'))
    for r in rows:
        print('%-14s %3d %-28s %-16s %6d %5s %-8s %-5s %s'
              % (r['set'], r['index'], r['name'],
                 '%.0fx%.0fx%.0f' % tuple(r['size']), r['triangle_count'],
                 r['lit'], r['interior'], r['has_bvt'], r['style']))
    return 0


def cmd_beats(a):
    levels = _load_or_build_beats(a)
    if a.level:
        levels = [lv for lv in levels if lv['level'] == a.level]
    for lv in levels:
        print('== %s ==' % lv['level'])
        for kind in BEAT_CLASSES.values():
            rows = lv[kind]
            if not rows:
                continue
            print('  %s (%d):' % (kind, len(rows)))
            for row in rows:
                extra = ''
                if kind == 'camera_paths':
                    extra = ' keys=%d duration=%.1f' % (row['key_count'], row['duration'])
                elif kind == 'animodels':
                    extra = ' model=%s handlers=%s' % (row['model'], ','.join(row['handlers']))
                elif kind == 'triggers':
                    extra = ' handlers=%s' % ','.join(row['handlers'])
                drivers = (' drivers=%s' % ','.join(row['drivers'])) if row['drivers'] else ''
                print('    %-32s sec=%s%s%s' % (row['name'], row['section'], extra, drivers))
    return 0


def cmd_find(a):
    rows = _load_or_build_sections(a)
    if a.style:
        want = a.style.lower()
        rows = [r for r in rows if want in [t.strip() for t in r['style'].split(',')]]
    if a.lit:
        rows = [r for r in rows if r['lit']]
    if a.min_size:
        mx, my, mz = a.min_size
        rows = [r for r in rows if r['size'][0] >= mx and r['size'][1] >= my
                and r['size'][2] >= mz]
    if a.opening:
        rows = [r for r in rows if any(o['axis'] == a.opening for o in r['openings'])]
    for r in rows:
        print('%-14s %3d %-28s %-16s style=%s interior=%s'
              % (r['set'], r['index'], r['name'],
                 '%.0fx%.0fx%.0f' % tuple(r['size']), r['style'], r['interior']))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('--bst-corpus', default=os.environ.get('BST_CORPUS'),
                       help='directory of shipped .bst sets (default $BST_CORPUS)')
        p.add_argument('--lvl-corpus', default=os.environ.get('LVL_CORPUS'),
                       help='directory of shipped world\\*.lvl/.sec/.dante (default $LVL_CORPUS)')
        p.add_argument('--dn-precomputed', default=None,
                       help='a dir of already-decompiled <level>/<level>.dn to reuse')
        default_out = (os.path.join(os.environ['GB_BUILD'], 'catalogue')
                       if os.environ.get('GB_BUILD') else None)
        p.add_argument('--out', default=default_out,
                       help='where sections/beats/audio json+csv live (default $GB_BUILD/catalogue)')

    p = sub.add_parser('build', help='rebuild sections.json/csv, beats.json/csv, audio.json')
    common(p)
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser('sections', help='print the section table, optionally for one set')
    common(p)
    p.add_argument('--set')
    p.add_argument('--rebuild', action='store_true')
    p.set_defaults(fn=cmd_sections)

    p = sub.add_parser('beats', help='print the beat table, optionally for one level')
    common(p)
    p.add_argument('--level')
    p.add_argument('--rebuild', action='store_true')
    p.set_defaults(fn=cmd_beats)

    p = sub.add_parser('find', help='filter sections by style, lighting, size, opening axis')
    common(p)
    p.add_argument('--style')
    p.add_argument('--lit', action='store_true')
    p.add_argument('--min-size', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    p.add_argument('--opening', choices=('x', 'y', 'z'))
    p.add_argument('--rebuild', action='store_true')
    p.set_defaults(fn=cmd_find)

    a = ap.parse_args(argv)
    if a.out is None:
        ap.error('--out is required (or set $GB_BUILD)')
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())

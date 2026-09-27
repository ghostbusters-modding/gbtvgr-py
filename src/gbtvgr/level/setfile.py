"""Assemble a whole `.bst` from a document and verify it the way the shipped
generators do: parse -> rebuild byte-identical, every section box containing
its collision tree, every material set-capable.
"""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import struct

import numpy as np

from gbtvgr.mesh.bvt import bvt_bbox
from gbtvgr.sets import bst as gbst

from .mathutil import F32
from . import bsp as gbsp
from . import lights as glights
from . import sections as gsec
from . import skybox as gsky

# fields the RE has not resolved, carried as the constants of the shipped sets
BOILER_HDR2A = struct.pack('<11I', 1, 0x18, 0x18, 0x18, 0, 0x43800000, 0x3F800000, 0, 1000,
                           0x42200000, 0x42000000)
BOILER_HDR2B = struct.pack('<6I', 0, 0, 2, 136, 130, 0x3F400000)
BOILER_HDR2C = struct.pack('<7I', 0x41000000, 0x41000000, 0x3E99999A, 1, 0, 0, 0)
SECTION_PAD = 8.0
DEFAULT_MAX_EDGE = 4.0
BLOB_ALIGN = 32


def empty_nav():
    return {'empty': True, 'f': 2, 'nverts': 0, 'nnodes': 0, 'nextra': 0, 'nparts': 0}


class SetBuild:
    def __init__(self):
        self.data = b''
        self.assets = {}          # archive path -> bytes to stage beside the set
        self.report = []
        self.warnings = []
        self.errors = []
        self.materials = []
        self.section_names = []
        self.section_boxes = []
        self.bsp_nodes = []

    def log(self, line):
        self.report.append(line)

    @property
    def ok(self):
        return not self.errors


def _material_entry(ref):
    b = ref.replace('/', '\\').encode('latin1')
    return {'ref': (b, b'\0' * ((-(len(b) + 1)) % 4)), 'embedded': None}


def _names3(set_name, index, tiles, size):
    if not tiles:
        return glights.tex_slot_empty() * 3
    w, h = size
    return b''.join(glights.tex_slot('lightmap\\%s\\%d_%d.tga' % (set_name, index, k), 3, w, h)
                    for k in range(3))


def compile_set(doc, library=None, nav=None, lightmaps=None, probe_tiles=None, log=None):
    """doc -> SetBuild; `lightmaps` maps a section id to a bake_lightmaps result,
    `probe_tiles` is [(ambient, env)] bytes in probe order."""
    out = SetBuild()
    set_name = doc.settings.get('set_name', 'level')
    # set lights shade world geometry per vertex: a coarse wall never lights up
    max_edge = doc.settings.get('max_edge', DEFAULT_MAX_EDGE)
    sections = doc.section_list()
    if not sections:
        out.errors.append('the set has no sections')
        return out
    lights = [n for n in doc.lights.children if n.kind == 'light']
    probes = [n for n in doc.probes.children if n.kind == 'probe']

    # -- materials --------------------------------------------------------------------
    refs = []
    for sec in sections:
        for mo in gsec.gather_meshes(sec):
            if mo.material.lower() not in [r.lower() for r in refs]:
                refs.append(mo.material)
    if not refs:
        refs.append(gsec.DEFAULT_MATERIAL)
    mat_index = {r.lower(): i for i, r in enumerate(refs)}
    if library is not None:
        for r in refs:
            info = library.material_info(r)
            if info is None:
                out.errors.append('material %s is not in the archives' % r)
            elif not info.set_ok:
                out.errors.append('material %s lacks the set-geometry bit 0x4 (mask %#x); '
                                  'the level would fail to prepare' % (r, info.mask))
    out.materials = refs

    # -- sections ------------------------------------------------------------------------
    per_light, lights_per_sec = glights.light_sections(lights, sections)
    probes_per_sec = glights.probe_sections(probes, sections)
    recs, blobs, boxes = [], [], []
    for si, sec in enumerate(sections):
        lm = (lightmaps or {}).get(sec.id)
        lmuv = lm.get('lmuv') if lm else None
        meshes = gsec.gather_meshes(sec, lmuv_by_node=lmuv, max_edge=max_edge)
        mesh_recs, blob, mesh_boxes = [], b'', []
        for mo in meshes:
            rec, box = gsec.mesh_record(mo, mat_index[mo.material.lower()], lightmapped=bool(lm))
            mesh_recs.append(rec)
            blob += rec['pkt']['vdata'] + rec['pkt']['idata']
            mesh_boxes.append(box)
        verts, tris, surf, flags, entries = gsec.collect_collision(doc, sec)
        root, arena = gsec.build_section_bvt(verts, tris, surf, flags)
        lo, hi = sec.lo.astype(np.float64), sec.hi.astype(np.float64)
        for blo, bhi in mesh_boxes:
            lo, hi = np.minimum(lo, blo), np.maximum(hi, bhi)
        if root is not None:
            rb = bvt_bbox(root)
            lo, hi = np.minimum(lo, rb[:3]), np.maximum(hi, rb[3:])
        lo, hi = lo - SECTION_PAD, hi + SECTION_PAD
        boxes.append((lo, hi))
        others = [k for k in range(len(sections)) if k != si]
        nb = sec.name.encode('latin1')[:0x3F]
        blob += b'\0' * ((-len(blob)) % BLOB_ALIGN)
        recs.append({
            'name': nb + b'\0' * (0x40 - len(nb)),
            'b1': 1, 'b2': 0, 'b3': 1, 'f3b4': 0, 'f3b8': 0, 'skip': 0, 'f3c0': 0,
            'names3': _names3(set_name, si, lm and lm.get('tiles'), lm.get('size') if lm else None),
            'extnames': [],
            'portidx': b''.join(struct.pack('<I', k) for k in probes_per_sec[si]),
            'meshes': mesh_recs,
            'lrefs': gsec.surface_entries(entries, (lo, hi)),
            'bvtflag': arena, 'bvt': root,
            'arr4f4': b''.join(struct.pack('<I', v) for v in others),
            'arr818': b''.join(struct.pack('<I', v) for v in lights_per_sec[si]),
            'arr470': b''.join(struct.pack('<I', v) for v in others),
            'dataofs': 0, 'fa20': len(blob),
            'bbox': struct.pack('<6f', *lo, *hi),
            'blob': blob, 'datagap': b'',
        })
        out.log('section %d %s: %d meshes, %d verts, %d collision tris, arena %d, %d lights, %d probes'
                % (si, sec.name, len(mesh_recs), sum(r['pkt']['nverts'] for r in mesh_recs),
                   len(tris), arena, len(lights_per_sec[si]), len(probes_per_sec[si])))
        if lm:
            for k, tile in enumerate(lm['tiles']):
                out.assets['art\\lightmap\\%s\\%d_%d.tex' % (set_name, si, k)] = tile
    out.section_names = [s.name for s in sections]
    out.section_boxes = boxes

    # -- lights, probes, sky ---------------------------------------------------------------
    light_recs = [glights.light_record(l, per_light[i]) for i, l in enumerate(lights)]
    probe_recs = [glights.probe_record(p, i, set_name) for i, p in enumerate(probes)]
    for i, (amb, env) in enumerate(probe_tiles or []):
        out.assets['art\\lightprobe\\%s\\ambient_%d.tex' % (set_name, i)] = amb
        out.assets['art\\lightprobe\\%s\\env_%d.tex' % (set_name, i)] = env
    if probes and not probe_tiles:
        out.warnings.append('%d probes reference lightprobe tiles that were not baked' % len(probes))
    donor_raw = None
    donor = doc.skybox.props.get('donor')
    if donor and library is not None:
        # read_set is the parsed dict here, not the editor's SetData wrapper
        donor_raw = library.read_set(donor)
        if donor_raw is None:
            out.warnings.append('skybox donor set %s not found; shipping no sky layers' % donor)
    sky = gsky.build_sky(doc.skybox, donor_raw)
    fog_a = doc.skybox.props.get('fog_a', (0.03, 0.034, 0.042))
    fog_b = doc.skybox.props.get('fog_b', (0.125, 0.14, 0.165))

    # -- bsp, nav, deps ---------------------------------------------------------------------
    bsp_bytes, glob, nodes = gbsp.build_bsp(boxes)
    out.bsp_nodes = nodes
    tested, problems = gbsp.verify(nodes, boxes)
    out.log('bsp: %d nodes, %d leaves, %d/%d sample points resolve to their section'
            % (len(nodes), gbsp.leaf_count(nodes), tested - len(problems), tested))
    if problems:
        out.warnings.append('bsp: %d sample points resolve to another section (overlapping boxes?)'
                            % len(problems))
    deps = glights.probe_deps(len(probes), set_name)
    for si, sec in enumerate(sections):
        lm = (lightmaps or {}).get(sec.id)
        if lm:
            for k in range(3):
                deps.append((('art\\lightmap\\%s\\%d_%d.tga' % (set_name, si, k)).encode('latin1'),
                             b'\0' * 16))

    m = {
        'lightver': 0x29, 'hash': b'\0' * 16, 'deps': deps, 'deppad': b'',
        'datasize': 0, 'boolA': 0,
        'fogA': struct.pack('<3f', *fog_a), 'fogB': struct.pack('<3f', *fog_b),
        'boolB': 0, 'f2f08': 1,
        'sky': sky,
        'f62d0': 0, 'hdr2a': BOILER_HDR2A, 'vec6310': struct.pack('<3f', 0.0, 0.0, 0.0),
        'hdr2b': BOILER_HDR2B, 'f6358': 0, 's6360': (b'', b'\0\0\0'), 'hdr2c': BOILER_HDR2C,
        'materials': [_material_entry(r) for r in refs],
        'mat16': b'\0' * (2 * len(refs)),
        'sections': recs, 'lights': light_recs, 'portals': probe_recs,
        'fx': b'', 'breakers': [], 'wblobs': [], 'wmesh': [],
        'watervis': b'\x01' * len(recs),
        'bspglob': glob, 'bsp': bsp_bytes,
        'nav': nav or empty_nav(),
    }
    m['datasize'] = max(len(s['blob']) for s in recs) + 32
    pre = len(gbst.build(m)) - sum(len(s['blob']) for s in recs)
    ofs = pre + ((-pre) % BLOB_ALIGN)
    for si, s in enumerate(recs):
        s['datagap'] = b'\0' * (ofs - pre) if si == 0 else b''
        s['dataofs'] = ofs
        ofs += s['fa20']
        pre = ofs
    final = gbst.build(m)

    # -- verification ------------------------------------------------------------------------
    m2 = gbst.parse(final)
    if gbst.build(m2) != final:
        out.errors.append('the authored set does not round-trip through the codec')
    for si, s in enumerate(m2['sections']):
        if s['bvt'] is None:
            continue
        rb = bvt_bbox(s['bvt'])
        sb = struct.unpack('<6f', s['bbox'])
        if not (all(rb[k] >= sb[k] - 1e-3 for k in range(3)) and all(rb[k] <= sb[k] + 1e-3 for k in range(3, 6))):
            out.errors.append('section %d: collision tree reaches outside the section box' % si)
    out.data = final
    out.log('%s.bst: %d bytes, %d sections, %d materials, %d lights, %d probes, nav %s'
            % (set_name, len(final), len(recs), len(refs), len(light_recs), len(probe_recs),
               'empty' if m['nav'].get('empty') else '%d nodes' % m['nav']['nnodes']))
    if log:
        for line in out.report:
            log(line)
    return out


def section_lightmap_uv_scale(section):
    """The world extent a lightmap tile of a section spans, for the bake."""
    return float(np.max(section.hi - section.lo))

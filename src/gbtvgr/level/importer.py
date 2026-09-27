"""Shipped sets and levels as editable documents.

A set imports as sections of mesh nodes with the collision tree beside them as
a hidden mesh; lightmap tiles and breakers do not survive the round trip, the
sky and fog come along as a donor reference. A level imports every actor block
verbatim so the shipped script keeps resolving its events.
"""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import collections

import numpy as np

from gbtvgr import lvl as glvl
from gbtvgr.lvl import KV, Block, ListF

from .geometry_mesh import EditMesh
from .mathutil import F32
from .scene import ActorNode, GroupNode, LightNode, MeshNode, ProbeNode, SectionNode

HERO_CLASSES = {'CGhostbuster'}


def _mesh_from_set(sm):
    idx = np.asarray(sm.indices, np.int32)
    uv = sm.uv0 if sm.uv0 is not None else np.zeros((len(sm.positions), 2), F32)
    m = EditMesh(sm.positions, idx, uv[idx], np.zeros(len(idx), np.int32), smooth=True)
    lit = sm.lit_colors()
    if lit is not None:
        m.colors = np.asarray(lit, np.uint8).copy()
    return m


def import_set(doc, setdata, sections=None, log=None, collision=True):
    """Add the set's sections (all, or the given indices) to the document."""
    wanted = set(sections) if sections is not None else None
    added = []
    for sec in setdata.sections:
        if wanted is not None and sec.index not in wanted:
            continue
        node = SectionNode(sec.name or 'section%d' % sec.index)
        node.set_bounds(sec.bbox[0], sec.bbox[1])
        node.props['names3'] = sec.raw['names3'].hex()
        doc.add_node(node, doc.sections)
        n_meshes = 0
        for sm in sec.meshes:
            if sm.instanced or sm.positions is None or len(sm.indices) == 0:
                continue
            mn = MeshNode(sm.name or 'mesh%d' % sm.index)
            mn.mesh = _mesh_from_set(sm)
            ref = setdata.material_ref(sm.material)
            mn.props['materials'] = [ref]
            mn.props['collide'] = not collision
            mn.props['walkable'] = not collision
            mn.props['uv_scale'] = 1.0
            node.add(mn)
            n_meshes += 1
        if collision:
            verts, tris, surf, flags = sec.collision
            if len(tris):
                cn = MeshNode('%s_collision' % node.name)
                cn.mesh = EditMesh(verts, tris, np.zeros((len(tris), 3, 2), F32),
                                   np.zeros(len(tris), np.int32))
                cn.props.update(collide=True, walkable=True, double_sided=True, materials=[])
                cn.visible = False
                node.add(cn)
        added.append(node)
        if log:
            log('imported section %d %s: %d meshes' % (sec.index, sec.name, n_meshes))
    for l in setdata.lights:
        ln = LightNode(l.name or 'light%d' % l.index)
        ln.pos[:] = l.pos
        ln.props.update(color=list(l.color), radius=float(l.radius))
        doc.add_node(ln, doc.lights)
    for p in setdata.probes:
        pn = ProbeNode('probe%d' % p.index)
        pn.pos[:] = p.pos
        pn.props['direction'] = [float(v) for v in p.direction]
        doc.add_node(pn, doc.probes)
    doc.skybox.props.update(donor=setdata.name, sun=[float(v) for v in setdata.sun],
                            fog_rgb=[int(v) for v in setdata.fog_rgb],
                            fog_a=[float(v) for v in setdata.fog_a],
                            fog_b=[float(v) for v in setdata.fog_b])
    doc.settings['donor_set'] = setdata.name
    doc.notify('reset')
    return added


def _parse_vec(v):
    try:
        return np.array([float(x) for x in v.split(',')], F32)
    except ValueError:
        return np.zeros(3, F32)


def _import_actor_block(doc, block, stem, section_index, added, groups_count):
    """One block's <actors>/<actor-groups> into doc, tagged with its layer."""
    # each file (.lvl or .sec) carries its own <actor-list>; never share one cmap across files
    cmap = glvl.actor_class_map(block)
    ab = glvl.top_block(block, 'actors')
    for a in (ab.children if ab else []):
        if not isinstance(a, Block):
            continue
        cls = cmap.get(a.tag, 'CActor')
        f = glvl.actor_fields(a)
        if cls in HERO_CLASSES:
            if a.tag == 'Ghostbuster0':
                doc.settings['hero_start'] = [float(v) for v in _parse_vec(f.get('pos', '0, 0, 0'))]
                doc.settings['hero_yaw'] = float(_parse_vec(f.get('orient', '0, 0, 0'))[0])
            continue
        n = ActorNode(a.tag)
        n.props.update(cls=cls, template=a.tag, template_level=stem, keep_events=True,
                       created=f.get('createStatus', '1') == '1', section=section_index)
        n.props['fields'] = collections.OrderedDict((c.key, c.value) for c in a.children
                                                    if isinstance(c, KV) and c.key != 'name')
        n.pos = _parse_vec(f.get('pos', '0, 0, 0'))
        n.orient = _parse_vec(f.get('orient', '0, 0, 0'))
        model = f.get('modelInstance', '')
        if model and model != '""' and not model.lower().endswith('.dfm'):
            n.props['model'] = model.replace('\\\\', '\\')
        n.props['solid'] = f.get('useCollisionParts') == '1'
        doc.add_node(n, doc.actors)
        added.append(n)
    groups = glvl.top_block(block, 'actor-groups')
    for c in (groups.children if groups else []):
        if isinstance(c, ListF):
            g = GroupNode(c.key)
            g.props['members'] = [t for _e, t in c.items]
            g.props['section'] = section_index
            doc.add_node(g, doc.groups)
            groups_count[0] += 1


def import_level(doc, data, stem, log=None):
    """Add the level's actors and groups; heroes become the document's start."""
    # data is raw .lvl bytes, or a wire Level: then every .sec imports too, section-tagged
    is_level = hasattr(data, 'lvl') and hasattr(data, 'secs')
    d = data.lvl if is_level else glvl.parse_lvl(data)
    blocks = [(0, d)] + ([(i + 1, blk) for i, (_name, blk) in enumerate(data.secs)] if is_level else [])
    doc.settings['template_level'] = stem
    doc.settings['level_stem'] = stem
    sf = glvl.top_kv(d, 'set-filename')
    if sf is not None:
        doc.settings['set_name'] = sf.value.rsplit('.', 1)[0]
    if is_level:
        doc.settings['sec_names'] = [name for name, _blk in data.secs]
    added = []
    groups_count = [0]
    for section_index, block in blocks:
        _import_actor_block(doc, block, stem, section_index, added, groups_count)
    doc.settings['keep_script'] = True
    if log:
        log('imported %d actors and %d groups from %s.lvl%s'
            % (len(added), groups_count[0], stem, ' + %d .sec' % (len(blocks) - 1) if len(blocks) > 1 else ''))
    doc.notify('reset')
    return added

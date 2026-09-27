"""The scene tree and the document that owns it.

Nodes mirror what the game files hold: sections with meshes and terrain, lights
and probes, the skybox and navmesh settings on the set side; actors, actor
groups, component instances and triggers on the level side. Nothing here
imports Qt, so the whole model runs under pytest.
"""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import base64
import collections
import json
import os
import uuid

import numpy as np

from .mathutil import F32, compose, identity

VERSION = 1


def _b64(arr, dtype):
    a = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
    return {'dtype': np.dtype(dtype).str, 'shape': list(a.shape),
            'data': base64.b64encode(a.tobytes()).decode('ascii')}


def _unb64(d):
    return np.frombuffer(base64.b64decode(d['data']), dtype=np.dtype(d['dtype'])).reshape(d['shape']).copy()


class Node:
    kind = 'node'
    transformable = True

    def __init__(self, name='', nid=None):
        self.id = nid or uuid.uuid4().hex[:12]
        self.name = name
        self.parent = None
        self.children = []
        self.visible = True
        self.locked = False
        self.pos = np.zeros(3, F32)
        self.orient = np.zeros(3, F32)        # yaw, pitch, roll in degrees
        self.scale = np.ones(3, F32)
        self.props = {}

    # -- tree -------------------------------------------------------------------
    def add(self, child, index=None):
        if child.parent is not None:
            child.parent.remove(child)
        child.parent = self
        if index is None or index >= len(self.children):
            self.children.append(child)
        else:
            self.children.insert(index, child)
        return child

    def remove(self, child):
        self.children.remove(child)
        child.parent = None
        return child

    def index_in_parent(self):
        return self.parent.children.index(self) if self.parent else 0

    def iter_descendants(self, kind=None):
        for c in self.children:
            if kind is None or c.kind == kind:
                yield c
            yield from c.iter_descendants(kind)

    def find(self, nid):
        if self.id == nid:
            return self
        for c in self.children:
            hit = c.find(nid)
            if hit is not None:
                return hit
        return None

    def ancestors(self):
        n = self.parent
        while n is not None:
            yield n
            n = n.parent

    def path(self):
        parts = [self.name]
        for a in self.ancestors():
            if a.parent is not None:
                parts.append(a.name)
        return '/'.join(reversed(parts))

    def section(self):
        """The section this node lives in, or None."""
        n = self
        while n is not None:
            if n.kind == 'section':
                return n
            n = n.parent
        return None

    # -- transforms ---------------------------------------------------------------
    def matrix(self):
        if not self.transformable:
            return identity()
        return compose(self.pos, self.orient, self.scale)

    def world_matrix(self):
        m = self.matrix()
        p = self.parent
        while p is not None:
            if p.transformable and p.kind not in ('root', 'folder'):
                m = p.matrix() @ m
            p = p.parent
        return m

    def world_pos(self):
        return self.world_matrix()[:3, 3].copy()

    def set_transform(self, pos=None, orient=None, scale=None):
        if pos is not None:
            self.pos = np.array(pos, F32)
        if orient is not None:
            self.orient = np.array(orient, F32)
        if scale is not None:
            self.scale = np.array(scale, F32)

    # -- serialisation ------------------------------------------------------------
    def to_dict(self):
        d = {'kind': self.kind, 'id': self.id, 'name': self.name,
             'visible': self.visible, 'locked': self.locked,
             'pos': [float(v) for v in self.pos], 'orient': [float(v) for v in self.orient],
             'scale': [float(v) for v in self.scale], 'props': self.props_to_dict(),
             'children': [c.to_dict() for c in self.children]}
        return d

    def props_to_dict(self):
        return json.loads(json.dumps(self.props, default=_json_default))

    def load_props(self, props):
        self.props = props

    @classmethod
    def from_dict(cls, d):
        klass = NODE_CLASSES.get(d.get('kind', 'node'), Node)
        n = klass(d.get('name', ''), d.get('id'))
        n.visible = d.get('visible', True)
        n.locked = d.get('locked', False)
        n.pos = np.array(d.get('pos', (0, 0, 0)), F32)
        n.orient = np.array(d.get('orient', (0, 0, 0)), F32)
        n.scale = np.array(d.get('scale', (1, 1, 1)), F32)
        n.load_props(d.get('props', {}))
        for c in d.get('children', []):
            n.add(Node.from_dict(c))
        return n

    def __repr__(self):
        return '<%s %s>' % (self.kind, self.name)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(repr(o))


class FolderNode(Node):
    kind = 'folder'
    transformable = False


class SectionNode(Node):
    """A portal cell: a box the player can be in, owning meshes and terrain.
    The box is the volume the engine consults this section's collision in."""
    kind = 'section'

    def __init__(self, name='', nid=None):
        Node.__init__(self, name, nid)
        self.props = {'lo': [-50.0, -10.0, -50.0], 'hi': [50.0, 30.0, 50.0]}

    @property
    def lo(self):
        return np.array(self.props['lo'], F32)

    @property
    def hi(self):
        return np.array(self.props['hi'], F32)

    def set_bounds(self, lo, hi):
        self.props['lo'] = [float(v) for v in lo]
        self.props['hi'] = [float(v) for v in hi]

    def contains(self, p):
        p = np.asarray(p, dtype=F32)
        return bool(np.all(p >= self.lo) and np.all(p <= self.hi))

    def center(self):
        return (self.lo + self.hi) / 2.0

    def size(self):
        return self.hi - self.lo


class MeshNode(Node):
    """Authored render geometry. `mesh` is a geometry_mesh.EditMesh; `materials`
    is the slot list its face material indices point into."""
    kind = 'mesh'

    def __init__(self, name='', nid=None):
        Node.__init__(self, name, nid)
        self.mesh = None
        self.props = {'materials': [], 'collide': True, 'walkable': True,
                      'double_sided': False, 'surface_mask': 0, 'uv_scale': 8.0}

    def props_to_dict(self):
        d = Node.props_to_dict(self)
        if self.mesh is not None:
            d['geometry'] = self.mesh.to_dict()
        return d

    def load_props(self, props):
        props = dict(props)
        geo = props.pop('geometry', None)
        self.props = props
        if geo is not None:
            from .geometry_mesh import EditMesh
            self.mesh = EditMesh.from_dict(geo)

    @property
    def materials(self):
        return self.props.setdefault('materials', [])


class TerrainNode(Node):
    """A heightmap patch; `terrain` is a geometry_terrain.Terrain."""
    kind = 'terrain'

    def __init__(self, name='', nid=None):
        Node.__init__(self, name, nid)
        self.terrain = None
        self.props = {'materials': [], 'collide': True, 'walkable': True, 'uv_scale': 8.0}

    def props_to_dict(self):
        d = Node.props_to_dict(self)
        if self.terrain is not None:
            d['heightmap'] = self.terrain.to_dict()
        return d

    def load_props(self, props):
        props = dict(props)
        hm = props.pop('heightmap', None)
        self.props = props
        if hm is not None:
            from .geometry_terrain import Terrain
            self.terrain = Terrain.from_dict(hm)

    @property
    def materials(self):
        return self.props.setdefault('materials', [])


class LightNode(Node):
    kind = 'light'

    def __init__(self, name='', nid=None):
        Node.__init__(self, name, nid)
        self.props = {'color': [255, 255, 255], 'radius': 30.0, 'gain': 1.0,
                      'shadow': False}

    @property
    def color(self):
        return tuple(int(c) for c in self.props['color'])

    @property
    def radius(self):
        return float(self.props['radius'])


class ProbeNode(Node):
    kind = 'probe'

    def __init__(self, name='', nid=None):
        Node.__init__(self, name, nid)
        self.props = {'direction': [0.0, -1.0, 0.0]}


class ActorNode(Node):
    """One `.lvl` actor: a class, the fields of its block, and the shipped actor
    it was cloned from. pos/orient are the real fields, mirrored for the tree."""
    kind = 'actor'

    def __init__(self, name='', nid=None):
        Node.__init__(self, name, nid)
        self.props = {'cls': 'CProp', 'fields': collections.OrderedDict(),
                      'template': None, 'template_level': None, 'model': None,
                      'solid': False, 'created': True}

    @property
    def cls(self):
        return self.props['cls']

    @property
    def fields(self):
        return self.props['fields']

    @property
    def model(self):
        return self.props.get('model')

    def load_props(self, props):
        props = dict(props)
        props['fields'] = collections.OrderedDict(props.get('fields', {}))
        self.props = props


class GroupNode(Node):
    """A `.lvl` actor group: the game's own folder for actors."""
    kind = 'group'
    transformable = False

    def __init__(self, name='', nid=None):
        Node.__init__(self, name, nid)
        self.props = {'members': []}

    @property
    def members(self):
        return self.props.setdefault('members', [])


class ComponentNode(Node):
    """A placed component instance: which definition, and what this instance
    overrides on it."""
    kind = 'component'

    def __init__(self, name='', nid=None):
        Node.__init__(self, name, nid)
        self.props = {'component': None, 'mesh': None, 'slots': {}, 'textures': {},
                      'solid': True, 'process_radius': 250, 'created': True}

    @property
    def component_id(self):
        return self.props.get('component')

    def overrides(self):
        return {'mesh': self.props.get('mesh'), 'slots': dict(self.props.get('slots', {})),
                'textures': {k: dict(v) for k, v in self.props.get('textures', {}).items()}}


class TriggerNode(Node):
    """A box volume with a block program; compiles to a CTrigger and script."""
    kind = 'trigger'

    def __init__(self, name='', nid=None):
        Node.__init__(self, name, nid)
        self.props = {'size': [10.0, 10.0, 10.0], 'once': True, 'enabled': True,
                      'on_enter': [], 'on_leave': [], 'on_state': [], 'state': None}

    @property
    def size(self):
        return np.array(self.props['size'], F32)


class SkyboxNode(Node):
    kind = 'skybox'
    transformable = False

    def __init__(self, name='Skybox', nid=None):
        Node.__init__(self, name, nid)
        self.props = {'layers': [], 'sun': [-0.57735, -0.57735, -0.57735],
                      'fog_rgb': [0, 0, 0], 'fog_a': [0.03, 0.034, 0.042],
                      'fog_b': [0.125, 0.14, 0.165], 'donor': None}


class NavNode(Node):
    kind = 'navmesh'
    transformable = False

    def __init__(self, name='Navmesh', nid=None):
        Node.__init__(self, name, nid)
        self.props = {'cell': 6.0, 'max_step': 2.0, 'agent_height': 8.0,
                      'max_slope_deg': 40.0, 'auto': True}


NODE_CLASSES = {c.kind: c for c in (Node, FolderNode, SectionNode, MeshNode, TerrainNode,
                                    LightNode, ProbeNode, ActorNode, GroupNode,
                                    ComponentNode, TriggerNode, SkyboxNode, NavNode)}


class Document:
    """The whole level: set side, level side, component definitions, settings."""

    def __init__(self, name='untitled'):
        self.path = None
        self.settings = {
            'set_name': name, 'level_stem': name + '1', 'template_level': 'cemetery2',
            'mod_name': name, 'archive_name': 'EDITOR.POD',
            'hero_start': [0.0, 0.0, 0.0], 'hero_yaw': 0.0,
            'cast': ['gb_player', 'gb_zeddemore', 'gb_stantz', 'gb_spengler'],
            'bake': {'ambient': 0.25, 'tint': [1.0, 1.0, 1.0], 'contrast': 2.0,
                     'lightmaps': False, 'tile_size': 256},
            'description': '',
        }
        self.components = collections.OrderedDict()
        self.listeners = []
        self._build_tree()

    def _build_tree(self):
        self.root = FolderNode('root')
        self.root.kind = 'root'
        self.set_folder = self.root.add(FolderNode('Set'))
        self.sections = self.set_folder.add(FolderNode('Sections'))
        self.lights = self.set_folder.add(FolderNode('Lights'))
        self.probes = self.set_folder.add(FolderNode('Light Probes'))
        self.skybox = self.set_folder.add(SkyboxNode())
        self.nav = self.set_folder.add(NavNode())
        self.level_folder = self.root.add(FolderNode('Level'))
        self.actors = self.level_folder.add(FolderNode('Actors'))
        self.groups = self.level_folder.add(FolderNode('Actor Groups'))
        self.components_folder = self.level_folder.add(FolderNode('Components'))
        self.triggers = self.level_folder.add(FolderNode('Triggers'))
        self.folders = {'sections': self.sections, 'lights': self.lights, 'probes': self.probes,
                        'actors': self.actors, 'groups': self.groups,
                        'components': self.components_folder, 'triggers': self.triggers}

    # -- change notification --------------------------------------------------------
    def listen(self, fn):
        self.listeners.append(fn)
        return fn

    def unlisten(self, fn):
        if fn in self.listeners:
            self.listeners.remove(fn)

    def notify(self, event, node=None, **kw):
        for fn in list(self.listeners):
            fn(event, node, **kw)

    # -- queries ----------------------------------------------------------------------
    def find(self, nid):
        return self.root.find(nid)

    def nodes(self, kind=None):
        return list(self.root.iter_descendants(kind))

    def section_list(self):
        return [c for c in self.sections.children if c.kind == 'section']

    def folder_for(self, node):
        return self.folders.get({'section': 'sections', 'light': 'lights', 'probe': 'probes',
                                 'actor': 'actors', 'group': 'groups', 'component': 'components',
                                 'trigger': 'triggers'}.get(node.kind), self.sections)

    def unique_name(self, base):
        names = {n.name for n in self.nodes()}
        if base not in names:
            return base
        i = 2
        while '%s%d' % (base, i) in names:
            i += 1
        return '%s%d' % (base, i)

    def section_at(self, p):
        """The first section whose box holds p, or None."""
        for s in self.section_list():
            if s.contains(p):
                return s
        return None

    # -- mutation (the undo commands call these) -----------------------------------------
    def add_node(self, node, parent=None, index=None):
        parent = parent or self.folder_for(node)
        parent.add(node, index)
        self.notify('added', node)
        return node

    def remove_node(self, node):
        parent = node.parent
        index = node.index_in_parent()
        parent.remove(node)
        self.notify('removed', node, parent=parent, index=index)
        return parent, index

    def set_props(self, node, changes):
        old = {k: node.props.get(k) for k in changes}
        node.props.update(changes)
        self.notify('changed', node, keys=list(changes))
        return old

    def set_transform(self, node, pos=None, orient=None, scale=None):
        old = (node.pos.copy(), node.orient.copy(), node.scale.copy())
        node.set_transform(pos, orient, scale)
        self.notify('transform', node)
        return old

    def geometry_changed(self, node):
        self.notify('geometry', node)

    # -- files -----------------------------------------------------------------------
    def to_dict(self):
        return {'version': VERSION, 'settings': self.settings,
                'components': [c.to_dict() for c in self.components.values()],
                'tree': self.root.to_dict()}

    def save(self, path):
        d = self.to_dict()
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(d, fh, indent=1, default=_json_default)
        os.replace(tmp, path)
        self.path = path
        self.notify('saved')

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as fh:
            d = json.load(fh)
        doc = cls()
        doc.settings.update(d.get('settings', {}))
        from .components import ComponentDef
        for c in d.get('components', []):
            cd = ComponentDef.from_dict(c)
            doc.components[cd.id] = cd
        doc.root = Node.from_dict(d['tree'])
        doc.root.kind = 'root'
        doc._rebind_folders()
        doc.path = path
        return doc

    def _rebind_folders(self):
        byname = {c.name: c for c in self.root.children}
        self.set_folder = byname['Set']
        self.level_folder = byname['Level']
        s = {c.name: c for c in self.set_folder.children}
        l = {c.name: c for c in self.level_folder.children}
        self.sections, self.lights, self.probes = s['Sections'], s['Lights'], s['Light Probes']
        self.skybox, self.nav = s['Skybox'], s['Navmesh']
        self.actors, self.groups = l['Actors'], l['Actor Groups']
        self.components_folder, self.triggers = l['Components'], l['Triggers']
        self.folders = {'sections': self.sections, 'lights': self.lights, 'probes': self.probes,
                        'actors': self.actors, 'groups': self.groups,
                        'components': self.components_folder, 'triggers': self.triggers}

    # -- helpers for whole-scene geometry ---------------------------------------------
    def mesh_nodes(self):
        return [n for n in self.root.iter_descendants() if n.kind in ('mesh', 'terrain')]

    def world_triangles(self, collide_only=False):
        """(verts, tris, node ids) for every authored mesh in world space."""
        vs, ts, owners, base = [], [], [], 0
        for n in self.mesh_nodes():
            geo = n.mesh if n.kind == 'mesh' else (n.terrain.mesh() if n.terrain else None)
            if geo is None or len(geo.faces) == 0:
                continue
            if collide_only and not n.props.get('collide', True):
                continue
            from .mathutil import transform_points
            v = transform_points(n.world_matrix(), geo.verts)
            vs.append(v)
            ts.append(np.asarray(geo.faces, np.int64) + base)
            owners.extend([n.id] * len(geo.faces))
            base += len(v)
        if not vs:
            return np.zeros((0, 3), F32), np.zeros((0, 3), np.int64), []
        return np.concatenate(vs), np.concatenate(ts), owners


def serialize_array(arr, dtype):
    return _b64(arr, dtype)


def deserialize_array(d):
    return _unb64(d)

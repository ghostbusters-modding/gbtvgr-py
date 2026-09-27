"""Assemble a `.lvl` actor table from a document, cloning each actor's block
from a shipped level so every class keeps its full field schema, and keeping
actor-list, Dante-Includes, actor-version-list and Dependencies in lockstep.
"""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import collections
import copy
import os
import re
import struct

from gbtvgr import lvl as glvl
from gbtvgr.lvl import KV, Block, ListF

WINSEP = '\\'
# the hero side of a template level's dependency block
DEP_PATS = ('gb_player', 'gb_zeddemore', 'gb_stantz', 'gb_spengler', 'gb_venkman',
            'ghostbuster', 'weap', 'proton', 'trap', 'civh', 'goggle', 'slimeblower',
            'dirt_explosion')
TRIG_EVENTS = [('onEvent', 'onEvent'), ('risingEdgeEvent', 'onRisingEdge'),
               ('fallingEdgeEvent', 'onFallingEdge'), ('actorEnterEvent', 'onActorEnter'),
               ('actorInsideEvent', 'onActorInside'), ('actorLeaveEvent', 'onActorLeave'),
               ('actorEnterUpStreamEvent', 'onActorEnterUpStream'),
               ('actorEnterDownStreamEvent', 'onActorEnterDownStream'),
               ('actorLeaveUpStreamEvent', 'onActorLeaveUpStream'),
               ('actorLeaveDownStreamEvent', 'onActorLeaveDownStream')]
EVENT_RE = re.compile(r'^[A-Za-z_@][^=]*\(.*\)\s*$')


def esc(path):
    """A path value as the file stores it: every backslash doubled."""
    return path.replace('/', WINSEP).replace(WINSEP, WINSEP * 2)


def vec(v):
    """Written as the float32 the engine will hold, so a value survives a re-read."""
    return '%s, %s, %s' % tuple(glvl.fmt_num(struct.unpack('<f', struct.pack('<f', float(x)))[0])
                                for x in v)


def blank_events(block):
    for c in block.children:
        if isinstance(c, KV) and EVENT_RE.match(c.value):
            c.value = '""'


def set_field(block, key, value, create=True):
    # a bare empty value is a fatal read in the engine; retail writes ""
    if value == '':
        value = '""'
    for c in block.children:
        if isinstance(c, KV) and c.key == key:
            c.value = value
            return True
    if create:
        block.children.append(KV(key, value))
    return False


def get_field(block, key):
    for c in block.children:
        if isinstance(c, KV) and c.key == key:
            return c.value
    return None


def prop_block(name, model, pos, orient, solid, created=True):
    """The 11-field CProp schema every shipped CProp carries, in order."""
    b = Block(name)
    for k, v in [('name', name), ('createStatus', '1' if created else '0'),
                 ('pos', vec(pos)), ('orient', vec(orient)), ('processRadius', '250'),
                 ('damageFilterEvent', '""'), ('restOnActorValid', '0'),
                 ('onSnaredStatusChangedEvent', '""'), ('modelInstance', esc(model)),
                 ('useCollisionParts', '1' if solid else '0'), ('propFlags', '0')]:
        b.children.append(KV(k, v))
    return b


class LevelBuild:
    def __init__(self):
        self.data = b''
        self.secs = []            # [(name, bytes)] for actors/groups tagged with a section
        self.report = []
        self.warnings = []
        self.errors = []
        self.actor_names = []

    @property
    def ok(self):
        return not self.errors


class TemplateLibrary:
    """Shipped actor blocks by class, searched across every level in the game."""

    def __init__(self, library):
        self.library = library
        self.docs = collections.OrderedDict()

    def doc(self, stem):
        if stem not in self.docs:
            raw = self.library.read_level(stem) if self.library else None
            self.docs[stem] = glvl.parse_lvl(raw) if raw else None
        return self.docs[stem]

    def stems(self):
        return self.library.level_stems() if self.library else []

    def find(self, cls=None, name=None, prefer=None):
        """(block, source stem) for an actor by name or class."""
        order = [prefer] + [s for s in self.stems() if s != prefer] if prefer else self.stems()
        for stem in order:
            d = self.doc(stem)
            if d is None:
                continue
            ab = glvl.top_block(d, 'actors')
            cmap = glvl.actor_class_map(d)
            if ab is None:
                continue
            for a in ab.children:
                if not isinstance(a, Block):
                    continue
                if name and a.tag == name:
                    return a, stem
                if name is None and cls and cmap.get(a.tag) == cls:
                    return a, stem
        return None, None

    def class_version(self, cls):
        for stem in self.stems():
            d = self.doc(stem)
            if d is None:
                continue
            avl = glvl.top_block(d, 'actor-version-list')
            for c in (avl.children if avl else []):
                if isinstance(c, KV) and c.key == cls:
                    return c.value
        return None

    def civh_for(self, cit_dep, prefer=None):
        order = [prefer] + [s for s in self.stems() if s != prefer] if prefer else self.stems()
        for stem in order:
            d = self.doc(stem)
            if d is None:
                continue
            hit = glvl.civh_family_entry(cit_dep, glvl.dep_set(d))
            if hit:
                return hit
        return None

    def group_extern_type(self, stem):
        d = self.doc(stem)
        if d is None:
            return 'CActorGroup'
        groups = {c.key for c in (glvl.top_block(d, 'actor-groups') or Block('x')).children
                  if isinstance(c, ListF)}
        di = glvl.top_list(d, 'Dante-Includes')
        for _t, line in (di.items if di else []):
            m = re.match(r'extern (\S+) (\S+);', line)
            if m and m.group(2) in groups:
                return m.group(1)
        return 'CActorGroup'


def new_root(tpl, set_name):
    """The empty .lvl skeleton, cinemat schema copied from the template level."""
    root = Block(None)
    root.children.append(KV('version', '19'))
    root.children.append(KV('set-filename', set_name + '.pst'))
    root.children.extend([ListF('Dependencies', []), ListF('Dante-Includes', []),
                          Block('actor-version-list'), Block('actor-list'),
                          Block('list-of-heros'), Block('actors'), Block('actor-groups'),
                          Block('selection-sets-for-editor-only'), Block('section-list')])
    for tag in ('cinemat-version-list',):
        b = glvl.top_block(tpl, tag) if tpl is not None else None
        root.children.append(copy.deepcopy(b) if b else Block(tag))
    root.children.append(Block('cinemat-list'))
    root.children.append(Block('cinemats'))
    return root


def template_deps(tpl):
    """The hero-side Dependencies lines of a template level, first spelling of each."""
    out, seen = [], set()
    tpl_deps = glvl.top_list(tpl, 'Dependencies')
    for _t, line in (tpl_deps.items if tpl_deps else []):
        low = line.lower()
        if any(p in low for p in DEP_PATS) and line not in seen:
            seen.add(line)
            out.append(line)
    return out


def actor_deps(block, tl, have, prefer=None):
    """(new Dependencies lines, warnings) an actor block needs beyond `have`."""
    deps, warnings = [], []
    seen = set(have)

    def add(line):
        if line not in seen:
            seen.add(line)
            deps.append(line)

    cit = get_field(block, 'charInfoFilename')
    if cit and cit != '""':
        depform = glvl.dep_path_for_cit(cit)
        add(depform)
        civh = glvl.civh_family_entry(depform, seen) or tl.civh_for(depform, prefer=prefer)
        if civh:
            add(civh)
        else:
            warnings.append('%s: no .civh found for %s' % (block.tag, cit))
    model = get_field(block, 'modelInstance')
    if model and model != '""' and '.dfm' not in model.lower():
        add('models' + WINSEP * 2 + os.path.splitext(model)[0] + '.smf')
    return deps, warnings


def emit_section_list(doc, names):
    """<section-list> of a parsed root, created after the editor block when absent."""
    sl = glvl.top_block(doc, 'section-list')
    if sl is None:
        sl = Block('section-list')
        kids = doc.children
        anchor = glvl.top_block(doc, 'selection-sets-for-editor-only')
        if anchor is not None:
            kids.insert(kids.index(anchor) + 1, sl)
        else:
            cine = glvl.top_block(doc, 'cinemat-version-list')
            kids.insert(kids.index(cine) if cine is not None else len(kids), sl)
    sl.children = [KV('Section_%d' % (i + 1), name) for i, name in enumerate(names)]
    return sl


def add_actor_block(doc, name, cls, block, tl, prefer=None):
    """Append one actor to a parsed .lvl or .sec root, its four tables kept in step."""
    al = glvl.top_block(doc, 'actor-list')
    if any(isinstance(c, KV) and c.key == name for c in al.children):
        raise ValueError('duplicate actor name %s' % name)
    block.tag = name
    set_field(block, 'name', name)
    glvl.top_block(doc, 'actors').children.append(block)
    al.children.append(KV(name, cls))
    glvl.top_list(doc, 'Dante-Includes').items.append((0, 'extern %s %s;' % (cls, name)))
    avl = glvl.top_block(doc, 'actor-version-list')
    warnings = []
    if not any(isinstance(c, KV) and c.key == cls for c in avl.children):
        ver = tl.class_version(cls)
        if ver is None:
            warnings.append('class %s has no actor-version-list entry in any shipped level' % cls)
        else:
            avl.children.append(KV(cls, ver))
    deps = glvl.top_list(doc, 'Dependencies')
    new, warn = actor_deps(block, tl, {line for _t, line in deps.items}, prefer)
    deps.items.extend((0, line) for line in new)
    return warnings + warn


def clone_actor(tl, cls, name, template=None, prefer=None, fields=None, pos=None,
                orient=None, created=True, keep_events=False):
    """A shipped actor block copied and re-labelled; fields land after the events are
    blanked, so an override can bind an event. Returns (block, source stem)."""
    blk, src = tl.find(cls=cls, name=template, prefer=prefer)
    if blk is None and template:
        raise LookupError('%s: no shipped actor %s to clone' % (name, template))
    if blk is None:
        raise LookupError('%s: no shipped %s actor to clone' % (name, cls))
    have = glvl.actor_class_map(tl.doc(src)).get(blk.tag)
    if have != cls:
        raise LookupError('%s: %s in %s is a %s, not a %s' % (name, blk.tag, src, have, cls))
    b = copy.deepcopy(blk)
    b.tag = name
    if not keep_events:
        blank_events(b)
    set_field(b, 'name', name)
    for k, v in (fields or {}).items():
        if k != 'name':
            set_field(b, k, v)
    if pos is not None:
        set_field(b, 'pos', vec(pos))
    if orient is not None:
        set_field(b, 'orient', vec(orient))
    set_field(b, 'createStatus', '1' if created else '0', create=False)
    return b, src


def _new_layer():
    """A fresh actor-table layer: same top-level shape as the main .lvl root,
    minus heroes/cinemats which only the main .lvl carries."""
    root = Block(None)
    root.children.append(KV('version', '19'))
    deps = ListF('Dependencies', [])
    incl = ListF('Dante-Includes', [])
    avl = Block('actor-version-list')
    al = Block('actor-list')
    actors = Block('actors')
    groups = Block('actor-groups')
    root.children.extend([deps, incl, avl, al, actors, groups])
    return {'root': root, 'deps': deps, 'incl': incl, 'avl': avl, 'al': al,
            'actors': actors, 'groups': groups, 'dep_seen': set(),
            'classes': collections.OrderedDict()}


def compile_level(doc, library, handlers=None, model_refs=None, log=None):
    """doc -> LevelBuild. `handlers` maps (actor name, field) -> event value from
    the script compiler; `model_refs` maps a component node id -> model ref."""
    # an actor/group with props['section'] set routes into out.secs, not the main .lvl
    from .mathutil import matrix_to_orient      # numpy stays out of the wire-layer callers
    out = LevelBuild()
    s = doc.settings
    tl = TemplateLibrary(library)
    stem = s.get('template_level', 'cemetery2')
    tpl = tl.doc(stem)
    if tpl is None:
        out.errors.append('template level %s is not in the archives' % stem)
        return out
    handlers = handlers or {}
    model_refs = model_refs or {}

    root = new_root(tpl, s.get('set_name', 'level'))
    deps = glvl.top_list(root, 'Dependencies')
    incl = glvl.top_list(root, 'Dante-Includes')
    avl = glvl.top_block(root, 'actor-version-list')
    al = glvl.top_block(root, 'actor-list')
    heroes = glvl.top_block(root, 'list-of-heros')
    actors = glvl.top_block(root, 'actors')
    groups = glvl.top_block(root, 'actor-groups')

    dep_seen = set()
    classes = collections.OrderedDict()
    layer0 = {'root': root, 'deps': deps, 'incl': incl, 'avl': avl, 'al': al,
              'actors': actors, 'groups': groups, 'dep_seen': dep_seen, 'classes': classes}
    layers = {0: layer0}

    def layer_for(section):
        sec = section or 0
        if sec not in layers:
            layers[sec] = _new_layer()
        return layers[sec]

    def add_dep(line, layer=layer0):
        if line not in layer['dep_seen']:
            layer['dep_seen'].add(line)
            layer['deps'].items.append((0, line))

    for line in template_deps(tpl):
        add_dep(line)
    # what a spawned character pulls in beyond its .cit/.civh (fx, physics, decals)
    for line in s.get('extra_deps') or ():
        add_dep(esc(line))

    def add_actor(name, cls, block, layer=layer0):
        if name in classes:
            out.errors.append('duplicate actor name %s' % name)
            return
        classes[name] = cls
        layer['classes'][name] = cls
        layer['actors'].children.append(block)
        layer['al'].children.append(KV(name, cls))
        layer['incl'].items.append((0, 'extern %s %s;' % (cls, name)))

    # -- heroes ------------------------------------------------------------------------
    gb0, _src = tl.find(name='Ghostbuster0', prefer=stem)
    if gb0 is None:
        out.errors.append('no Ghostbuster0 in %s' % stem)
        return out
    b = copy.deepcopy(gb0)
    blank_events(b)
    for k, v in (s.get('hero_events') or {}).items():
        set_field(b, k, v)
    set_field(b, 'pos', vec(s.get('hero_start', (0, 0, 0))))
    set_field(b, 'orient', vec((s.get('hero_yaw', 0.0), 0.0, 0.0)))
    set_field(b, 'restOnActorValid', '0')
    add_actor('Ghostbuster0', 'CGhostbuster', b)
    heroes.children.append(KV('hero_0', 'Ghostbuster0'))
    cmap = glvl.actor_class_map(tpl)
    ab = glvl.top_block(tpl, 'actors')
    companions = [a for a in ab.children if isinstance(a, Block)
                  and cmap.get(a.tag) == 'CGhostbuster' and a.tag != 'Ghostbuster0']
    hx, hy, hz = s.get('hero_start', (0, 0, 0))
    listed = s.get('companions') or {}
    if not isinstance(listed, dict):
        listed = {row[0]: tuple(row[1:]) for row in listed}
    for i, a in enumerate(companions[:4]):
        b = copy.deepcopy(a)
        blank_events(b)
        set_field(b, 'pos', vec(listed.get(a.tag, (hx + 4.0 * (i + 1), hy, hz - 3.0))))
        set_field(b, 'orient', vec((s.get('hero_yaw', 0.0), 0.0, 0.0)))
        set_field(b, 'restOnActorValid', '0')
        add_actor(a.tag, 'CGhostbuster', b)

    # -- actors ---------------------------------------------------------------------------
    for node in doc.nodes('actor'):
        cls = node.props.get('cls', 'CProp')
        tname = node.props.get('template')
        tsrc = node.props.get('template_level') or stem
        blk, src = tl.find(cls=cls, name=tname, prefer=tsrc)
        if blk is None and tname:
            blk, src = tl.find(cls=cls, prefer=tsrc)
        if blk is None:
            if cls == 'CProp' and node.model:
                blk, src = prop_block(node.name, node.model, node.world_pos(), node.orient,
                                      node.props.get('solid', False),
                                      node.props.get('created', True)), None
            else:
                out.errors.append('%s: no shipped %s actor to clone' % (node.name, cls))
                continue
        b = copy.deepcopy(blk)
        b.tag = node.name
        if not node.props.get('keep_events'):
            blank_events(b)
        set_field(b, 'name', node.name)
        for k, v in node.fields.items():
            if k in ('name',):
                continue
            set_field(b, k, v)
        set_field(b, 'pos', vec(node.world_pos()))
        set_field(b, 'orient', vec(matrix_to_orient(node.world_matrix())))
        set_field(b, 'createStatus', '1' if node.props.get('created', True) else '0', create=False)
        if node.model and get_field(b, 'modelInstance') is not None:
            set_field(b, 'modelInstance', esc(node.model))
        for (an, field), value in handlers.items():
            if an == node.name:
                set_field(b, field, value)
        lay = layer_for(node.props.get('section'))
        add_actor(node.name, cls, b, lay)
        cit = get_field(b, 'charInfoFilename')
        if cit and cit != '""':
            depform = glvl.dep_path_for_cit(cit)
            add_dep(depform, lay)
            civh = glvl.civh_family_entry(depform, lay['dep_seen']) or tl.civh_for(depform, prefer=src or stem)
            if civh:
                add_dep(civh, lay)
            else:
                out.warnings.append('%s: no .civh found for %s' % (node.name, cit))
        model = get_field(b, 'modelInstance')
        if model and model != '""' and '.dfm' not in model.lower():
            add_dep('models' + WINSEP * 2 + os.path.splitext(model)[0] + '.smf', lay)

    # -- triggers ---------------------------------------------------------------------------
    trig_tpl, _src = tl.find(cls='CTrigger', prefer=stem)
    for node in doc.nodes('trigger'):
        tpl_blk = trig_tpl
        if node.props.get('template'):
            tpl_blk, _src = tl.find(cls='CTrigger', name=node.props['template'],
                                    prefer=node.props.get('template_level') or stem)
        if tpl_blk is None:
            out.errors.append('%s: no shipped CTrigger to clone' % node.name)
            continue
        b = copy.deepcopy(tpl_blk)
        b.tag = node.name
        blank_events(b)
        set_field(b, 'name', node.name)
        set_field(b, 'createStatus', '1' if node.props.get('enabled', True) else '0')
        set_field(b, 'pos', vec(node.world_pos()))
        set_field(b, 'orient', vec((0.0, 0.0, 0.0)))
        set_field(b, 'restOnActorValid', '0')
        set_field(b, 'shape', '0')
        set_field(b, 'triggerSize', vec(node.props.get('size', (10, 10, 10))))
        set_field(b, 'oneShot', '1' if node.props.get('once', True) else '0')
        set_field(b, 'activationType', '1')
        for field, suffix in TRIG_EVENTS:
            value = handlers.get((node.name, field))
            if value:
                set_field(b, field, value)
        for k, v in (node.props.get('fields') or {}).items():
            if k != 'name':
                set_field(b, k, v)
        add_actor(node.name, 'CTrigger', b)

    # -- components -------------------------------------------------------------------------
    for node in doc.nodes('component'):
        cd = doc.components.get(node.component_id)
        ref = model_refs.get(node.id) or (cd.mesh if cd else None)
        if not ref:
            out.errors.append('%s: component has no mesh' % node.name)
            continue
        b = prop_block(node.name, ref, node.world_pos(), matrix_to_orient(node.world_matrix()),
                       node.props.get('solid', True), node.props.get('created', True))
        set_field(b, 'processRadius', str(node.props.get('process_radius', 250)))
        add_actor(node.name, 'CProp', b)
        add_dep('models' + WINSEP * 2 + esc(ref) + '.smf')

    # -- groups -------------------------------------------------------------------------------
    gtype = tl.group_extern_type(stem)
    for node in doc.nodes('group'):
        members = [m for m in node.members if m in classes]
        missing = [m for m in node.members if m not in classes]
        if missing:
            out.warnings.append('%s: members not in the level: %s' % (node.name, ', '.join(missing)))
        lay = layer_for(node.props.get('section'))
        lay['groups'].children.append(ListF(node.name, [(0, m) for m in members]))
        lay['incl'].items.append((0, 'extern %s %s;' % (gtype, node.name)))

    # -- versions -------------------------------------------------------------------------------
    for lay in layers.values():
        for cls in collections.OrderedDict.fromkeys(lay['classes'].values()):
            ver = tl.class_version(cls)
            if ver is None:
                out.warnings.append('class %s has no actor-version-list entry in any shipped level' % cls)
                continue
            lay['avl'].children.append(KV(cls, ver))

    out.actor_names = list(classes)
    sec_names = s.get('sec_names') or []
    for sec_index, lay in sorted(layers.items()):
        if sec_index == 0:
            continue
        name = sec_names[sec_index - 1] if sec_index - 1 < len(sec_names) else '%s_sec%d' % (stem, sec_index)
        sec_bytes = glvl.serialize_lvl(lay['root'])
        out.secs.append((name, sec_bytes))
        out.errors.extend('%s.sec: %s' % (name, e) for e in validate(sec_bytes))
    emit_section_list(root, [name for name, _b in out.secs])
    out.data = glvl.serialize_lvl(root)
    out.errors.extend(validate(out.data))
    out.report.append('%s.lvl: %d actors, %d dependencies, %d groups'
                      % (s.get('level_stem', 'level'), len(classes), len(deps.items),
                         len(groups.children)))
    if out.secs:
        out.report.append('%d .sec layer(s): %s' % (len(out.secs), ', '.join(n for n, _b in out.secs)))
    if log:
        for line in out.report:
            log(line)
    return out


def validate(data):
    """The checks `gbtvgr lvl validate` makes, as a list of problems."""
    problems = []
    doc = glvl.parse_lvl(data)
    for ln, line in enumerate(data.decode('latin1').split('\r\n'), 1):
        if '\\' in line.replace('\\\\', ''):
            problems.append('line %d: single backslash: %r' % (ln, line.strip()[:70]))
        if line.rstrip().endswith('=') and '\t' in line:
            problems.append('line %d: empty value (the engine fails the read): %r'
                            % (ln, line.strip()[:70]))
    deps = glvl.dep_set(doc)
    names = glvl.actor_class_map(doc)
    ab = glvl.top_block(doc, 'actors')
    seen = set()
    for a in (ab.children if ab else []):
        if not isinstance(a, Block):
            continue
        if a.tag in seen:
            problems.append('duplicate actor block <%s>' % a.tag)
        seen.add(a.tag)
        if a.tag not in names:
            problems.append('<%s> has no actor-list entry' % a.tag)
        f = glvl.actor_fields(a)
        cit = f.get('charInfoFilename')
        if cit and cit != '""':
            depform = glvl.dep_path_for_cit(cit)
            if depform not in deps:
                problems.append('%s: missing Dependencies entry %s' % (a.tag, depform))
            elif glvl.civh_family_entry(depform, deps) is None:
                problems.append('%s: no sibling .civh for %s' % (a.tag, depform))
        for vk in ('pos', 'orient'):
            if vk in f and not glvl.looks_like_vector(f[vk]):
                problems.append('%s: malformed %s = %r' % (a.tag, vk, f[vk]))
    for n in names:
        if n not in seen:
            problems.append('actor-list entry %s has no block' % n)
    if glvl.serialize_lvl(doc) != data:
        problems.append('the level does not round-trip through the codec')
    return problems

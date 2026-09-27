"""Placing things into a level at the wire layer: the floor off a section's BVT, a
grafted section, props and actors cloned from shipped rows, the squad."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import os
import struct

from .. import lvl as glvl
from ..mesh.bvt import bvt_collect
from . import levelfile, placement, transplant, wire

STAND = 0.25              # a character's pos sits this far above the collision floor
# hero-local (right, back) offsets in ft; local +Z is the way a character faces
SQUAD_LAYOUT = {'Ray': (8.0, -4.0), 'Egon': (4.0, -4.0), 'Winston': (-4.0, -4.0),
                'Peter': (-8.0, -4.0)}


class PlaceError(Exception):
    pass


# --- sections ------------------------------------------------------------------------
def section_bbox(m, index):
    return struct.unpack('<6f', m['sections'][index]['bbox'])


def section_at(m, x, y, z):
    """Index of the first set section whose bbox holds the point, or None."""
    for i in range(len(m['sections'])):
        b = section_bbox(m, i)
        if b[0] <= x <= b[3] and b[1] <= y <= b[4] and b[2] <= z <= b[5]:
            return i
    return None


class Floor:
    """Height queries against one section's collision tree, collected once."""

    def __init__(self, m, section_index, below=None):
        sec = m['sections'][section_index]
        if sec['bvt'] is None:
            raise PlaceError('section %d carries no BVT' % section_index)
        self.verts, self.tris, _surf, _flags = bvt_collect(sec['bvt'])
        self.bbox = struct.unpack('<6f', sec['bbox'])
        self.below = below

    def at(self, x, z):
        """The highest up-facing triangle under (x, z), capped by `below`, or None."""
        best = None
        for a, b, c in self.tris:
            p0, p1, p2 = self.verts[a], self.verts[b], self.verts[c]
            d = (p1[2] - p2[2]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[2] - p2[2])
            if abs(d) < 1e-9:
                continue
            l0 = ((p1[2] - p2[2]) * (x - p2[0]) + (p2[0] - p1[0]) * (z - p2[2])) / d
            l1 = ((p2[2] - p0[2]) * (x - p2[0]) + (p0[0] - p2[0]) * (z - p2[2])) / d
            l2 = 1.0 - l0 - l1
            if l0 < -1e-6 or l1 < -1e-6 or l2 < -1e-6:
                continue
            y = l0 * p0[1] + l1 * p1[1] + l2 * p2[1]
            up = (p1[2] - p0[2]) * (p2[0] - p0[0]) - (p1[0] - p0[0]) * (p2[2] - p0[2])
            if up > 0 and (self.below is None or y < self.below) and (best is None or y > best):
                best = y
        return best

    __call__ = at

    def need(self, what, x, z):
        """at(), but a miss or a point outside the section bbox is an error."""
        y = self.at(x, z)
        if y is None:
            raise PlaceError('%s at (%g, %g) has no collision floor under it' % (what, x, z))
        if not (self.bbox[0] <= x <= self.bbox[3] and self.bbox[2] <= z <= self.bbox[5]):
            raise PlaceError('%s at (%g, %g) is outside the section bbox' % (what, x, z))
        return y


def floor_at(m, section_index, x, z, below=None):
    return Floor(m, section_index, below).at(x, z)


# --- a grafted section ---------------------------------------------------------------------
def place_section(host_m, donor_m, donor_index, offset, lightmap_dir=None, split=None):
    """transplant.append_section plus the index the new section got: (host, staged, index)."""
    host_m, staged = transplant.append_section(host_m, donor_m, donor_index, offset,
                                               lightmap_dir=lightmap_dir, split=split)
    return host_m, staged, len(host_m['sections']) - 1


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(data)
    # a networked drive can drop a write on the floor; read it back before trusting it
    with open(path, 'rb') as fh:
        if fh.read() != data:
            raise PlaceError('%s did not read back as written' % path)


def stage_files(staged, library, out_dir):
    """Copy the (src, dst) tiles and cubes of a graft out of the POD chain; returns the paths."""
    paths = []
    for src, dst in staged:
        data = library.read(src)
        if data is None:
            raise PlaceError('%s is not in the archives' % src)
        path = os.path.join(out_dir, *dst.split('\\'))
        _write(path, data)
        paths.append(path)
    return paths


# --- actors ---------------------------------------------------------------------------------
def _templates(library):
    if isinstance(library, levelfile.TemplateLibrary):
        return library
    return levelfile.TemplateLibrary(library)


def layer_doc(level, section_index=0):
    """The parsed root an actor lands in: 0 is the .lvl, N the N-th .sec."""
    if section_index == 0:
        return level.lvl
    if not 1 <= section_index <= len(level.secs):
        raise PlaceError('level %s has no .sec layer %d' % (level.stem, section_index))
    return level.secs[section_index - 1][1]


def new_level(stem, set_name, library, template_level='abyss', bst=None):
    """An empty wire.Level whose .lvl carries the template's hero-side Dependencies."""
    tl = _templates(library)
    tpl = tl.doc(template_level)
    if tpl is None:
        raise PlaceError('template level %s is not in the archives' % template_level)
    root = levelfile.new_root(tpl, set_name)
    deps = glvl.top_list(root, 'Dependencies')
    deps.items.extend((0, line) for line in levelfile.template_deps(tpl))
    return wire.Level(stem, root, [], set_name, bst)


def place_actor(level, cls, template_name, name, pos, fields=None, section_index=0,
                *, library, template_level=None, yaw=0.0, created=True, keep_events=False,
                warnings=None):
    """Clone the named shipped actor (or the first of its class when template_name is
    None) into the level; `fields` overrides land after the clone. Returns the block."""
    tl = _templates(library)
    try:
        blk, src = levelfile.clone_actor(tl, cls, name, template=template_name,
                                         prefer=template_level, fields=fields, pos=pos,
                                         orient=(yaw, 0.0, 0.0), created=created,
                                         keep_events=keep_events)
    except LookupError as exc:
        raise PlaceError(str(exc))
    doc = layer_doc(level, section_index)
    try:
        warn = levelfile.add_actor_block(doc, name, cls, blk, tl, prefer=src or template_level)
    except ValueError as exc:
        raise PlaceError(str(exc))
    if warn and warnings is None:
        raise PlaceError('; '.join(warn))
    if warn:
        warnings.extend(warn)
    return blk


def resolve_yaw(smb_dir, model_ref, yaw):
    """A number is the yaw; (dx, dz) turns the model's front that way; a wall side letter
    (N/E/S/W) turns its front into the room."""
    if isinstance(yaw, (int, float)):
        return float(yaw)
    if isinstance(yaw, str):
        yaw = placement.INWARD[yaw.upper()]
    return float(placement.yaw_to_face_dir(smb_dir, model_ref, tuple(yaw)))


def place_prop(level, model_ref, pos, yaw, name, template_library, smb_dir, section_index=0,
               fields=None, template_name=None, template_level=None, solid=None, created=True,
               bst_section=None):
    """A CProp cloned from a shipped one, its real .smb box checked against the set section
    holding pos. `solid` None means: solid when the model carries collision meshes."""
    me = placement.smb_meta_dir(smb_dir, model_ref)
    if me is None:
        raise PlaceError('%s: no .smb for %r under %s' % (name, model_ref, smb_dir))
    if level.bst is None:
        raise PlaceError('%s: the level has no set loaded to check bounds against' % name)
    x, y, z = pos
    yaw = resolve_yaw(smb_dir, model_ref, yaw)
    si = section_at(level.bst, x, y, z) if bst_section is None else bst_section
    if si is None:
        raise PlaceError('%s at (%g, %g, %g) is inside no set section' % (name, x, y, z))
    bbox = level.bst['sections'][si]['bbox']
    if not placement.in_section(smb_dir, model_ref, x, y, z, yaw, bbox):
        raise PlaceError('%s: %s at (%g, %g, %g) yaw %g pokes out of set section %d'
                         % (name, model_ref, x, y, z, yaw, si))
    f = {'modelInstance': levelfile.esc(model_ref),
         'useCollisionParts': '1' if (me['coll'] > 0 if solid is None else solid) else '0',
         'restOnActorValid': '0'}
    f.update(fields or {})
    return place_actor(level, 'CProp', template_name, name, pos, f, section_index,
                       library=template_library, template_level=template_level, yaw=yaw,
                       created=created)


def place_squad(level, hero_pos, yaw, floor, template_level='abyss', fields=None,
                layout=None, stand=STAND, tol=1.0, *, library):
    """Ghostbuster0 at hero_pos and the template's four companions behind it, each on its
    own floor(x, z); a companion whose floor is more than tol off the hero's is an error."""
    tl = _templates(library)
    tpl = tl.doc(template_level)
    if tpl is None:
        raise PlaceError('template level %s is not in the archives' % template_level)
    layout = layout or SQUAD_LAYOUT
    hx, hy, hz = hero_pos
    f = {'restOnActorValid': '0'}
    f.update(fields or {})
    place_actor(level, 'CGhostbuster', 'Ghostbuster0', 'Ghostbuster0', hero_pos, f,
                library=tl, template_level=template_level, yaw=yaw)
    heroes = glvl.top_block(level.lvl, 'list-of-heros')
    heroes.children.append(glvl.KV('hero_0', 'Ghostbuster0'))
    cmap = glvl.actor_class_map(tpl)
    ab = glvl.top_block(tpl, 'actors')
    names = [a.tag for a in ab.children if isinstance(a, glvl.Block)
             and cmap.get(a.tag) == 'CGhostbuster' and a.tag != 'Ghostbuster0'][:4]
    placed = ['Ghostbuster0']
    for cname in names:
        if cname not in layout:
            raise PlaceError('%s: no squad layout entry for %s' % (template_level, cname))
        dx, dz = placement.rot_xz(layout[cname][0], layout[cname][1], yaw)
        x, z = hx + dx, hz + dz
        y = floor(x, z)
        if y is None:
            raise PlaceError('%s at (%g, %g) has no collision floor under it' % (cname, x, z))
        if abs(y - (hy - stand)) > tol:
            raise PlaceError('%s stands %.2f from the hero floor' % (cname, y - (hy - stand)))
        place_actor(level, 'CGhostbuster', cname, cname, (x, y + stand, z),
                    {'restOnActorValid': '0'}, library=tl, template_level=template_level, yaw=yaw)
        placed.append(cname)
    return placed


def emit_section_list(doc, names):
    """The <section-list> block levelfile's skeleton leaves empty; [] is the retail shape."""
    return levelfile.emit_section_list(doc, names)

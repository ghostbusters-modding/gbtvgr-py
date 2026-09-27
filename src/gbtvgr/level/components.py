"""Components: reusable mesh + material + texture bundles, and the export-time
resolve step that turns every unique (mesh, bindings) combination into exactly
one shipped .smb, with derived .mtb clones for retextured slots."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import collections
import hashlib
import os
import uuid

from gbtvgr.mesh import smb

from . import materials as mlib


class ComponentDef:
    """A base mesh, one material binding per slot, and the variants offered."""

    def __init__(self, name, mesh, slots=None, cid=None):
        self.id = cid or uuid.uuid4().hex[:12]
        self.name = name
        self.mesh = mesh                       # model ref, e.g. graveyard\crypt01
        self.slots = list(slots or [])         # [material ref or None] per slot
        self.mesh_variants = []                # other model refs sharing the slot contract
        self.slot_variants = {}                # slot index -> [material refs]
        self.texture_variants = {}             # slot index -> {layer: [texture paths]}
        self.solid = True
        self.thumbnail = None

    def to_dict(self):
        return {'id': self.id, 'name': self.name, 'mesh': self.mesh, 'slots': self.slots,
                'mesh_variants': self.mesh_variants,
                'slot_variants': {str(k): v for k, v in self.slot_variants.items()},
                'texture_variants': {str(k): {str(l): t for l, t in v.items()}
                                     for k, v in self.texture_variants.items()},
                'solid': self.solid}

    @classmethod
    def from_dict(cls, d):
        c = cls(d['name'], d['mesh'], d.get('slots'), d.get('id'))
        c.mesh_variants = list(d.get('mesh_variants', []))
        c.slot_variants = {int(k): list(v) for k, v in d.get('slot_variants', {}).items()}
        c.texture_variants = {int(k): {int(l): list(t) for l, t in v.items()}
                              for k, v in d.get('texture_variants', {}).items()}
        c.solid = d.get('solid', True)
        return c

    def resolve(self, overrides=None):
        """The concrete (mesh, per-slot binding) an instance asks for."""
        ov = overrides or {}
        mesh = ov.get('mesh') or self.mesh
        slots = []
        for i, base in enumerate(self.slots):
            ref = ov.get('slots', {}).get(str(i), ov.get('slots', {}).get(i)) or base
            tex = ov.get('textures', {}).get(str(i), ov.get('textures', {}).get(i)) or {}
            tex = tuple(sorted((int(l), t) for l, t in tex.items()))
            slots.append((ref, tex))
        return Variant(mesh, tuple(slots))


class Variant:
    """One resolved combination; content-addressed so identical instances share a file."""
    __slots__ = ('mesh', 'slots', 'key')

    def __init__(self, mesh, slots):
        self.mesh = mesh
        self.slots = slots
        h = hashlib.sha1()
        h.update(mesh.lower().encode('latin1'))
        for ref, tex in slots:
            h.update(b'|' + (ref or '').lower().encode('latin1'))
            for layer, path in tex:
                h.update(b'@%d=%s' % (layer, path.lower().encode('latin1')))
        self.key = h.hexdigest()[:10]

    @property
    def is_base(self):
        return all(not tex for _ref, tex in self.slots)

    def __eq__(self, o):
        return isinstance(o, Variant) and o.key == self.key

    def __hash__(self):
        return hash(self.key)


def short_name(base, key):
    stem = os.path.splitext(base.replace('/', '\\').split('\\')[-1])[0]
    return '%s_%s' % (stem[:20], key[:8])


class ExportPlan:
    """What the resolve step decided: every unique variant, the .smb each one
    ships as, and the .mtb clones the retextures need."""

    def __init__(self):
        self.variants = collections.OrderedDict()     # key -> Variant
        self.model_ref = {}                           # key -> model ref the .lvl uses
        self.material_clones = collections.OrderedDict()  # clone ref -> (base ref, {layer: tex})
        self.instances = []                           # (node, key)

    def add(self, node, variant):
        self.instances.append((node, variant.key))
        if variant.key in self.variants:
            return self.model_ref[variant.key]
        self.variants[variant.key] = variant
        needs_file = False
        for si, (ref, tex) in enumerate(variant.slots):
            if tex and ref:
                clone = mlib.clone_name(ref, hashlib.sha1(
                    (ref.lower() + repr(tex)).encode('latin1')).hexdigest()[:8])
                self.material_clones.setdefault(clone, (ref, dict(tex)))
                needs_file = True
        base_slots = [b for b, _t in variant.slots]
        if not needs_file and not variant_changes_slots(variant, base_slots):
            self.model_ref[variant.key] = variant.mesh
        else:
            self.model_ref[variant.key] = 'editor\\' + short_name(variant.mesh, variant.key)
        return self.model_ref[variant.key]

    def slot_refs(self, variant):
        """The material refs the exported .smb binds per slot."""
        out = []
        for ref, tex in variant.slots:
            if tex and ref:
                clone = mlib.clone_name(ref, hashlib.sha1(
                    (ref.lower() + repr(tex)).encode('latin1')).hexdigest()[:8])
                out.append(clone)
            else:
                out.append(ref)
        return out


def variant_changes_slots(variant, base_slots):
    """True when a slot binding differs from the base model's own material list;
    the resolver fills that in from the model at export time."""
    return getattr(variant, '_rebound', False)


def resolve_all(doc):
    """Every component instance in the document -> ExportPlan."""
    plan = ExportPlan()
    for node in doc.nodes('component'):
        cd = doc.components.get(node.component_id)
        if cd is None:
            continue
        v = cd.resolve(node.overrides())
        plan.add(node, v)
    return plan


# -- writing the files ---------------------------------------------------------------
def _rewrite_refs(raw, refs):
    """A parsed .smb with its material refs replaced; every pad that depends on
    an absolute offset is recomputed."""
    import copy
    m = copy.deepcopy(raw)
    pos = 20 + 16
    for name, _h in m['deps']:
        pos += len(name) + 1 + 16
    pos += 1
    m['deppad'] = b'\0' * ((-pos) % 4)
    pos += len(m['deppad']) + 24
    for e, ref in zip(m['materials'], refs):
        if ref is None or e['embedded'] is not None:
            e2 = e
        else:
            e2 = e
            e2['ref'] = ref.replace('/', '\\').encode('latin1')
        pos += len(e2['ref']) + 1
        e2['refpad'] = b'\0' * ((-pos) % 4)
        pos += len(e2['refpad'])
        if e2['embedded'] is not None:
            e2['embedded']['deppad'] = b''
            probe = mlib.build(e2['embedded'], base_offset=pos)
            pos += len(probe)
    m['hdrpad'] = b''
    probe = smb.build(m)
    datalen = sum(len(p['vdata']) + len(p['idata']) + sum(len(b) for b in p['mdata'])
                  for p in m['parts'])
    m['hdrpad'] = b'\0' * ((-(len(probe) - datalen)) % 16)
    return m


def write_plan(plan, library, out_dir, log=None):
    """Write files/models/editor/*.smb and files/materials/*.mtb for the plan.
    Returns {model ref: path} of what was written."""
    written = {}
    for clone, (base, tex) in plan.material_clones.items():
        rec = library.read_material(base)
        if rec is None:
            raise FileNotFoundError('material %s is not in the archives' % base)
        for layer, path in tex.items():
            rec = mlib.retexture(rec, layer, path)
        dst = os.path.join(out_dir, 'materials', *(clone + '.mtb').split('\\'))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'wb') as fh:
            fh.write(mlib.build(rec))
        written[clone] = dst
        if log:
            log('material clone %s <- %s' % (clone, base))
    for key, v in plan.variants.items():
        ref = plan.model_ref[key]
        if ref == v.mesh:
            continue
        blob = library.read(library.model_ref_path(v.mesh))
        if blob is None:
            raise FileNotFoundError('model %s is not in the archives' % v.mesh)
        raw = smb.parse(blob)
        refs = plan.slot_refs(v)
        refs += [None] * (len(raw['materials']) - len(refs))
        m = _rewrite_refs(raw, refs[:len(raw['materials'])])
        out = smb.build(m)
        smb.parse(out)
        dst = os.path.join(out_dir, 'models', *(ref + '.smb').split('\\'))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'wb') as fh:
            fh.write(out)
        written[ref] = dst
        if log:
            log('model %s <- %s' % (ref, v.mesh))
    return written


def component_from_model(library, ref, name=None):
    """An imported mesh becomes a component definition in one step."""
    model = library.read_model(ref)
    if model is None:
        raise FileNotFoundError(ref)
    slots = [None if m.startswith('embedded:') else m for m in model.materials]
    cd = ComponentDef(name or ref.split('\\')[-1], ref, slots)
    cd.solid = model.collision_part_count > 0
    return cd

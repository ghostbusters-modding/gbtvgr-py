"""The script<->level<->set cross-check: extern binding, handler export,
class/property validity, schema versions, Dependencies resolution."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
#
# The level's own arming-invariant / entry-guard / blocker-sequence checks
# (DESIGN-specific: which trigger is armed by which beat) are NOT here --
# those stay in the level's check_*.py, which calls cross_check() for
# everything else and adds its own source_checks(code) to the same list.
import os
import re

from dante import module as dante_module

from .. import lvl as _lvl
from . import wire

# what the engine opens instead of what the .lvl asks for
REMAP = {'.cit': '.cib', '.smf': '.smb', '.mtl': '.mtb', '.tfa': '.tfb',
         '.dfm': '.bfm', '.tga': '.tex', '.pst': '.bst',
         '.phys2': '.phys2b', '.mtl2': '.mtb', '.fxa': '.fxe'}
# the `.civh` family exist only as text inside a .lvl (no file ships)
NO_FILE = ('.civh',)

TRIG_SLOTS = ('onEvent', 'onRisingEdge', 'onFallingEdge', 'onActorEnter',
              'onActorInside', 'onActorLeave', 'onActorEnterUpStream',
              'onActorEnterDownStream', 'onActorLeaveUpStream',
              'onActorLeaveDownStream')

# THE COLLISION BUDGET.  Level prepare returns 0 -- black front end, no
# script fault -- when the total collision-mesh count registered by SOLID
# CProps gets too high.  Bisected in-game once: 1215 loads, 1393 does not,
# so the cap sits below the lowest LOADING measurement, with real margin.
MAX_COLL_SOLID = 1200


BANNED_CLASSES = ('CBipedLarge', 'CGolemPartBase', 'CFlyerMedium')


def cross_check(P, L, dn, code, lvl, m, podindex_path=None,
                 max_coll_solid=MAX_COLL_SOLID, banned_classes=BANNED_CLASSES,
                 local_deps=()):
    """Everything the source-independent cross-check needs: `P` the plan
    module, `L` the level's make_dock_lvl-shaped module, `dn` the raw .dn
    text, `code` its comments stripped, `lvl` the .lvl text, `m` the parsed
    .bst.  Returns (bad, stats) -- the caller combines `bad` with its own
    source_checks(code) before deciding pass/fail, so both share one list."""
    actor_list = dict(re.findall(
        r'^\t(\w+) = (\w+)\r$',
        re.search(r'<actor-list>\r\n(.*?)</actor-list>', lvl, re.S).group(1),
        re.M))
    groups = set(re.findall(r'^\t(\w+) = <\r$', lvl, re.M))
    sections = {s['name'].split(b'\0')[0].decode() for s in m['sections']}
    defined = set(re.findall(r'^void (\w+)\(', dn, re.M))
    bad = []

    # ---- THE COLLISION BUDGET ---------------------------------------------
    try:
        def _coll(model):
            me = L.smb_meta(model)
            return None if me is None else me['coll']
        tot, nsolid, heaviest = 0, 0, (0, '')
        for mm in re.finditer(r'\t<(\w+)>\r\n(.*?)\t</\1>\r\n', lvl, re.S):
            if actor_list.get(mm.group(1)) != 'CProp':
                continue
            fl = dict(re.findall(r'^\t\t(\w+) = (.*)\r$', mm.group(2), re.M))
            if fl.get('useCollisionParts') != '1':
                continue
            c = _coll(fl.get('modelInstance', '').replace('\\\\', '\\'))
            if c is None:
                continue
            tot += c
            nsolid += 1
            if c > heaviest[0]:
                heaviest = (c, mm.group(1))
        if tot > max_coll_solid:
            bad.append('collision budget: %d collision meshes over %d solid '
                       'props exceeds MAX_COLL_SOLID = %d.  Make a prop '
                       'non-solid (add it to DRESSING) or give it a lighter '
                       'model.' % (tot, nsolid, max_coll_solid))
        else:
            print('collision budget OK: %d collision meshes over %d solid '
                  'props (cap %d; 1215 loads in-game, 1393 does not); '
                  'heaviest solid prop %s at %d'
                  % (tot, nsolid, max_coll_solid, heaviest[1], heaviest[0]))
    except Exception as _e:                       # pragma: no cover
        print('collision budget: NOT CHECKED (%s)' % _e)

    for cls, nm in re.findall(r'^extern (C\w+) (\w+);', code, re.M):
        if cls == 'CDialogDatabaseEntry':
            continue
        if cls == 'CActorGroup':
            if nm not in groups:
                bad.append('extern CActorGroup %s: no such group in the .lvl'
                           % nm)
        elif cls == 'CRoom':
            bad.append('extern CRoom %s: this pass switches no mesh part, so '
                       'the module must declare no CRoom' % nm)
        elif nm not in actor_list:
            bad.append('extern %s %s: no such actor in the .lvl' % (cls, nm))
        elif actor_list[nm] != cls:
            bad.append('%s is %s in the .lvl but %s in the module'
                       % (nm, actor_list[nm], cls))

    for fn in sorted(set(re.findall(r'= (?:void )?(\w+)\(@C\w+', lvl))):
        if fn not in defined:
            bad.append('the .lvl binds %s(), which the module does not export'
                       % fn)

    for nm, cls in actor_list.items():
        if cls != 'CTrigger':
            continue
        for slot in TRIG_SLOTS:
            if slot == 'onEvent':
                continue     # onEvent is not part of the auto-derived family
            want = '%s_%s' % (nm, slot)
            if want not in defined:
                bad.append('trigger %s: the engine can auto-derive %s(), '
                           'which the module does not export' % (nm, want))

    # ---- prop placement, read back out of the .lvl that will SHIP --------
    nprops = 0
    for mm in re.finditer(r'\t<(\w+)>\r\n(.*?)\t</\1>\r\n', lvl, re.S):
        nm, body = mm.group(1), mm.group(2)
        f = dict(re.findall(r'^\t\t(\w+) = (.*)\r$', body, re.M))
        if 'useCollisionParts' not in f:
            continue
        nprops += 1
        model = f['modelInstance'].replace('\\\\', '\\')
        me = L.smb_meta(model)
        if me is None:
            continue
        pos = [float(v) for v in f['pos'].split(',')]
        yaw = float(f.get('orient', '0, 0, 0').split(',')[0])
        # resolve the room from the model's FOOTPRINT centre, not its
        # origin: a wall prop whose origin sits in a back corner puts that
        # origin just past the wall plane, which is correct geometry and the
        # wrong room.
        ex0, ez0, ex1, ez1 = L.world_extent(model, yaw)
        room = P.room_at(pos[0] + (ex0 + ex1) / 2.0,
                         pos[2] + (ez0 + ez1) / 2.0, pos[1])
        if room is None:
            # Two families stand in no room BY DESIGN and are checked by the
            # level's own generator: door leaves/poses (L.DOOR_NAMES) and
            # water props (L.WATER_NAMES).  Anything else out here fell off
            # the map.
            if nm in L.DOOR_NAMES or nm in L.WATER_NAMES:
                continue
            if f['useCollisionParts'] == '1':
                bad.append('%s at (%.1f, %.1f) is SOLID and in no room'
                           % (nm, pos[0], pos[2]))
            continue
        floor = P.floor_y(room, pos[0] + (ex0 + ex1) / 2.0,
                          pos[2] + (ez0 + ez1) / 2.0)
        # the y the generator GROUNDS this model on: the union of the
        # non-debris parts for a GROUND_UNION vehicle (its wheels start
        # above its main part), the main part's lo for everything else --
        # same rule as the placement, so the two agree.
        base = pos[1] + L.ground_lo(model)
        mounted = nm in L.MOUNTED
        if not mounted and base > floor + 0.1:
            bad.append('%s FLOATS %.2f ft above the %s floor'
                       % (nm, base - floor, room))
        if not mounted and base < floor - 0.1:
            bad.append('%s is SUNK %.2f ft into the %s floor'
                       % (nm, floor - base, room))
        if nm in L.WALL_PROPS and not L.facing_ok(nm, model, room, pos[0],
                                                  pos[2], yaw):
            bad.append('%s is a %s-wall prop showing its BACK to the room'
                       % (nm, L.WALL_PROPS[nm][0]))
        if f['useCollisionParts'] == '1' and me['coll'] == 0:
            bad.append('%s has useCollisionParts = 1 but %s carries no '
                       'collision mesh' % (nm, model))

    # ---- (a) EVERY CLASS PLACED MUST CARRY A SCHEMA VERSION ---------------
    declared = set(re.findall(r'^\t(\w+) = \d+\r$', re.search(
        r'<actor-version-list>\r\n(.*?)</actor-version-list>',
        lvl, re.S).group(1), re.M))
    used = set(actor_list.values())
    if used - declared:
        bad.append('classes placed but with no schema version: %s'
                   % ', '.join(sorted(used - declared)))

    # ---- (b) classes the level's roster forbids (PIER 34 bans golems and
    #     their parts; a level that fields them passes banned_classes=()) ---
    for banned in banned_classes:
        placed = [nm for nm, cls in actor_list.items() if cls == banned]
        if placed:
            bad.append('the actor list places %s (%s): the roster has no %s'
                       % (banned, ', '.join(placed[:4]), banned))

    # ---- (c) NO SOLID PROP MAY CARRY A DEGENERATE COLLISION BOX -- checked
    #     against the file that will SHIP, not the table that produced it --
    for mm in re.finditer(r'\t<(\w+)>\r\n(.*?)\t</\1>\r\n', lvl, re.S):
        nm, body = mm.group(1), mm.group(2)
        f = dict(re.findall(r'^\t\t(\w+) = (.*)\r$', body, re.M))
        if f.get('useCollisionParts') != '1':
            continue
        me = L.smb_meta(f.get('modelInstance', '').replace('\\\\', '\\'))
        if me is not None and me['thin'] is not None \
                and me['thin'] < L.MIN_COLL_BOX:
            bad.append('%s is solid and its model has a %.3f ft collision '
                       'box (< %.2f) -- CProp::addCollisionGeom rejects it '
                       'and prepare returns 0'
                       % (nm, me['thin'], L.MIN_COLL_BOX))

    ndeps = 0
    if podindex_path and os.path.exists(podindex_path):
        have = set(l.strip().lower()
                   for l in open(podindex_path) if l.strip())
        # files the mod ships itself (a cloned .cib) are not in any POD
        have |= {d.lower().replace('\\\\', '\\') for d in local_deps}
        deps = [l.strip().replace('\\\\', '\\') for l in re.search(
            r'^Dependencies = <\r\n(.*?)\r\n>', lvl, re.S | re.M
        ).group(1).split('\r\n') if l.strip()]
        ndeps = len(deps)
        for d in deps:
            dl = d.lower()
            base, ext = os.path.splitext(dl)
            if ext in NO_FILE:
                continue
            cands = {dl, 'art\\' + dl}
            if ext in REMAP:
                cands.add(base + REMAP[ext])
                cands.add('art\\' + base + REMAP[ext])
            if not (cands & have):
                bad.append('Dependencies names %s, which is in no POD' % d)

    stats = {'nexterns': len(re.findall(r'^extern C', dn, re.M)),
             'nactors': len(actor_list), 'ngroups': len(groups),
             'nsections': len(sections),
             'nhandlers': len(set(re.findall(r'= (?:void )?(\w+)\(@C\w+', lvl))),
             'nprops': nprops, 'ndeps': ndeps}
    return bad, stats


# ============================================================================
# Level-based checks: the same invariants, read off a wire.Level (.lvl + every
# .sec it recurses into) and a compiled .dante instead of raw text/paths.
# ============================================================================

def _merged_classes(level):
    """name -> class, .lvl then every .sec in order; a .sec redefining a name
    (never seen in the corpus) wins, matching the engine's flat actor table."""
    m = {}
    for cm in wire.actor_classes(level):
        m.update(cm)
    return m


def _merged_groups(level):
    names = set()
    for _name, doc in level.docs():
        ag = _lvl.top_block(doc, 'actor-groups')
        if ag is not None:
            names |= {c.key for c in ag.children
                      if isinstance(c, (_lvl.KV, _lvl.ListF))}
    return names


def _walk_kv(node):
    """Every KV field anywhere under an actor block, nested <Keys>/<Key_N>
    sections included (event bindings can live there too)."""
    for c in node.children:
        if isinstance(c, _lvl.KV):
            yield c
        elif isinstance(c, _lvl.Block):
            for kv in _walk_kv(c):
                yield kv


def check_sections(level):
    """Every .sec level.lvl's <section-list> names must have actually
    loaded (fatal: the engine's own recursive open, RE_LEVEL_LOAD.md sec3)."""
    have = {name for name, _blk in level.secs}
    return ['section-list names %s.sec, which did not load' % s
            for s in wire.sec_names(level.lvl) if s not in have]


def version_classes(level):
    """Every class the level's own actor-version-list blocks pin, .lvl and .sec."""
    out = set()
    for _name, doc in level.docs():
        avl = _lvl.top_block(doc, 'actor-version-list')
        if avl is not None:
            out |= {c.key for c in avl.children if isinstance(c, _lvl.KV)}
    return out


def corpus_classes(source, sets=None):
    """The union of version_classes over every level a source holds: the
    complete list of classes Sony ever placed, which dante_api.json is not."""
    src = wire.open_source(source, sets=sets)
    out = set()
    for stem in src.level_stems():
        out |= version_classes(wire.open_level(src, stem, load_set=False))
    return out


def class_check(level, dante_api_path=None, known=None):
    """Unknown actor class across .lvl+.sec is fatal (RE_LEVEL_LOAD.md step 6h).
    Pass known=corpus_classes(...); the dante_api.json default lacks CProp."""
    if known is None:
        d, _ = _lvl.load_dante_api(dante_api_path or _lvl.DEFAULT_DANTE_API)
        known = (set(d['classes']) | {p['owner'] for p in d['properties']}
                 | set(dante_module.EXTRA_CLASSES))
    known = set(known)
    maps = wire.actor_classes(level)
    bad = []
    for si, actor in wire.actors(level):
        cls = maps[si].get(actor.tag)
        if cls is not None and cls not in known:
            bad.append('%s is class %s, not a known actor class'
                       % (actor.tag, cls))
    return bad


def prop_check(level, dante_api_path=None):
    """Unregistered property keys are ignored by the engine's reflection
    setter (never fatal): warn only, same coverage as `gbtvgr lvl validate`."""
    _, by_field = _lvl.load_dante_api(dante_api_path or _lvl.DEFAULT_DANTE_API)
    warn = []
    for _si, actor in wire.actors(level):
        for fld in _lvl.iter_actor_field_names(actor):
            if _lvl.lookup_field(by_field, fld)[0] is None:
                warn.append('%s: unregistered property %s' % (actor.tag, fld))
    return warn


_HANDLER_RE = re.compile(r'^(?:void )?(\w+)\(@C\w+')
_FN_NAME_RE = re.compile(r'\s(\w+)\(')


def _norm_proto(p):
    """'void f(@CGhost g)' and 'void f( @CGhost )' compare equal: types only."""
    head, _, args = p.strip().partition('(')
    args = args.rstrip(')').strip()
    types = [a.strip().split()[0] for a in args.split(',') if a.strip()] if args else []
    return '%s(%s)' % (' '.join(head.split()), ','.join(types))


def script_externs(mod):
    """(class, name) fixups the module does not export and that lack the '@'
    engine-global sigil: placed actors, not singletons (dante_format.md FIXUPS)."""
    own = {sym for k, _off, sym in mod.exports if k == 'D'}
    out = set()
    for k, sym, _sites in mod.fixups:
        if k != 'D' or sym in own or sym.startswith('@'):
            continue
        cls, _, name = sym.partition(' ')
        if name and re.match(r'^C[A-Z]', cls):
            out.add((cls, name))
    return sorted(out)


# classes an `extern` names that are never a .lvl/.sec actor: CDialogDatabaseEntry
# is lang.py's territory; CRoom is a mesh-side room grouping (0 matches as an
# actor name across the full 20-level corpus); CMaterialSystem/CPostProcessFx/
# CEctoCarEffects are one-per-level engine singletons (always "gMaterialSystem"/
# "postProcessFx"/"Ecto1", 0 matches as an actor name); CPortalLight references
# a .bst light record (level_components.md "Lights": no .lvl/.sec form at all).
# Confirmed against the full shipped corpus, not just the reference levels.
NON_ACTOR_CLASSES = frozenset(('CDialogDatabaseEntry', 'CRoom', 'CMaterialSystem',
                               'CPostProcessFx', 'CEctoCarEffects', 'CPortalLight'))


# the register-indirect member reads the shipped modules make and the game survives:
# CDialogDatabaseEntry::tag (hotel1a) and global.dante's isLastActorSpawnedDead(@CSpawn)
SAFE_INDIRECT_MEMBERS = frozenset(('tag', 'CSpawn::lastActorSpawned'))


def check_indirect_members(mod, allowed=SAFE_INDIRECT_MEMBERS):
    """A member read through a @Class register (a 16-bit fixup site) faults the VM for
    CDialogDatabaseEntry::text; only the corpus-proven members in `allowed` pass."""
    bad = []
    for kind, sym, sites in mod.fixups:
        if kind != 'M':
            continue
        cls_member = sym.split(' ', 1)[-1]
        if cls_member in allowed or cls_member.rsplit('::', 1)[-1] in allowed:
            continue
        at = ['@%X' % a for a, bits in sites if bits == 16]
        if at:
            bad.append('%s is read through a register at %s: the VM faults on that; '
                       'read the member on a global extern' % (sym, ', '.join(at)))
    return bad


def indirect_member_check(level, allowed=SAFE_INDIRECT_MEMBERS):
    """check_indirect_members over a wire.Level's compiled script."""
    if level.script is None:
        return ['no compiled script for this level']
    mod = dante_module.Dante(level.script.decode('latin-1'), level.stem)
    return check_indirect_members(mod, allowed)


def script_handlers(mod):
    """Bare function names this compiled module exports (its `C` exports)."""
    names = set()
    for k, _off, proto in mod.exports:
        if k != 'C':
            continue
        m = _FN_NAME_RE.search(proto)
        if m:
            names.add(m.group(1))
    return names


def cross_check_level(level):
    """Externs resolve to actors across .lvl+.sec and named handlers are
    exported; returns (bad, stats)."""
    if level.script is None:
        return (['no compiled script for this level'], {})
    mod = dante_module.Dante(level.script.decode('latin-1'), level.stem)
    actor_list = _merged_classes(level)
    groups = _merged_groups(level)
    defined = script_handlers(mod)
    externs = script_externs(mod)
    bad = []
    for cls, name in externs:
        if cls in NON_ACTOR_CLASSES:
            continue          # CDialogDatabaseEntry is lang.py's territory; see above
        if cls == 'CActorGroup':
            if name not in groups:
                bad.append('extern CActorGroup %s: no such group in the '
                           '.lvl/.sec' % name)
        elif name not in actor_list:
            bad.append('extern %s %s: no such actor in the .lvl or any .sec'
                       % (cls, name))
        elif actor_list[name] != cls:
            bad.append('%s is %s in the level but %s in the script'
                       % (name, actor_list[name], cls))

    # the engine binds by the whole prototype: a handler declared @CGhost where the
    # level says @CCharacter is "Function not found" at prepare
    protos = set(_norm_proto(proto) for k, _off, proto in mod.exports if k == 'C')
    bound, bound_protos = set(), {}
    for _si, actor in wire.actors(level):
        for kv in _walk_kv(actor):
            m = _HANDLER_RE.match(kv.value)
            if m:
                bound.add(m.group(1))
                bound_protos[m.group(1)] = _norm_proto(kv.value)
    for fn in sorted(bound):
        if fn not in defined:
            bad.append('the level binds %s(), which the script does not '
                       'export' % fn)
        elif bound_protos[fn] not in protos:
            bad.append('the level binds %s but the script exports it with another '
                       'prototype' % bound_protos[fn])

    stats = {'nexterns': len(externs), 'nactors': len(actor_list),
             'ngroups': len(groups), 'nhandlers': len(defined),
             'nbound': len(bound)}
    return bad, stats

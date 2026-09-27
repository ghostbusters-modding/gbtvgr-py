"""Level-agnostic .lvl authoring helpers: .smb metadata, facing maths, wall-run
layout, donor-block text ops and the placement invariant checker."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
#
# Functions that need a level's own tables (PROP_DIR, WALL_PROPS, DRESSING,
# PROPS, ...) take that level's make_dock_lvl-shaped MODULE as `mod` instead
# of importing it, so this module never imports a level's files.  Each one
# has a `_dir` twin below that takes a plain smb corpus directory instead of
# `mod`, for callers with no level authoring table at all (an offline gate).
import math
import os
import re
import struct

from gbtvgr import smb

_META_CACHE = {}
DEFAULT_DEBRIS_RE = re.compile(r'shadow|gib|shard|chunk|chip|glass', re.I)


def _smb_meta(smb_dir, model, ground_union=frozenset(), debris_re=DEFAULT_DEBRIS_RE, pose=0):
    """The measurement smb_meta()/smb_meta_dir() share, keyed by (smb_dir, model, pose).
    Parts are measured in the pose the game draws: 0 on a CProp and on a CAniModel
    until setFrame; an animated door's raw frame can stand 90 degrees off it."""
    key = (smb_dir, model, pose)
    if key in _META_CACHE:
        return _META_CACHE[key]
    fn = os.path.join(smb_dir, model.lower().replace('\\', os.sep) + '.smb')
    if not os.path.exists(fn):
        _META_CACHE[key] = None
        return None
    m = smb.parse(open(fn, 'rb').read())
    parts = []
    for i, part in enumerate(m['parts']):
        nm = part['name'].split(b'\0')[0].decode('latin1')
        pts = smb.posed_corners(m, i, struct.unpack('<6f', part['bbox']), pose)
        parts.append((nm, smb.bbox_of(pts), pts))

    def union(sel):
        lo, hi = [1e9] * 3, [-1e9] * 3
        for row in sel:
            b = row[1]
            for k in range(3):
                lo[k] = min(lo[k], b[k])
                hi[k] = max(hi[k], b[k + 3])
        return tuple(lo), tuple(hi)

    all_lo, all_hi = union(parts)
    main = [pb for pb in parts if pb[0].lower().startswith('intact')]
    if main:
        pname = main[0][0] + ('' if len(main) == 1 else '(+%d)' % (len(main) - 1))
        mlo, mhi = union(main)
    else:
        # no intact naming: the biggest part and every peer at least a quarter its
        # volume (a double door's two leaves), never the chips around them
        vol = lambda b: (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2])
        big = max(parts, key=lambda pb: vol(pb[1]))
        main = [pb for pb in parts if vol(pb[1]) >= 0.25 * vol(big[1])]
        pname = big[0] + ('' if len(main) == 1 else '(+%d)' % (len(main) - 1))
        mlo, mhi = union(main)
    corners = [c for pb in main for c in pb[2]]
    # THE THINNEST COLLISION BOX (CProp::addCollisionGeom rejects one too
    # thin, and the whole prepare returns 0)
    thin = None
    for c in m['collisions']:
        v = c['verts']
        n = len(v) // 12
        if n == 0:
            thin = 0.0
            continue
        pts = [struct.unpack_from('<3f', v, i * 12) for i in range(n)]
        d = min(max(p[k] for p in pts) - min(p[k] for p in pts)
                for k in range(3))
        thin = d if thin is None else min(thin, d)
    glo = mlo[1]
    if model in ground_union:
        keep = [pb for pb in parts if not debris_re.search(pb[0])]
        glo = union(keep)[0][1]
    _META_CACHE[key] = {'part': pname, 'lo': tuple(mlo), 'hi': tuple(mhi),
                        'all_lo': all_lo, 'all_hi': all_hi, 'corners': corners,
                        'coll': len(m['collisions']), 'thin': thin, 'glo': glo}
    return _META_CACHE[key]


def smb_meta(mod, model):
    """Everything the placer needs about a model, read out of the shipped
    .smb itself: the MAIN box, the union of every part's, and how many
    collision meshes the file carries.

    THE MAIN BOX is the union of every part named `intact*` (most models are
    breakables: intact render parts plus broken/chunk*/gib*/chip* debris in
    the same local space, only shown after the thing shatters -- sizing off
    every part mis-sizes and lifts the prop).  A model with no `intact` part
    takes its largest-volume part.

    `coll` is the count of `coll_*` meshes; `thin` is the thinnest collision
    box; `glo` is the local y the prop is GROUNDED on -- the main box's
    floor, or for a `mod.GROUND_UNION` vehicle the lowest non-debris vertex
    (its tyres)."""
    return _smb_meta(mod.PROP_DIR, model, getattr(mod, 'GROUND_UNION', frozenset()),
                     getattr(mod, '_DEBRIS', DEFAULT_DEBRIS_RE))


def smb_meta_dir(smb_dir, model, ground_union=frozenset(), debris_re=DEFAULT_DEBRIS_RE, pose=0):
    """`smb_meta`, off a plain smb corpus directory instead of a level `mod`."""
    return _smb_meta(smb_dir, model, ground_union, debris_re, pose)


def smb_bbox(mod, model):
    """The MAIN box -- what the prop actually looks like."""
    me = smb_meta(mod, model)
    return None if me is None else (me['lo'], me['hi'])


def smb_bbox_dir(smb_dir, model):
    me = smb_meta_dir(smb_dir, model)
    return None if me is None else (me['lo'], me['hi'])


def ground_lo(mod, model):
    """The model-local y this generator grounds `model` on: the lowest
    non-debris vertex for a `mod.GROUND_UNION` vehicle (its wheels), the
    main part's lowest vertex for everything else."""
    me = smb_meta(mod, model)
    return None if me is None else me['glo']


def ground_lo_dir(smb_dir, model, ground_union=frozenset(), debris_re=DEFAULT_DEBRIS_RE):
    me = smb_meta_dir(smb_dir, model, ground_union, debris_re)
    return None if me is None else me['glo']


def is_solid(mod, name, model, dy):
    """Solidity is DERIVED, never hand-kept: no collision mesh -> never
    solid; mounted (explicit dy) unless in `mod.SOLID_MOUNTED` -> never
    solid; shorter than `mod.SOLID_MIN_H` -> never solid; named in
    `mod.DRESSING` -> never solid; otherwise SOLID."""
    me = smb_meta(mod, model)
    if me is None or me['coll'] == 0:
        return False
    if dy is not None and name not in getattr(mod, 'SOLID_MOUNTED', frozenset()):
        return False
    if name in getattr(mod, 'DRESSING', frozenset()):
        return False
    return (me['hi'][1] - me['lo'][1]) >= getattr(mod, 'SOLID_MIN_H', 1.5)


# ---------------------------------------------------------------------------
# FACING AND PLACEMENT.  YAW CONVENTION -- confirmed against the shipped
# corpus (the lost_island church_pew fan), do not re-derive:
#     world = (dx*cos(yaw) + dz*sin(yaw), -dx*sin(yaw) + dz*cos(yaw))
# i.e. +90 deg maps local +Z to world +X and local +X to world -Z.
# ---------------------------------------------------------------------------
INWARD = {'W': (1.0, 0.0), 'E': (-1.0, 0.0),
          'S': (0.0, 1.0), 'N': (0.0, -1.0)}


def orient3(o):
    """(yaw, pitch, roll) from a bare yaw or a triple."""
    if isinstance(o, (int, float)):
        return (float(o), 0.0, 0.0)
    return (float(o[0]), float(o[1]), float(o[2]))


def rot_xz(dx, dz, yaw):
    a = math.radians(yaw)
    ca, sa = math.cos(a), math.sin(a)
    return dx * ca + dz * sa, -dx * sa + dz * ca


def rot3(p, orient):
    """A local point through (yaw, pitch, roll): roll about local z, then
    pitch about local x, then the corpus yaw about y."""
    yaw, pitch, roll = orient3(orient)
    x, y, z = p
    r = math.radians(roll)
    x, y = x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r)
    q = math.radians(pitch)
    y, z = y * math.cos(q) - z * math.sin(q), y * math.sin(q) + z * math.cos(q)
    x, z = rot_xz(x, z, yaw)
    return x, y, z


def front_local(mod, model):
    """Which way the model faces IN ITS OWN SPACE, read off the mesh: a
    panel faces along its thin axis; otherwise the axis whose bbox centre
    sits furthest off-origin; a symmetric mesh faces -Z (the asset set's
    front)."""
    return front_local_dir(mod.PROP_DIR, model)


def front_local_dir(smb_dir, model):
    me = smb_meta_dir(smb_dir, model)
    if me is None:
        return (0.0, -1.0)
    lo, hi = me['lo'], me['hi']
    ex, ez = hi[0] - lo[0], hi[2] - lo[2]
    cx, cz = (lo[0] + hi[0]) / 2.0, (lo[2] + hi[2]) / 2.0
    if ez > 1e-6 and ex / max(ez, 1e-6) < 0.25:
        return (-1.0 if cx <= 0 else 1.0, 0.0)          # panel, thin in X
    if ex > 1e-6 and ez / max(ex, 1e-6) < 0.25:
        return (0.0, -1.0 if cz <= 0 else 1.0)          # panel, thin in Z
    rx = abs(cx) / ex if ex > 1e-6 else 0.0
    rz = abs(cz) / ez if ez > 1e-6 else 0.0
    if max(rx, rz) < 0.15:
        return (0.0, -1.0)
    if rz >= rx * 0.9:
        return (0.0, -1.0 if cz <= 0 else 1.0)
    return (-1.0 if cx <= 0 else 1.0, 0.0)


def long_axis(mod, model):
    return long_axis_dir(mod.PROP_DIR, model)


def long_axis_dir(smb_dir, model):
    me = smb_meta_dir(smb_dir, model)
    if me is None:
        return 'x'
    lo, hi = me['lo'], me['hi']
    return 'x' if (hi[0] - lo[0]) >= (hi[2] - lo[2]) else 'z'


def yaw_to_face(mod, model, want):
    return yaw_to_face_dir(mod.PROP_DIR, model, want)


def yaw_to_face_dir(smb_dir, model, want):
    fx, fz = front_local_dir(smb_dir, model)
    for yaw in (0, 90, 180, 270):
        rx, rz = rot_xz(fx, fz, yaw)
        if abs(rx - want[0]) < 1e-6 and abs(rz - want[1]) < 1e-6:
            return yaw
    return 0


def yaw_to_align(mod, model, side):
    """The quarter turn that lays the model's LONG axis along the wall and,
    of the two that do, the one whose front looks into the room."""
    return yaw_to_align_dir(mod.PROP_DIR, model, side)


def yaw_to_align_dir(smb_dir, model, side):
    runs_z = side in ('W', 'E')
    want = INWARD[side]
    best = None
    for yaw in (0, 90, 180, 270):
        lx, lz = rot_xz(*((1.0, 0.0) if long_axis_dir(smb_dir, model) == 'x'
                          else (0.0, 1.0)), yaw)
        if (abs(lz) > abs(lx)) != runs_z:
            continue
        fx, fz = rot_xz(*front_local_dir(smb_dir, model), yaw)
        score = fx * want[0] + fz * want[1]
        if best is None or score > best[0]:
            best = (score, yaw)
    return 0 if best is None else best[1]


def _need_meta(mod, model):
    me = smb_meta(mod, model)
    if me is None:
        raise SystemExit(
            'no .smb for %r under %s -- add its family to the extraction '
            'list in build.sh.  (Placing a model the build cannot measure '
            'means its bounding box, its facing and its collision boxes all '
            'go unchecked.)' % (model, mod.PROP_DIR))
    return me


def _need_meta_dir(smb_dir, model):
    me = smb_meta_dir(smb_dir, model)
    if me is None:
        raise SystemExit('no .smb for %r under %s' % (model, smb_dir))
    return me


def world_extent(mod, model, yaw):
    """The model's rotated horizontal bounds, main part only (yaw only)."""
    return world_extent3(mod, model, yaw)


def world_extent_dir(smb_dir, model, yaw):
    return world_extent3_dir(smb_dir, model, yaw)


def world_extent3(mod, model, orient):
    """Rotated horizontal bounds through a full (yaw, pitch, roll)."""
    return world_extent3_dir(mod.PROP_DIR, model, orient)


def world_extent3_dir(smb_dir, model, orient):
    me = _need_meta_dir(smb_dir, model)
    xs, zs = [], []
    for c in me['corners']:
        rx, _ry, rz = rot3(c, orient)
        xs.append(rx)
        zs.append(rz)
    return min(xs), min(zs), max(xs), max(zs)


def world_aabb(mod, model, x, y, z, orient):
    return world_aabb_dir(mod.PROP_DIR, model, x, y, z, orient)


def world_aabb_dir(smb_dir, model, x, y, z, orient):
    me = smb_meta_dir(smb_dir, model)
    if me is None:
        return None
    pts = [rot3(c, orient) for c in me['corners']]
    return (x + min(p[0] for p in pts), y + min(p[1] for p in pts),
            z + min(p[2] for p in pts), x + max(p[0] for p in pts),
            y + max(p[1] for p in pts), z + max(p[2] for p in pts))


def in_section(smb_dir, model, x, y, z, orient, bbox, tol=0.25):
    """True if model's world AABB at (x,y,z,orient) fits inside `bbox` (a raw
    24-byte lo3+hi3 .bst section bbox, tol ft slack); None if unmeasurable."""
    aabb = world_aabb_dir(smb_dir, model, x, y, z, orient)
    if aabb is None:
        return None
    lo = struct.unpack_from('<3f', bbox, 0)
    hi = struct.unpack_from('<3f', bbox, 12)
    return (aabb[0] >= lo[0] - tol and aabb[1] >= lo[1] - tol
            and aabb[2] >= lo[2] - tol and aabb[3] <= hi[0] + tol
            and aabb[4] <= hi[1] + tol and aabb[5] <= hi[2] + tol)


def run_layout(mod, runs, model_of, wall_props):
    """Resolve a RUNS table -- (room, side, lo, hi, [prop names]) laid along
    that stretch of wall with EQUAL gaps, measured on each model's rotated
    footprint -- into an along-wall centre per prop."""
    at_of = {}
    for room, side, lo, hi, members in runs:
        widths = []
        for nm in members:
            model = model_of[nm]
            mode = wall_props.get(nm, (side, 'face'))[1]
            yaw = (yaw_to_align(mod, model, side) if mode == 'along'
                   else yaw_to_face(mod, model, INWARD[side]))
            x0, z0, x1, z1 = world_extent(mod, model, yaw)
            widths.append((z1 - z0) if side in ('W', 'E') else (x1 - x0))
        slack = (hi - lo) - sum(widths)
        if slack < 0:
            raise SystemExit('run %s %s: %.1f ft of props in %.1f ft of wall'
                             % (room, side, sum(widths), hi - lo))
        gap = slack / (len(members) + 1)
        at = lo + gap
        for nm, w in zip(members, widths):
            at_of[nm] = at + w / 2.0
            at += w + gap
    return at_of


# ---------------------------------------------------------------------------
# donor-block text ops: every prop/actor is cloned out of a shipped .lvl's
# own text and edited in place -- never built from scratch, so an authored
# actor is always a schema the engine has already accepted once.
# ---------------------------------------------------------------------------
def block(raw, name):
    """One actor block out of a donor, NESTED sections included: the close
    tag is matched at one-tab indent by name, so a <Keys>/<Key_0> section
    (two and three tabs) inside a CCameraPathActor is captured whole.  The
    tag balance is asserted so a donor edit cannot hand back half a block."""
    m = re.search(r'^\t<%s>\r\n.*?^\t</%s>\r\n' % (re.escape(name),
                                                   re.escape(name)),
                  raw, re.S | re.M)
    if not m:
        raise SystemExit('no <%s> block in the donor' % name)
    blk = m.group(0)
    opens = sorted(re.findall(r'^\t+<(\w+)>\r$', blk, re.M))
    closes = sorted(re.findall(r'^\t+</(\w+)>\r$', blk, re.M))
    if opens != closes:
        raise SystemExit('<%s>: nested sections are unbalanced (%s vs %s)'
                         % (name, opens, closes))
    return blk


def set_field(blk, field, value):
    return re.sub(r'^(\t\t%s = )[^\r\n]*' % re.escape(field),
                  lambda mm: mm.group(1) + value, blk, flags=re.M)


def set_key_field(blk, field, value):
    """A field of a nested <Keys>/<Key_N> section (four tabs deep)."""
    out, n = re.subn(r'^(\t\t\t\t%s = )[^\r\n]*' % re.escape(field),
                     lambda mm: mm.group(1) + value, blk, flags=re.M)
    if n != 1:
        raise SystemExit('key field %s: %d matches, wanted 1' % (field, n))
    return out


def blank_events(blk):
    return re.sub(r'^(\t\t\w+ = )[A-Za-z@][^=\r\n]*\([^\r\n]*\)(?=\r?$)',
                  r'\g<1>""', blk, flags=re.M)


def rename(blk, old, new):
    return blk.replace('<%s>' % old, '<%s>' % new) \
              .replace('</%s>' % old, '</%s>' % new) \
              .replace('= %s\r' % old, '= %s\r' % new)


def vec(x, y, z):
    return '%.2f, %.2f, %.2f' % (x, y, z)


def prop_block(name, model, pos, orient, solid, created=True):
    """The 11-field CProp schema every shipped CProp actor carries, in the
    same order -- nothing is omitted.  `created` False emits createStatus 0:
    a solid door prop that is DOWN at load."""
    yaw, pitch, roll = orient3(orient)
    f = [('name', name), ('createStatus', '1' if created else '0'),
         ('pos', vec(*pos)), ('orient', '%g, %g, %g' % (yaw, pitch, roll)),
         ('processRadius', '250'), ('damageFilterEvent', '""'),
         # restOnActorValid MUST be 0: 1 means "settle onto whatever is below
         # at creation", which can teleport a prop onto donor terrain far
         # below the level.
         ('restOnActorValid', '0'),
         ('onSnaredStatusChangedEvent', '""'),
         ('modelInstance', model.replace('\\', '\\\\')),
         ('useCollisionParts', '1' if solid else '0'), ('propFlags', '0')]
    return ('\t<%s>\r\n' % name + ''.join('\t\t%s = %s\r\n' % kv for kv in f)
            + '\t</%s>\r\n' % name)


# ===========================================================================
# THE PLACEMENT INVARIANT.  `mod` carries every table this checks (PROPS,
# DOORS, POSES, WATER_PROPS, SPAWNERS, TRIGGERS, LITEBULBS, EMITTERS,
# CAMERAS, WALL_PROPS) plus the level's own place()/is_solid()/facing_ok()/
# check_doors()/check_poses()/check_water() -- none of those tables are
# lifted (DESIGN content), only the walk over them.
# ===========================================================================
def check_placement(mod):
    """The invariants a navmesh and walls depend on:
      1. a solid prop stands inside a nav exclusion for its room (a scenery
         room, nav=False, has no nav to protect and is exempt)
      2. a solid prop's model carries collision meshes, none thinner than
         mod.MIN_COLL_BOX
      3. no prop pokes through a wall, floor or ceiling of its room
      4. door props stand in their band and cover their opening
      5. water props stay in the harbour/world bounds."""
    P = mod.P
    bad, nsolid = [], 0
    for name, model, room, ax, az, dy, ao, _legacy in mod.PROPS:
        r = P.ROOM[room]
        x, y, z, orient = mod.place(name, model, room, ax, az, dy, ao)
        yaw = orient[0]
        solid = mod.is_solid(name, model, dy)
        nsolid += 1 if solid else 0
        me = _need_meta(mod, model)
        if solid and me['coll'] == 0:
            bad.append('%s is SOLID but %s carries no collision mesh' % (name, model))
        if solid and me['thin'] is not None and me['thin'] < mod.MIN_COLL_BOX:
            bad.append('%s is SOLID but %s has a collision box only %.3f ft '
                       'thick (< %.2f): CProp::addCollisionGeom rejects it'
                       % (name, model, me['thin'], mod.MIN_COLL_BOX))
        if solid and r['nav']:
            fx0, fz0, fx1, fz1 = world_extent3(mod, model, orient)
            fx0, fx1, fz0, fz1 = x + fx0, x + fx1, z + fz0, z + fz1
            ok = False
            for rk, ex0, ez0, ex1, ez1 in P.NAV_EXCLUDE:
                if rk != room:
                    continue
                a0 = ex0 - 1.5 if abs(ex0 - r['x0']) < 0.01 else ex0
                a1 = ex1 + 1.5 if abs(ex1 - r['x1']) < 0.01 else ex1
                b0 = ez0 - 1.5 if abs(ez0 - r['z0']) < 0.01 else ez0
                b1 = ez1 + 1.5 if abs(ez1 - r['z1']) < 0.01 else ez1
                if (a0 - 0.02 <= fx0 and fx1 <= a1 + 0.02
                        and b0 - 0.02 <= fz0 and fz1 <= b1 + 0.02):
                    ok = True
                    break
            if not ok:
                bad.append('%s is SOLID but its footprint (%.1f..%.1f, '
                           '%.1f..%.1f) is not inside one %s nav exclusion'
                           % (name, fx0, fx1, fz0, fz1, room))
        if not mod.facing_ok(name, model, room, x, z, yaw):
            bad.append('%s is a %s-wall prop whose front points AT the wall'
                       % (name, mod.WALL_PROPS[name][0]))
        bb = world_aabb(mod, model, x, y, z, orient)
        fy = P.floor_y(room, x, z)
        cy = P.ceil_y(room, x, z)
        # the base is what the prop stands on: its wheels for a vehicle
        base = min(bb[1], y + me['glo'])
        why = []
        if bb[0] < r['x0'] - 0.25:
            why.append('through the west wall by %.1f' % (r['x0'] - bb[0]))
        if bb[3] > r['x1'] + 0.25:
            why.append('through the east wall by %.1f' % (bb[3] - r['x1']))
        if bb[2] < r['z0'] - 0.25:
            why.append('through the south wall by %.1f' % (r['z0'] - bb[2]))
        if bb[5] > r['z1'] + 0.25:
            why.append('through the north wall by %.1f' % (bb[5] - r['z1']))
        if base < fy - 0.05:
            why.append('under the floor by %.1f' % (fy - base))
        if dy is None and base > fy + 0.1:
            why.append('FLOATING %.2f ft above the floor' % (base - fy))
        if bb[4] > cy + 0.3:
            why.append('through the ceiling by %.1f' % (bb[4] - cy))
        if why:
            bad.append('%-20s %-40s %s @(%.1f, %.1f): %s'
                       % (name, model, room, x, z, '; '.join(why)))
    for nm, room, pos, dy in mod.SPAWNERS:
        r = P.ROOM[room]
        if not (r['x0'] <= pos[0] <= r['x1'] and r['z0'] <= pos[1] <= r['z1']):
            bad.append('spawner %s is outside %s' % (nm, room))
    for nm, room, x, z, size in mod.TRIGGERS:
        if room is None:
            continue
        r = P.ROOM[room]
        if not (r['x0'] <= x <= r['x1'] and r['z0'] <= z <= r['z1']):
            bad.append('trigger %s is outside %s' % (nm, room))
    for rows in (mod.LITEBULBS, mod.EMITTERS):
        for row in rows:
            nm, room, x, z = row[0], row[1], row[2], row[3]
            if room is None:
                continue
            r = P.ROOM[room]
            if not (r['x0'] <= x <= r['x1'] and r['z0'] <= z <= r['z1']):
                bad.append('%s is outside %s' % (nm, room))
    for nm, (cx, cy, cz), (lx, lz) in mod.CAMERAS:
        room = P.room_at(cx, cz)
        if room is None:
            bad.append('camera %s is in no room' % nm)
        elif not (P.floor_y(room, cx, cz) + 1.0 <= cy <= P.ceil_y(room, cx, cz) - 1.0):
            bad.append('camera %s is outside %s vertically' % (nm, room))
        if P.room_at(lx, lz) is None and not any(
                a['axis'] == 'z' and a['band'][0] <= lz <= a['band'][1]
                and a['s0'] <= lx <= a['s1'] or
                a['axis'] == 'x' and a['band'][0] <= lx <= a['band'][1]
                and a['s0'] <= lz <= a['s1'] for a in P.APERTURES):
            bad.append('camera %s looks at (%.0f, %.0f), which is inside a wall'
                       % (nm, lx, lz))
    bad += mod.check_doors()
    bad += mod.check_poses()
    bad += mod.check_water()
    names = ([p[0] for p in mod.PROPS] + [d[0] for d in mod.DOORS]
             + [p[0] for p in mod.POSES] + list(mod.WATER_NAMES))
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        bad.append('duplicate prop names: %s' % ', '.join(sorted(dup)))
    if bad:
        raise SystemExit('placement errors (%d):\n  ' % len(bad)
                         + '\n  '.join(bad))
    print('placement OK: %d room props (%d SOLID, every one inside a nav '
          'exclusion with collision >= %.2f ft), %d door leaves in their '
          'bands, %d door poses (%d solid open leaves inside their strips), '
          '%d water props, %d spawners, %d triggers, %d cameras'
          % (len(mod.PROPS), nsolid, mod.MIN_COLL_BOX, len(mod.DOORS),
             len(mod.POSES), sum(1 for p in mod.POSES if p[8]),
             len(mod.WATER_PROPS), len(mod.SPAWNERS), len(mod.TRIGGERS),
             len(mod.CAMERAS)))
    return nsolid

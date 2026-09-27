"""Prop clusters for set dressing: crate islands, barrel rows, cordons and the
like as lists of prop rows, seeded so a level builds the same twice."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
#
# A row is dict(model, pos, yaw, solid, aabb); footprints come from KNOWN or
# a measure() table, so nothing here needs the corpus.
import math
import random

from .placement import rot_xz, smb_meta_dir

LANE = 3.0                # clear walking lane at a cluster's edge
GAP = 0.5                 # daylight between two footprints
SOLID_MIN_H = 1.5         # a shorter prop is stepped over, not collided

# model -> (lo, hi, collision mesh count), main box in model space, ft
KNOWN = {
    'museum\\crate01': ((-1.99, 0.0, -1.99), (1.99, 3.98, 1.99), 14),
    'industrial\\woodcrate01': ((-1.99, 0.03, -1.99), (1.99, 4.01, 1.99), 18),
    'industrial\\woodcrate02old': ((-1.5, 0.02, -3.0), (1.5, 2.02, 3.0), 21),
    'industrial\\woodcrate03': ((-1.0, 0.0, -0.83), (1.0, 1.18, 0.83), 8),
    'industrial\\woodcrate04': ((-1.53, 0.0, -3.34), (1.53, 6.19, 3.34), 41),
    'museum\\metalcrate04': ((-1.53, 0.0, -1.77), (1.52, 3.73, 1.77), 10),
    'museum\\metalcrate06': ((-1.54, -0.15, -3.46), (1.54, 3.59, 3.42), 14),
    'lost_island\\crate02': ((-3.38, 0.0, -3.38), (3.38, 9.18, 3.38), 1),
    'industrial\\pallet_wood': ((-1.63, 0.05, -1.88), (1.62, 0.48, 1.88), 7),
    'lost_island\\stbarrel1': ((-1.59, 0.0, -1.57), (1.59, 4.51, 1.57), 1),
    'lost_island\\stbarrel2': ((-1.84, 0.0, -1.81), (1.84, 4.51, 1.81), 7),
    'industrial\\explosivebarrel': ((-1.11, -0.14, -1.11), (1.11, 3.77, 1.11), 5),
    'lost_island\\scranegear': ((-3.4, -3.45, -0.53), (3.4, 3.45, 0.53), 1),
    'lost_island\\junk1': ((-0.8, 0.0, -0.81), (0.83, 1.3, 0.81), 1),
    'lost_island\\junk4': ((-0.02, -0.01, -0.9), (1.16, 0.74, 1.25), 4),
    'lost_island\\junk8': ((-0.69, 0.0, -0.46), (0.85, 0.75, 0.37), 1),
    'lost_island\\junk10': ((-0.51, -0.04, -0.6), (0.67, 0.85, 0.72), 1),
    'lost_island\\junk11': ((-0.97, -0.04, -0.78), (0.42, 0.85, 0.55), 5),
    'lost_island\\lifesaver': ((-1.91, 0.01, -1.91), (1.91, 0.54, 1.91), 1),
    'street\\cautionsawhorse01': ((-0.81, 0.0, -1.93), (0.88, 5.08, 1.68), 8),
    'street\\cone01': ((-0.69, 0.0, -0.69), (0.69, 2.38, 0.69), 2),
    'street\\cautionbarrel01': ((-1.2, 0.0, -1.16), (1.2, 4.03, 1.18), 6),
    'museum\\swinglight': ((-1.0, -5.65, -1.0), (1.0, 0.0, 1.0), 1),
    'ecto1\\ecto1b_tire': ((-0.42, -1.24, -1.24), (0.42, 1.24, 1.24), 1),
}
CRATES = ('museum\\crate01', 'industrial\\woodcrate04', 'museum\\metalcrate04',
          'industrial\\woodcrate02old', 'museum\\metalcrate06')
SMALL_CRATES = ('museum\\metalcrate04', 'industrial\\woodcrate02old',
                'industrial\\woodcrate03')
STACK_OVERHANG = 0.3      # a top crate may hang this far past its base
PALLET = 'industrial\\pallet_wood'
BARRELS = ('lost_island\\stbarrel1', 'lost_island\\stbarrel2',
           'industrial\\explosivebarrel')
SPOOL = 'lost_island\\scranegear'      # no cable drum ships; a gear on edge reads as one
JUNK = ('lost_island\\junk1', 'lost_island\\junk4', 'lost_island\\junk8',
        'lost_island\\junk10', 'lost_island\\junk11', 'lost_island\\lifesaver')
SAWHORSE = 'street\\cautionsawhorse01'
CONE = 'street\\cone01'
LAMP = 'museum\\swinglight'            # a hanging head: the origin is the hook
BOLLARD = 'street\\cautionbarrel01'    # no bollard ships; kit.bollard() is the geometry one
TYRE = 'ecto1\\ecto1b_tire'


# ===========================================================================
# footprints
# ===========================================================================
def measure(models, smb_dir):
    """A `sizes` table off a .smb corpus, for models KNOWN does not carry."""
    out = {}
    for m in models:
        me = smb_meta_dir(smb_dir, m)
        if me is None:
            raise KeyError('no .smb for %r under %s' % (m, smb_dir))
        out[m] = (me['lo'], me['hi'], me['coll'])
    return out


def meta(model, sizes=None):
    if sizes and model in sizes:
        return sizes[model]
    if model in KNOWN:
        return KNOWN[model]
    raise KeyError('%r is not in dress.KNOWN: pass sizes=measure([...], smb_dir)'
                   % model)


def footprint(model, yaw, sizes=None):
    """(x0, z0, x1, z1) of the model's main box at `yaw`, about its origin."""
    lo, hi, _c = meta(model, sizes)
    pts = [rot_xz(x, z, yaw) for x in (lo[0], hi[0]) for z in (lo[2], hi[2])]
    return (min(p[0] for p in pts), min(p[1] for p in pts),
            max(p[0] for p in pts), max(p[1] for p in pts))


def row(model, x, z, yaw, y=0.0, sizes=None, solid=None, hang=False):
    """One prop row grounded on y (its main box's floor on the floor), or with
    its origin AT y when `hang` (a lamp head hung from a hook)."""
    lo, hi, coll = meta(model, sizes)
    py = y if hang else y - lo[1]
    fx0, fz0, fx1, fz1 = footprint(model, yaw, sizes)
    if solid is None:
        solid = coll > 0 and (hi[1] - lo[1]) >= SOLID_MIN_H and not hang
    return {'model': model, 'pos': (x, py, z), 'yaw': yaw % 360.0,
            'solid': bool(solid),
            'aabb': (x + fx0, py + lo[1], z + fz0, x + fx1, py + hi[1], z + fz1)}


def _clear(aabb, rows, gap=GAP):
    """True if `aabb` keeps `gap` from every row it shares height with."""
    x0, y0, z0, x1, y1, z1 = aabb
    for r in rows:
        ox0, oy0, oz0, ox1, oy1, oz1 = r['aabb']
        if oy1 <= y0 + 1e-6 or oy0 >= y1 - 1e-6:
            continue
        if x0 < ox1 + gap and x1 > ox0 - gap and z0 < oz1 + gap and z1 > oz0 - gap:
            return False
    return True


def _inside(aabb, rect):
    return (aabb[0] >= rect[0] - 1e-6 and aabb[2] >= rect[1] - 1e-6
            and aabb[3] <= rect[2] + 1e-6 and aabb[5] <= rect[3] + 1e-6)


def overlaps(rows, gap=0.0):
    """Pairs of rows whose footprints overlap at a shared height: the check."""
    bad = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if not _clear(rows[i]['aabb'], [rows[j]], gap):
                bad.append((i, j))
    return bad


def nav_boxes(rows):
    """The solid rows as kit.nav_exclusions boxes."""
    return [r['aabb'] for r in rows if r['solid']]


def _inner(anchor, size, lane=LANE):
    ax, az = anchor
    w, d = size
    return (ax - w / 2.0 + lane, az - d / 2.0 + lane,
            ax + w / 2.0 - lane, az + d / 2.0 - lane)


def _pack(rng, inner, picks, rows, y, sizes, jitter=0.4):
    """Seeded shelf packing: rows filled left to right with the first candidate
    from picks(rng) that fits, each row starting past the deepest of the last."""
    x0, z0, x1, z1 = inner
    placed = []
    z = z0
    while z < z1:
        x, deepest, n0 = x0, 0.0, len(placed)
        while x < x1:
            hit = None
            for model, yaw in picks(rng):
                fx0, fz0, fx1, fz1 = footprint(model, yaw, sizes)
                w, d = fx1 - fx0, fz1 - fz0
                if x + w > x1 + 1e-6 or z + d > z1 + 1e-6:
                    continue
                jx = min(jitter, x1 - (x + w)) * rng.random()
                jz = min(jitter, z1 - (z + d)) * rng.random()
                r = row(model, x + jx - fx0, z + jz - fz0, yaw, y, sizes)
                if _clear(r['aabb'], rows + placed):
                    hit = r
                    break
            if hit is None:
                break
            placed.append(hit)
            x = hit['aabb'][3] + GAP + rng.random() * 0.25
            deepest = max(deepest, hit['aabb'][5] - z)
        if len(placed) == n0:
            break
        z += deepest + GAP + rng.random() * 0.25
    return placed


# ===========================================================================
# clusters
# ===========================================================================
def crate_island(anchor, footprint_size, models=CRATES, seed=0, sizes=None,
                 y=0.0, lane=LANE, stack=True):
    """Crates packed into `footprint_size` (w, d) about `anchor` (x, z), a
    clear lane at the edge, some small ones stacked on the big ones."""
    rng = random.Random(seed)
    inner = _inner(anchor, footprint_size, lane)

    def picks(rng):
        ms = list(models)
        rng.shuffle(ms)
        out = []
        for m in ms:
            yaw = rng.choice((0, 180)) + rng.uniform(-6, 6)
            out += [(m, yaw), (m, yaw + 90.0)]
        return out
    rows = _pack(rng, inner, picks, [], y, sizes)
    if stack:
        small = [m for m in SMALL_CRATES if sizes is None or m in sizes or m in KNOWN]
        for base in list(rows):
            bx0, _by0, bz0, bx1, by1, bz1 = base['aabb']
            if by1 - y < 3.0 or rng.random() > 0.4:
                continue
            m = rng.choice(small)
            yaw = base['yaw'] + rng.uniform(-10, 10)
            fx0, fz0, fx1, fz1 = footprint(m, yaw, sizes)
            if fx1 - fx0 > (bx1 - bx0) + STACK_OVERHANG or \
                    fz1 - fz0 > (bz1 - bz0) + STACK_OVERHANG:
                continue
            cx = (bx0 + bx1) / 2.0 - (fx0 + fx1) / 2.0
            cz = (bz0 + bz1) / 2.0 - (fz0 + fz1) / 2.0
            top = row(m, cx, cz, yaw, by1, sizes)
            if _inside(top['aabb'], inner) and _clear(top['aabb'], rows):
                rows.append(top)
    return rows


def pallet_stack(anchor, n=6, model=PALLET, seed=0, sizes=None, y=0.0, yaw=0.0):
    """`n` pallets stacked at `anchor`, each a few degrees off the last."""
    rng = random.Random(seed)
    rows = []
    lo, hi, _c = meta(model, sizes)
    h = hi[1] - lo[1]
    for k in range(n):
        rows.append(row(model, anchor[0], anchor[1], yaw + rng.uniform(-4, 4),
                        y + k * h, sizes, solid=True))
    return rows


def barrel_row(anchor, axis, count, models=BARRELS, seed=0, sizes=None, y=0.0,
               gap=GAP):
    """`count` barrels in a line from `anchor` along +axis, touching but for `gap`."""
    rng = random.Random(seed)
    rows = []
    a = anchor[0] if axis == 'x' else anchor[1]
    c = anchor[1] if axis == 'x' else anchor[0]
    for _k in range(count):
        m = rng.choice(models)
        yaw = rng.uniform(0, 360)
        fx0, fz0, fx1, fz1 = footprint(m, yaw, sizes)
        lo_a, hi_a = (fx0, fx1) if axis == 'x' else (fz0, fz1)
        pos_a = a - lo_a
        cc = c + rng.uniform(-0.3, 0.3)
        x, z = (pos_a, cc) if axis == 'x' else (cc, pos_a)
        rows.append(row(m, x, z, yaw, y, sizes))
        a = pos_a + hi_a + gap
    return rows


def cable_spools(anchor, footprint_size, count=4, model=SPOOL, seed=0,
                 sizes=None, y=0.0, lane=LANE):
    """Drums standing on edge, packed into the footprint, two ways round."""
    rng = random.Random(seed)
    inner = _inner(anchor, footprint_size, lane)

    def picks(rng):
        first = rng.choice((0, 90))
        return [(model, first + rng.uniform(-12, 12)),
                (model, 90 - first + rng.uniform(-12, 12))]
    rows = _pack(rng, inner, picks, [], y, sizes)
    return rows[:count]


def net_pile(anchor, count=8, models=JUNK, seed=0, sizes=None, y=0.0,
             radius=4.0):
    """Small junk scattered within `radius` of `anchor`, nothing overlapping.
    No net model ships; pass `models` for whatever reads as a pile."""
    rng = random.Random(seed)
    rows = []
    for _t in range(count * 12):
        if len(rows) >= count:
            break
        m = rng.choice(models)
        rr = radius * math.sqrt(rng.random())
        ang = rng.uniform(0, 2 * math.pi)
        x, z = anchor[0] + rr * math.cos(ang), anchor[1] + rr * math.sin(ang)
        r = row(m, x, z, rng.uniform(0, 360), y, sizes)
        ax0, _a, az0, ax1, _b, az1 = r['aabb']
        if max(abs(ax0 - anchor[0]), abs(ax1 - anchor[0]), abs(az0 - anchor[1]),
               abs(az1 - anchor[1])) > radius:
            continue
        if _clear(r['aabb'], rows, 0.2):
            rows.append(r)
    return rows


def _along(line, pitch, ends=False):
    """Points along the line ((x0, z0), (x1, z1)) at `pitch`, the row centred
    (or one at each end with no gap over pitch)."""
    (x0, z0), (x1, z1) = line
    ln = math.hypot(x1 - x0, z1 - z0)
    if ln < 1e-6:
        return [(x0, z0)]
    if ends:
        n = max(2, int(math.ceil(ln / pitch - 1e-6)) + 1)
        ts = [k / (n - 1) for k in range(n)]
    else:
        n = max(1, int(math.floor(ln / pitch + 1e-6)) + 1)
        start = (ln - (n - 1) * pitch) / 2.0
        ts = [(start + k * pitch) / ln for k in range(n)]
    return [(x0 + (x1 - x0) * t, z0 + (z1 - z0) * t) for t in ts]


def _line_yaw(line, model, sizes=None):
    """The yaw laying the model's long axis along the line."""
    (x0, z0), (x1, z1) = line
    lo, hi, _c = meta(model, sizes)
    long_x = (hi[0] - lo[0]) >= (hi[2] - lo[2])
    ang = math.degrees(math.atan2(x1 - x0, z1 - z0))
    return (ang + (90.0 if long_x else 0.0)) % 360.0


def sawhorse_cordon(line, pitch=8.0, models=(SAWHORSE, CONE), sizes=None, y=0.0):
    """Sawhorses along the line, side-on to it, a cone between each pair."""
    horse, cone = models
    yaw = _line_yaw(line, horse, sizes)
    pts = _along(line, pitch)
    rows = [row(horse, x, z, yaw, y, sizes) for x, z in pts]
    for (ax, az), (bx, bz) in zip(pts, pts[1:]):
        rows.append(row(cone, (ax + bx) / 2.0, (az + bz) / 2.0, 0.0, y, sizes))
    return rows


def lamp_post_row(line, pitch=24.0, model=LAMP, height=12.0, sizes=None, y=0.0,
                  yaw=0.0):
    """Lamp heads every `pitch` along the line, hung `height` over y.  No
    post model ships: the default is the museum's hanging swinglight."""
    return [row(model, x, z, yaw, y + height, sizes, hang=True)
            for x, z in _along(line, pitch)]


def bollard_row(line, pitch=6.0, model=BOLLARD, sizes=None, y=0.0, yaw=0.0):
    """One prop every `pitch` along the line, both ends taken."""
    return [row(model, x, z, yaw, y, sizes) for x, z in _along(line, pitch, ends=True)]


def tyre_fenders(line, pitch=6.0, model=TYRE, drop=1.5, sizes=None, y=0.0,
                 side=+1):
    """Tyres hung off a quay edge running along the line, faces out, centres
    `drop` below the deck; `side` picks which side of the line they hang on."""
    (x0, z0), (x1, z1) = line
    ln = math.hypot(x1 - x0, z1 - z0) or 1.0
    px, pz = -(z1 - z0) / ln * side, (x1 - x0) / ln * side
    lo, hi, _c = meta(model, sizes)
    thin = min(hi[0] - lo[0], hi[2] - lo[2]) / 2.0 + 0.1
    thin_x = (hi[0] - lo[0]) <= (hi[2] - lo[2])
    yaw = math.degrees(math.atan2(-pz, px)) if thin_x else \
        math.degrees(math.atan2(-pz, px)) + 90.0
    rows = []
    for x, z in _along(line, pitch):
        r = row(model, x + px * thin, z + pz * thin, yaw, y - drop, sizes,
                solid=False, hang=True)
        rows.append(r)
    return rows

"""Architectural detail kit: trusses, columns, ducts, stairs, railings, frames and
the rest, emitted as quads through any object with Geo's `quad` method."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
#
# Solids face out, pockets of room air (window_bay, skylight_monitor) face in;
# every element returns the solid boxes it made, for nav exclusions.
import math

DEFAULT_TESS = 2.0
FACES = ('-x', '+x', '-y', '+y', '-z', '+z')
# geo.build_nav drops a link at 2.0 ft of climb per 6 ft cell, so a flight
# only stays walkable for the nav while run > NAV_RUN_PER_RISE * rise.
NAV_RUN_PER_RISE = 3.0


# ===========================================================================
# the quad layer
# ===========================================================================
def step_of(g, tess):
    """The tessellation step in ft: an explicit `tess` (0 = one quad per face),
    else the emitter's own (Geo carries it as mod.TESS), else DEFAULT_TESS."""
    if tess is not None:
        return float(tess) if tess > 0 else None
    t = getattr(g, 'tess', None)
    if t is None:
        t = getattr(getattr(g, 'mod', None), 'TESS', None)
    return float(t) if t else DEFAULT_TESS


def _slices(lo, hi, step):
    if step is None or hi - lo <= step + 1e-6:
        return [(lo, hi)]
    n = int(math.ceil((hi - lo) / step - 1e-6))
    return [(lo + (hi - lo) * k / n, lo + (hi - lo) * (k + 1) / n)
            for k in range(n)]


def _grid(g, sec, mat, room, pt, u0, u1, v0, v1, n, step, collide, mesh,
          vstep=0):
    """pt(u, v) over the rect, sliced every `step` along u and `vstep` along v
    (0 = same as u, None = one slice); one collision quad where g can take it."""
    if u1 - u0 < 1e-6 or v1 - v0 < 1e-6:
        return
    cq = getattr(g, 'collide_quad', None)
    per_quad = collide and cq is None
    vs = step if vstep == 0 else vstep
    for ua, ub in _slices(u0, u1, step):
        for va, vb in _slices(v0, v1, vs):
            g.quad(sec, mat, room, [pt(ua, va), pt(ub, va), pt(ub, vb),
                                    pt(ua, vb)], n, collide=per_quad,
                   mesh=mesh)
    if collide and cq is not None:
        cq([pt(u0, v0), pt(u1, v0), pt(u1, v1), pt(u0, v1)])


def rect(g, sec, mat, room, axis, at, a0, b0, a1, b1, sign, collide=True,
         mesh=None, tess=None):
    """One axis-aligned rectangle on the plane axis=at, facing `sign`.
    (a, b) is (z, y) on an x plane, (x, z) on a y plane, (x, y) on a z plane."""
    s = float(sign)
    if axis == 'x':
        pt, n = (lambda u, v: (at, v, u)), (s, 0.0, 0.0)
    elif axis == 'y':
        pt, n = (lambda u, v: (u, at, v)), (0.0, s, 0.0)
    else:
        pt, n = (lambda u, v: (u, v, at)), (0.0, 0.0, s)
    _grid(g, sec, mat, room, pt, a0, a1, b0, b1, n, step_of(g, tess),
          collide, mesh)


def box(g, sec, room, x0, y0, z0, x1, y1, z1, mat, collide=True, top=True,
        bottom=True, skip=(), mesh=None, tess=None):
    """Six outward faces.  `skip` names faces to leave out ('-x', '+y', ...):
    the one against a wall, the one buried in another member."""
    skip = set(skip)
    if not top:
        skip.add('+y')
    if not bottom:
        skip.add('-y')
    if '-x' not in skip:
        rect(g, sec, mat, room, 'x', x0, z0, y0, z1, y1, -1, collide, mesh, tess)
    if '+x' not in skip:
        rect(g, sec, mat, room, 'x', x1, z0, y0, z1, y1, +1, collide, mesh, tess)
    if '-z' not in skip:
        rect(g, sec, mat, room, 'z', z0, x0, y0, x1, y1, -1, collide, mesh, tess)
    if '+z' not in skip:
        rect(g, sec, mat, room, 'z', z1, x0, y0, x1, y1, +1, collide, mesh, tess)
    if '+y' not in skip:
        rect(g, sec, mat, room, 'y', y1, x0, z0, x1, z1, +1, collide, mesh, tess)
    if '-y' not in skip:
        rect(g, sec, mat, room, 'y', y0, x0, z0, x1, z1, -1, collide, mesh, tess)
    return [(x0, y0, z0, x1, y1, z1)]


def _ring_y(g, sec, mat, room, y, outer, inner, sign, collide, mesh, tess):
    """A horizontal square annulus: the exposed top of a band around a shaft."""
    ox0, oz0, ox1, oz1 = outer
    ix0, iz0, ix1, iz1 = inner
    rect(g, sec, mat, room, 'y', y, ox0, oz0, ox1, iz0, sign, collide, mesh, tess)
    rect(g, sec, mat, room, 'y', y, ox0, iz1, ox1, oz1, sign, collide, mesh, tess)
    rect(g, sec, mat, room, 'y', y, ox0, iz0, ix0, iz1, sign, collide, mesh, tess)
    rect(g, sec, mat, room, 'y', y, ix1, iz0, ox1, iz1, sign, collide, mesh, tess)


def _p3(axis, a, y, c):
    """(along, y, across) to xyz: `axis` is the along axis."""
    return (a, y, c) if axis == 'x' else (c, y, a)


def _other(axis):
    return 'z' if axis == 'x' else 'x'


def _prism(g, sec, mat, room, axis, c0, c1, poly, caps=(True, True),
           collide=False, mesh=None, tess=None, skip_edges=()):
    """A convex quad `poly` of (along, y) points extruded across c0..c1: the
    slanted members (a truss diagonal, a stringer) that a box cannot be."""
    area = sum(poly[i][0] * poly[(i + 1) % 4][1] - poly[(i + 1) % 4][0]
               * poly[i][1] for i in range(4))
    sgn = 1.0 if area >= 0 else -1.0
    step = step_of(g, tess)
    for i in range(4):
        if i in skip_edges:
            continue
        (pa, py), (qa, qy) = poly[i], poly[(i + 1) % 4]
        ln = math.hypot(qa - pa, qy - py)
        if ln < 1e-6:
            continue
        n = _p3(axis, sgn * (qy - py) / ln, -sgn * (qa - pa) / ln, 0.0)

        def pt(u, v, pa=pa, py=py, qa=qa, qy=qy, ln=ln):
            return _p3(axis, pa + (qa - pa) * u / ln, py + (qy - py) * u / ln, v)
        _grid(g, sec, mat, room, pt, 0.0, ln, c0, c1, n, step, collide, mesh)
    for c, s, on in ((c0, -1.0, caps[0]), (c1, +1.0, caps[1])):
        if on:
            g.quad(sec, mat, room, [_p3(axis, a, y, c) for a, y in poly],
                   _p3(axis, 0.0, 0.0, s), collide=collide, mesh=mesh)


def _bar(g, sec, mat, room, axis, at, p0, p1, size, collide, mesh, tess):
    """A square-section bar from p0 to p1 in the (along, y) plane, ends buried."""
    da, dy = p1[0] - p0[0], p1[1] - p0[1]
    ln = math.hypot(da, dy) or 1.0
    pa, py = -dy / ln * size / 2.0, da / ln * size / 2.0
    poly = [(p0[0] - pa, p0[1] - py), (p1[0] - pa, p1[1] - py),
            (p1[0] + pa, p1[1] + py), (p0[0] + pa, p0[1] + py)]
    _prism(g, sec, mat, room, axis, at - size / 2.0, at + size / 2.0, poly,
           caps=(False, False), collide=collide, mesh=mesh, tess=tess)


def _spread(lo, hi, pitch, ends=True):
    """Positions lo..hi: with `ends`, one at each end and no gap over `pitch`;
    without, a centred row at exactly `pitch`."""
    if ends:
        n = max(2, int(math.ceil((hi - lo) / pitch - 1e-6)) + 1)
        return [lo + (hi - lo) * k / (n - 1) for k in range(n)]
    n = max(1, int(math.floor((hi - lo) / pitch + 1e-6)) + 1)
    start = lo + ((hi - lo) - (n - 1) * pitch) / 2.0
    return [start + k * pitch for k in range(n)]


def _spans(lo, hi, cuts):
    """[lo, hi] minus the intervals in `cuts`."""
    spans = [(lo, hi)]
    for c0, c1 in cuts:
        out = []
        for s0, s1 in spans:
            if c1 <= s0 or c0 >= s1:
                out.append((s0, s1))
                continue
            if c0 > s0:
                out.append((s0, c0))
            if c1 < s1:
                out.append((c1, s1))
        spans = out
    return spans


def _wall_box(g, sec, room, axis, at, sign, s0, s1, y0, y1, depth, mat,
              collide, top, bottom, mesh, tess, skip=()):
    """A box standing proud of the wall plane axis=at on the `sign` side; the
    face on the wall is not drawn, the wall already is."""
    lo, hi = sorted((at, at + sign * depth))
    skip = set(skip) | {('-' if sign > 0 else '+') + axis}
    if axis == 'x':
        return box(g, sec, room, lo, y0, s0, hi, y1, s1, mat, collide, top,
                   bottom, skip, mesh, tess)
    return box(g, sec, room, s0, y0, lo, s1, y1, hi, mat, collide, top,
               bottom, skip, mesh, tess)


# ===========================================================================
# structure
# ===========================================================================
def column(g, sec, room, x0, z0, x1, z1, y0, y1, mat, mat_band=None,
           base_h=3.0, cap_h=1.5, proud=0.25, collide=True, top=True,
           bottom=False, skip=(), mesh=None, tess=None):
    """A box column with a base band and a cap band in `mat_band`, proud of the
    shaft.  The shaft stops at each band, so nothing is coplanar."""
    skip = set(skip)
    if mat_band is None or (base_h <= 0 and cap_h <= 0):
        return box(g, sec, room, x0, y0, z0, x1, y1, z1, mat, collide, top,
                   bottom, skip, mesh, tess)
    ya = y0 + base_h if base_h > 0 else y0
    yb = y1 - cap_h if cap_h > 0 else y1
    assert ya < yb, 'column bands meet: no shaft left between them'
    bx0 = x0 - (0.0 if '-x' in skip else proud)
    bx1 = x1 + (0.0 if '+x' in skip else proud)
    bz0 = z0 - (0.0 if '-z' in skip else proud)
    bz1 = z1 + (0.0 if '+z' in skip else proud)
    outer, inner = (bx0, bz0, bx1, bz1), (x0, z0, x1, z1)
    boxes = box(g, sec, room, x0, ya, z0, x1, yb, z1, mat, collide,
                top and cap_h <= 0, bottom and base_h <= 0, skip, mesh, tess)
    if base_h > 0:
        boxes += box(g, sec, room, bx0, y0, bz0, bx1, ya, bz1, mat_band,
                     collide, False, bottom, skip, mesh, tess)
        _ring_y(g, sec, mat_band, room, ya, outer, inner, +1, collide, mesh, tess)
    if cap_h > 0:
        boxes += box(g, sec, room, bx0, yb, bz0, bx1, y1, bz1, mat_band,
                     collide, top, False, skip, mesh, tess)
        _ring_y(g, sec, mat_band, room, yb, outer, inner, -1, collide, mesh, tess)
    return boxes


def beam(g, sec, room, axis, at, s0, s1, y0, y1, width, mat, collide=False,
         top=True, bottom=True, skip=(), mesh=None, tess=None):
    """A horizontal member along `axis` from s0 to s1, centred at `at` across."""
    h = width / 2.0
    if axis == 'x':
        return box(g, sec, room, s0, y0, at - h, s1, y1, at + h, mat, collide,
                   top, bottom, skip, mesh, tess)
    return box(g, sec, room, at - h, y0, s0, at + h, y1, s1, mat, collide,
               top, bottom, skip, mesh, tess)


def truss(g, sec, room, axis, at, span_lo, span_hi, y_bottom, depth, mat,
          chord=0.5, web=0.25, bays=None, collide=False, caps=(True, True),
          mesh=None, tess=None):
    """An open-web truss: two chords, verticals on the bay lines and Warren
    diagonals.  Overhead, no collision; caps=False where a chord bears on a wall."""
    span = span_hi - span_lo
    if bays is None:
        bays = max(1, int(round(span / max(depth, 1.0))))
    y_top = y_bottom + depth
    skip = {'-' + axis} if not caps[0] else set()
    if not caps[1]:
        skip.add('+' + axis)
    beam(g, sec, room, axis, at, span_lo, span_hi, y_top - chord, y_top, chord,
         mat, collide, skip=skip, mesh=mesh, tess=tess)
    beam(g, sec, room, axis, at, span_lo, span_hi, y_bottom, y_bottom + chord,
         chord, mat, collide, skip=skip, mesh=mesh, tess=tess)
    lo, hi = y_bottom + chord, y_top - chord
    for k in range(bays + 1):
        a = span_lo + span * k / bays
        a0, a1 = max(span_lo, a - web / 2.0), min(span_hi, a + web / 2.0)
        vskip = {f for f in skip if (k == 0) == (f[0] == '-')}
        beam(g, sec, room, axis, at, a0, a1, lo, hi, web, mat, collide,
             top=False, bottom=False, skip=vskip, mesh=mesh, tess=tess)
    for k in range(bays):
        a0 = span_lo + span * k / bays
        a1 = a0 + span / bays
        p0, p1 = ((a0, lo), (a1, hi)) if k % 2 == 0 else ((a0, hi), (a1, lo))
        _bar(g, sec, mat, room, axis, at, p0, p1, web, collide, mesh, tess)
    h = chord / 2.0
    if axis == 'x':
        return [(span_lo, y_bottom, at - h, span_hi, y_top, at + h)]
    return [(at - h, y_bottom, span_lo, at + h, y_top, span_hi)]


def purlins(g, sec, room, axis, span_lo, span_hi, at_lo, at_hi, pitch, y,
            size, mat, collide=False, top=True, over=(), skip=(), mesh=None,
            tess=None):
    """Small beams along `axis` every `pitch` across at_lo..at_hi, sitting on y.
    `over` lists the (lo, hi) chord spans they cross, where no underside is drawn."""
    boxes = []
    h = size / 2.0
    for a in _spread(at_lo, at_hi, pitch, ends=False):
        boxes += beam(g, sec, room, axis, a, span_lo, span_hi, y, y + size,
                      size, mat, collide, top=top, bottom=False, skip=skip,
                      mesh=mesh, tess=tess)
        for s0, s1 in _spans(span_lo, span_hi, over):
            if axis == 'x':
                rect(g, sec, mat, room, 'y', y, s0, a - h, s1, a + h, -1,
                     collide, mesh, tess)
            else:
                rect(g, sec, mat, room, 'y', y, a - h, s0, a + h, s1, -1,
                     collide, mesh, tess)
    return boxes


def duct(g, sec, room, path, y0, width, height, mat, collide=False,
         end_caps=(True, True), mesh=None, tess=None):
    """A rectangular run along an axis-aligned polyline of (x, z) points, with
    a cube joint at every elbow that the runs open into."""
    assert len(path) >= 2, 'a duct needs two points'
    y1 = y0 + height
    h = width / 2.0
    boxes = []
    last = len(path) - 2
    for i, (p, q) in enumerate(zip(path, path[1:])):
        k = 0 if abs(q[0] - p[0]) > abs(q[1] - p[1]) else 1
        along = 'xz'[k]
        assert abs(p[1 - k] - q[1 - k]) < 1e-6, 'duct runs must be axis-aligned'
        fwd = q[k] > p[k]
        s0, s1 = sorted((p[k], q[k]))
        # which end of the sorted run is the polyline's start
        start_lo = fwd
        joint_lo = (i > 0) if start_lo else (i < last)
        joint_hi = (i < last) if start_lo else (i > 0)
        cap_lo = end_caps[0] if start_lo else end_caps[1]
        cap_hi = end_caps[1] if start_lo else end_caps[0]
        skip = set()
        if joint_lo:
            s0 += h
            skip.add('-' + along)
        elif not cap_lo:
            skip.add('-' + along)
        if joint_hi:
            s1 -= h
            skip.add('+' + along)
        elif not cap_hi:
            skip.add('+' + along)
        boxes += beam(g, sec, room, along, p[1 - k], s0, s1, y0, y1, width, mat,
                      collide, skip=skip, mesh=mesh, tess=tess)
    for i in range(1, len(path) - 1):
        cx, cz = path[i]
        skip = set()
        for ox, oz in (path[i - 1], path[i + 1]):
            if abs(ox - cx) > abs(oz - cz):
                skip.add('+x' if ox > cx else '-x')
            else:
                skip.add('+z' if oz > cz else '-z')
        boxes += box(g, sec, room, cx - h, y0, cz - h, cx + h, y1, cz + h, mat,
                     collide, skip=skip, mesh=mesh, tess=tess)
    return boxes


def _octagon_cap(g, sec, mat, room, axis, s, at, y, radius, sign, collide, mesh):
    """An octagon end as three quads: the middle band and two trapezoids."""
    R = radius * math.cos(math.radians(22.5))
    h = radius * math.sin(math.radians(22.5))
    n = _p3(axis, sign, 0.0, 0.0)
    for quad in (((y - h, at - R), (y - h, at + R), (y + h, at + R), (y + h, at - R)),
                 ((y + h, at - R), (y + h, at + R), (y + R, at + h), (y + R, at - h)),
                 ((y - R, at - h), (y - R, at + h), (y - h, at + R), (y - h, at - R))):
        g.quad(sec, mat, room, [_p3(axis, s, yy, cc) for yy, cc in quad], n,
               collide=collide, mesh=mesh)


def pipe(g, sec, room, axis, at, y, s0, s1, radius, mat, brackets=None,
         bracket_to=None, mat_bracket=None, bracket_size=0.3, collide=False,
         caps=(True, True), mesh=None, tess=None):
    """An octagonal run, flats top and bottom; `brackets` is a pitch and
    `bracket_to` the (axis, coord) the hangers reach: a ceiling or a wall."""
    step = step_of(g, tess)
    ring = [(at + radius * math.cos(math.radians(22.5 + 45.0 * j)),
             y + radius * math.sin(math.radians(22.5 + 45.0 * j)))
            for j in range(9)]
    for j in range(8):
        (ca, ya), (cb, yb) = ring[j], ring[j + 1]
        na = math.radians(45.0 * (j + 1))
        n = _p3(axis, 0.0, math.sin(na), math.cos(na))

        def pt(u, v, ca=ca, ya=ya, cb=cb, yb=yb):
            return _p3(axis, u, ya + (yb - ya) * v, ca + (cb - ca) * v)
        _grid(g, sec, mat, room, pt, s0, s1, 0.0, 1.0, n, step, collide, mesh,
              vstep=None)
    R = radius * math.cos(math.radians(22.5))
    for s, sign, on in ((s0, -1.0, caps[0]), (s1, 1.0, caps[1])):
        if on:
            _octagon_cap(g, sec, mat, room, axis, s, at, y, radius, sign,
                         collide, mesh)
    boxes = []
    if brackets and bracket_to:
        bm = mat_bracket or mat
        bax, bc = bracket_to
        hs = bracket_size / 2.0
        for a in _spread(s0 + 1.0, s1 - 1.0, brackets, ends=True):
            if bax == 'y':
                lo, hi = (y + R, bc) if bc > y else (bc, y - R)
                x0, _y, z0 = _p3(axis, a - hs, 0.0, at - hs)
                x1, _y, z1 = _p3(axis, a + hs, 0.0, at + hs)
                boxes += box(g, sec, room, x0, lo, z0, x1, hi, z1, bm, collide,
                             top=False, bottom=False, mesh=mesh, tess=0)
            else:
                assert bax == _other(axis), 'a wall bracket reaches across the pipe'
                lo, hi = (at + R, bc) if bc > at else (bc, at - R)
                boxes += beam(g, sec, room, bax, a, lo, hi, y - hs, y + hs,
                              bracket_size, bm, collide,
                              skip={'-' + bax, '+' + bax}, mesh=mesh, tess=0)
    if axis == 'x':
        boxes.append((s0, y - radius, at - radius, s1, y + radius, at + radius))
    else:
        boxes.append((at - radius, y - radius, s0, at + radius, y + radius, s1))
    return boxes


# ===========================================================================
# circulation: railings, stairs, decks, ladders
# ===========================================================================
def railing(g, sec, room, x0, z0, x1, z1, y, mat_post, mat_rail, height=3.5,
            post_pitch=6.0, rails=2, post=0.25, rail=0.2, collide=True,
            wall_ends=(False, False), mesh=None, tess=None):
    """Posts and rails along an axis-aligned line on the deck at y; the rails
    collide as one thin invisible box.  wall_ends: which end post meets a wall."""
    along = 'x' if abs(x1 - x0) >= abs(z1 - z0) else 'z'
    if along == 'x':
        assert abs(z1 - z0) < 1e-6, 'railing runs must be axis-aligned'
        a0, a1, at = min(x0, x1), max(x0, x1), z0
    else:
        assert abs(x1 - x0) < 1e-6, 'railing runs must be axis-aligned'
        a0, a1, at = min(z0, z1), max(z0, z1), x0
    hp = post / 2.0
    posts = _spread(a0 + hp, a1 - hp, post_pitch, ends=True)
    for i, a in enumerate(posts):
        pskip = set()
        if wall_ends[0] and i == 0:
            pskip.add('-' + along)
        if wall_ends[1] and i == len(posts) - 1:
            pskip.add('+' + along)
        beam(g, sec, room, along, at, a - hp, a + hp, y, y + height, post,
             mat_post, collide, bottom=False, skip=pskip, mesh=mesh, tess=0)
    cq = getattr(g, 'collide_quad', None)
    # rails run between posts, ends buried, so nothing shares a post's face
    for k in range(rails):
        top = y + height * (k + 1) / rails
        for pa, pb in zip(posts, posts[1:]):
            beam(g, sec, room, along, at, pa + hp, pb - hp, top - rail, top,
                 rail, mat_rail, collide and cq is None,
                 skip={'-' + along, '+' + along}, mesh=mesh, tess=tess)
    if collide and cq is not None:
        hr = rail / 2.0
        c0, c1, yt = at - hr, at + hr, y + height
        cq([_p3(along, a0, y, c0), _p3(along, a1, y, c0),
            _p3(along, a1, yt, c0), _p3(along, a0, yt, c0)])
        cq([_p3(along, a0, y, c1), _p3(along, a1, y, c1),
            _p3(along, a1, yt, c1), _p3(along, a0, yt, c1)])
        cq([_p3(along, a0, yt, c0), _p3(along, a1, yt, c0),
            _p3(along, a1, yt, c1), _p3(along, a0, yt, c1)])
    lo, hi = _p3(along, a0, y, at - hp), _p3(along, a1, y + height, at + hp)
    return [(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])]


def stair(g, sec, room, axis, sign, x0, z0, width, run, rise, treads,
          mat_tread, mat_riser, mat_stringer=None, y0=0.0, stringer=0.4,
          stringer_depth=1.5, collide_risers=False, soffit=True,
          end_caps=(True, False), check_nav=True, mesh=None, tess=None):
    """A flight from the corner (x0, z0) climbing `sign` along `axis`; only the
    treads collide.  Returns (boxes, slope), the slope being the nav's ramp."""
    if check_nav:
        assert run > NAV_RUN_PER_RISE * rise, \
            'stair run %.2f too short for rise %.2f: the nav needs run > %g x rise' \
            % (run, rise, NAV_RUN_PER_RISE)
    a_start = x0 if axis == 'x' else z0
    c0 = z0 if axis == 'x' else x0
    c1 = c0 + width
    y_top = y0 + treads * rise
    a_end = a_start + sign * treads * run
    for i in range(treads):
        a = a_start + sign * i * run
        b = a + sign * run
        ylo, yhi = y0 + i * rise, y0 + (i + 1) * rise
        rect(g, sec, mat_riser, room, axis, a, c0, ylo, c1, yhi, -sign,
             collide_risers, mesh, tess)
        lo, hi = min(a, b), max(a, b)
        if axis == 'x':
            rect(g, sec, mat_tread, room, 'y', yhi, lo, c0, hi, c1, +1, True,
                 mesh, tess)
        else:
            rect(g, sec, mat_tread, room, 'y', yhi, c0, lo, c1, hi, +1, True,
                 mesh, tess)
    if soffit:
        hyp = math.hypot(run, rise)
        n = _p3(axis, sign * rise / hyp, -run / hyp, 0.0)
        ln = treads * hyp

        def pt(u, v):
            return _p3(axis, a_start + sign * u * run / hyp,
                       y0 + u * rise / hyp, v)
        _grid(g, sec, mat_riser, room, pt, 0.0, ln, c0, c1, n,
              step_of(g, tess), True, mesh)
    if mat_stringer is not None and stringer > 0:
        depth = min(stringer_depth, (y_top - y0) * 0.9)
        poly = [(a_start, y0), (a_end, y_top - depth), (a_end, y_top),
                (a_start, y0 + rise)]
        # edge 3 stands at the foot, edge 1 against the landing
        skip_edges = [i for i, on in ((3, end_caps[0]), (1, end_caps[1])) if not on]
        for s0, s1 in ((c0 - stringer, c0), (c1, c1 + stringer)):
            _prism(g, sec, mat_stringer, room, axis, s0, s1, poly, collide=True,
                   mesh=mesh, tess=tess, skip_edges=skip_edges)
    lo = _p3(axis, min(a_start, a_end), y0, c0 - stringer)
    hi = _p3(axis, max(a_start, a_end), y_top, c1 + stringer)
    slope = [_p3(axis, a_start, y0, c0), _p3(axis, a_start, y0, c1),
             _p3(axis, a_end, y_top, c1), _p3(axis, a_end, y_top, c0)]
    return [(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])], slope


def platform(g, sec, room, x0, z0, x1, z1, y, thickness, mat_deck, mat_edge,
             supports=True, mat_support=None, post=1.0, post_pitch=12.0,
             floor_y=0.0, collide=True, skip=(), mesh=None, tess=None):
    """A raised deck with fascia (sides not in `skip`) and a grid of posts down
    to floor_y.  Returns the posts: the deck is a nav zone, not an exclusion."""
    yb = y - thickness
    rect(g, sec, mat_deck, room, 'y', y, x0, z0, x1, z1, +1, collide, mesh, tess)
    rect(g, sec, mat_edge, room, 'y', yb, x0, z0, x1, z1, -1, collide, mesh, tess)
    box(g, sec, room, x0, yb, z0, x1, y, z1, mat_edge, collide, top=False,
        bottom=False, skip=skip, mesh=mesh, tess=tess)
    boxes = []
    if supports and yb > floor_y + 1e-6:
        sm = mat_support or mat_edge
        hp = post / 2.0
        edge = {'-x': x0, '+x': x1, '-z': z0, '+z': z1}
        for px in _spread(x0 + hp, x1 - hp, post_pitch, ends=True):
            for pz in _spread(z0 + hp, z1 - hp, post_pitch, ends=True):
                face = {'-x': px - hp, '+x': px + hp, '-z': pz - hp, '+z': pz + hp}
                pskip = {f for f in skip if abs(face[f] - edge[f]) < 1e-6}
                boxes += box(g, sec, room, px - hp, floor_y, pz - hp, px + hp,
                             yb, pz + hp, sm, collide, top=False, bottom=False,
                             skip=pskip, mesh=mesh, tess=tess)
    return boxes


def ladder(g, sec, room, axis, at, sign, c0, c1, y0, y1, mat, rung_pitch=1.0,
           stile=0.2, standoff=0.4, collide=True, mesh=None):
    """Two stiles and rungs against the plane axis=at, on the `sign` side,
    spanning c0..c1 across.  Purely a visual and a collision, no nav."""
    lo, hi = sorted((at + sign * standoff, at + sign * (standoff + stile)))
    boxes = []
    for s0, s1 in ((c0, c0 + stile), (c1 - stile, c1)):
        p0, p1 = _p3(axis, lo, y0, s0), _p3(axis, hi, y1, s1)
        boxes += box(g, sec, room, p0[0], y0, p0[2], p1[0], y1, p1[2], mat,
                     collide, bottom=False, mesh=mesh, tess=0)
    yr = y0 + rung_pitch
    other = _other(axis)
    while yr < y1 - 0.5:
        p0, p1 = _p3(axis, lo, yr - stile / 2.0, c0 + stile), \
            _p3(axis, hi, yr + stile / 2.0, c1 - stile)
        box(g, sec, room, p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], mat, False,
            skip={'-' + other, '+' + other}, mesh=mesh, tess=0)
        yr += rung_pitch
    p0, p1 = _p3(axis, lo, y0, c0), _p3(axis, hi, y1, c1)
    return [(p0[0], y0, p0[2], p1[0], y1, p1[2])]


def catwalk(g, sec, room, x0, z0, x1, z1, y, thickness, mat_deck, mat_edge,
            mat_post, mat_rail, access=None, access_end='lo', mat_tread=None,
            mat_riser=None, floor_y=0.0, supports=True, mat_support=None,
            rail_height=3.5, stair_run=2.0, stair_rise=0.5, end_rail=True,
            collide=True, mesh=None, tess=None):
    """A platform railed both long sides, `access` a stair or a ladder at
    `access_end` ('lo'/'hi').  Returns (boxes, slope), slope None without a stair."""
    along = 'x' if (x1 - x0) >= (z1 - z0) else 'z'
    boxes = platform(g, sec, room, x0, z0, x1, z1, y, thickness, mat_deck,
                     mat_edge, supports, mat_support, floor_y=floor_y,
                     collide=collide, mesh=mesh, tess=tess)
    if along == 'x':
        a0, a1, c0, c1 = x0, x1, z0, z1
    else:
        a0, a1, c0, c1 = z0, z1, x0, x1
    for c in (c0, c1):
        p, q = _p3(along, a0, y, c), _p3(along, a1, y, c)
        boxes += railing(g, sec, room, p[0], p[2], q[0], q[2], y, mat_post,
                         mat_rail, rail_height, collide=collide, mesh=mesh,
                         tess=tess)
    open_a = a0 if access_end == 'lo' else a1
    closed_a = a1 if access_end == 'lo' else a0
    if end_rail:
        ends = [closed_a] if access else [a0, a1]
        for a in ends:
            p, q = _p3(along, a, y, c0), _p3(along, a, y, c1)
            boxes += railing(g, sec, room, p[0], p[2], q[0], q[2], y, mat_post,
                             mat_rail, rail_height, collide=collide, mesh=mesh,
                             tess=tess)
    slope = None
    sign = +1 if access_end == 'lo' else -1
    if access == 'stair':
        n = max(1, int(round((y - floor_y) / stair_rise)))
        rise = (y - floor_y) / n
        start = open_a - sign * n * stair_run
        p = _p3(along, start, 0.0, c0)
        sb, slope = stair(g, sec, room, along, sign, p[0], p[2], c1 - c0,
                          stair_run, rise, n, mat_tread or mat_deck,
                          mat_riser or mat_edge, mat_edge, y0=floor_y,
                          mesh=mesh, tess=tess)
        boxes += sb
    elif access == 'ladder':
        boxes += ladder(g, sec, room, along, open_a, -sign,
                        (c0 + c1) / 2.0 - 1.0, (c0 + c1) / 2.0 + 1.0, floor_y,
                        y + 3.0, mat_post, collide=collide, mesh=mesh)
    return boxes, slope


# ===========================================================================
# wall furniture
# ===========================================================================
def door_frame(g, sec, room, axis, at, s0, s1, y0, y1, depth, jamb, mat,
               sign=+1, collide=True, mesh=None, tess=None):
    """Jambs and a lintel round the aperture s0..s1 x y0..y1 in the wall plane
    axis=at, standing `depth` proud on the room side (`sign`)."""
    boxes = _wall_box(g, sec, room, axis, at, sign, s0 - jamb, s0, y0, y1,
                      depth, mat, collide, False, False, mesh, tess)
    boxes += _wall_box(g, sec, room, axis, at, sign, s1, s1 + jamb, y0, y1,
                       depth, mat, collide, False, False, mesh, tess)
    # the lintel sits on the jambs: its underside shows only over the opening
    boxes += _wall_box(g, sec, room, axis, at, sign, s0 - jamb, s1 + jamb, y1,
                       y1 + jamb, depth, mat, collide, True, False, mesh, tess)
    lo, hi = sorted((at, at + sign * depth))
    if axis == 'x':
        rect(g, sec, mat, room, 'y', y1, lo, s0, hi, s1, -1, collide, mesh, tess)
    else:
        rect(g, sec, mat, room, 'y', y1, s0, lo, s1, hi, -1, collide, mesh, tess)
    return boxes


def window_bay(g, sec, room, axis, at, s0, s1, y0, y1, depth, mat_reveal,
               mat_pane, sign=+1, collide=True, mesh=None, tess=None):
    """A window recessed `depth` into the wall away from the room (`sign` side):
    reveals facing in, a back pane in `mat_pane`.  The caller cuts the hole."""
    lo, hi = sorted((at, at - sign * depth))
    pane = at - sign * depth
    other = _other(axis)
    if axis == 'x':
        rect(g, sec, mat_reveal, room, other, s0, lo, y0, hi, y1, +1, collide, mesh, tess)
        rect(g, sec, mat_reveal, room, other, s1, lo, y0, hi, y1, -1, collide, mesh, tess)
        rect(g, sec, mat_reveal, room, 'y', y0, lo, s0, hi, s1, +1, collide, mesh, tess)
        rect(g, sec, mat_reveal, room, 'y', y1, lo, s0, hi, s1, -1, collide, mesh, tess)
    else:
        rect(g, sec, mat_reveal, room, other, s0, lo, y0, hi, y1, +1, collide, mesh, tess)
        rect(g, sec, mat_reveal, room, other, s1, lo, y0, hi, y1, -1, collide, mesh, tess)
        rect(g, sec, mat_reveal, room, 'y', y0, s0, lo, s1, hi, +1, collide, mesh, tess)
        rect(g, sec, mat_reveal, room, 'y', y1, s0, lo, s1, hi, -1, collide, mesh, tess)
    rect(g, sec, mat_pane, room, axis, pane, s0, y0, s1, y1, sign, collide,
         mesh, tess)
    return []


def cornice(g, sec, room, axis, at, s0, s1, y0, y1, depth, mat, sign=+1,
            top=True, bottom=True, collide=False, skip=(), mesh=None, tess=None):
    """A band proud of the wall plane axis=at by `depth` on the room side.
    top=False under the ceiling; `skip` the end faces that meet the side walls."""
    return _wall_box(g, sec, room, axis, at, sign, s0, s1, y0, y1, depth, mat,
                     collide, top, bottom, mesh, tess, skip)


def dado(g, sec, room, axis, at, s0, s1, y0, y1, depth, mat, sign=+1,
         top=True, bottom=True, collide=True, skip=(), mesh=None, tess=None):
    """A cornice at hand height: the same band, but it collides."""
    return _wall_box(g, sec, room, axis, at, sign, s0, s1, y0, y1, depth, mat,
                     collide, top, bottom, mesh, tess, skip)


def pilasters(g, sec, room, axis, at, s0, s1, y0, y1, pitch, width, depth,
              mat, sign=+1, mat_band=None, base_h=3.0, cap_h=1.5, proud=0.15,
              collide=True, top=True, ends=True, mesh=None, tess=None):
    """Columns against the wall plane axis=at every `pitch` along s0..s1,
    `depth` proud on the room side; bands as column()."""
    boxes = []
    hw = width / 2.0
    lo, hi = sorted((at, at + sign * depth))
    skip = {('-' if sign > 0 else '+') + axis}
    for a in _spread(s0 + hw, s1 - hw, pitch, ends=ends):
        if axis == 'x':
            boxes += column(g, sec, room, lo, a - hw, hi, a + hw, y0, y1, mat,
                            mat_band, base_h, cap_h, proud, collide, top,
                            False, skip, mesh, tess)
        else:
            boxes += column(g, sec, room, a - hw, lo, a + hw, hi, y0, y1, mat,
                            mat_band, base_h, cap_h, proud, collide, top,
                            False, skip, mesh, tess)
    return boxes


# ===========================================================================
# set pieces
# ===========================================================================
def container(g, sec, room, x0, z0, length, width, height, mat_side, mat_end,
              mat_top, axis='x', y0=0.0, rib_pitch=2.5, rib_proud=0.15,
              collide=True, bottom=False, mesh=None, tess=0):
    """A container corrugated by flat strips and proud rib boxes alternating
    every rib_pitch / 2; untessellated by default to stay under 200 quads."""
    a0 = x0 if axis == 'x' else z0
    c0 = z0 if axis == 'x' else x0
    a1, c1, y1 = a0 + length, c0 + width, y0 + height
    other = _other(axis)
    rect(g, sec, mat_end, room, axis, a0, c0, y0, c1, y1, -1, collide, mesh, tess)
    rect(g, sec, mat_end, room, axis, a1, c0, y0, c1, y1, +1, collide, mesh, tess)
    lo, hi = _p3(axis, a0, y0, c0), _p3(axis, a1, y1, c1)
    rect(g, sec, mat_top, room, 'y', y1, lo[0], lo[2], hi[0], hi[2], +1,
         collide, mesh, tess)
    if bottom:
        rect(g, sec, mat_top, room, 'y', y0, lo[0], lo[2], hi[0], hi[2], -1,
             collide, mesh, tess)
    n = max(1, int(round(length / rib_pitch)))
    pitch = length / n
    for c, sgn in ((c0, -1), (c1, +1)):
        for k in range(n):
            sa = a0 + k * pitch
            sm = sa + pitch / 2.0
            sb = sa + pitch
            rect(g, sec, mat_side, room, other, c, sa, y0, sm, y1, sgn,
                 collide, mesh, tess)
            rlo, rhi = sorted((c, c + sgn * rib_proud))
            p0, p1 = _p3(axis, sm, y0, rlo), _p3(axis, sb, y1, rhi)
            box(g, sec, room, p0[0], y0, p0[2], p1[0], y1, p1[2], mat_side,
                collide, bottom=False,
                skip={('-' if sgn > 0 else '+') + other}, mesh=mesh, tess=0)
    return [(lo[0], y0, lo[2], hi[0], y1, hi[2])]


def gantry(g, sec, room, axis, at, s0, s1, y_top, mat_leg, mat_beam, leg=1.5,
           beam_h=2.0, beam_w=None, floor_y=0.0, collide=True, mesh=None,
           tess=None):
    """A crane gantry: a leg at each end of s0..s1 and a bridge beam across
    the top.  The legs collide; the beam is overhead."""
    yb = y_top - beam_h
    boxes = []
    for a0, a1 in ((s0, s0 + leg), (s1 - leg, s1)):
        boxes += beam(g, sec, room, axis, at, a0, a1, floor_y, yb, leg, mat_leg,
                      collide, top=False, bottom=False, mesh=mesh, tess=tess)
    beam(g, sec, room, axis, at, s0, s1, yb, y_top, beam_w or leg, mat_beam,
         False, mesh=mesh, tess=tess)
    return boxes


def skylight_monitor(g, sec, room, x0, z0, x1, z1, y_roof, height, mat_wall,
                     mat_glass, mat_top, band=(2.0, 5.0), collide=False,
                     mesh=None, tess=None):
    """A roof monitor over a ceiling hole the caller cuts: walls facing in with
    a clerestory band in `mat_glass`, and a lid."""
    b0, b1 = y_roof + band[0], y_roof + band[1]
    y1 = y_roof + height
    strips = [(y_roof, b0, mat_wall), (b0, b1, mat_glass), (b1, y1, mat_wall)]
    for ya, yb, m in strips:
        if yb - ya < 1e-6:
            continue
        rect(g, sec, m, room, 'x', x0, z0, ya, z1, yb, +1, collide, mesh, tess)
        rect(g, sec, m, room, 'x', x1, z0, ya, z1, yb, -1, collide, mesh, tess)
        rect(g, sec, m, room, 'z', z0, x0, ya, x1, yb, +1, collide, mesh, tess)
        rect(g, sec, m, room, 'z', z1, x0, ya, x1, yb, -1, collide, mesh, tess)
    rect(g, sec, mat_top, room, 'y', y1, x0, z0, x1, z1, -1, collide, mesh, tess)
    return []


def bollard(g, sec, room, x, z, y0, mat, height=3.0, radius=0.5, collide=True,
            mesh=None):
    """An octagonal post standing on the floor at (x, z)."""
    ring = [(x + radius * math.cos(math.radians(22.5 + 45.0 * j)),
             z + radius * math.sin(math.radians(22.5 + 45.0 * j)))
            for j in range(9)]
    y1 = y0 + height
    for j in range(8):
        (xa, za), (xb, zb) = ring[j], ring[j + 1]
        na = math.radians(45.0 * (j + 1))
        g.quad(sec, mat, room, [(xa, y0, za), (xb, y0, zb), (xb, y1, zb),
                                (xa, y1, za)], (math.cos(na), 0.0, math.sin(na)),
               collide=collide, mesh=mesh)
    R = radius * math.cos(math.radians(22.5))
    h = radius * math.sin(math.radians(22.5))
    for quad in (((x - R, z - h), (x + R, z - h), (x + R, z + h), (x - R, z + h)),
                 ((x - R, z + h), (x + R, z + h), (x + h, z + R), (x - h, z + R)),
                 ((x - h, z - R), (x + h, z - R), (x + R, z - h), (x - R, z - h))):
        g.quad(sec, mat, room, [(px, y1, pz) for px, pz in quad],
               (0.0, 1.0, 0.0), collide=collide, mesh=mesh)
    return [(x - radius, y0, z - radius, x + radius, y1, z + radius)]


def kerb_run(g, sec, room, x0, z0, x1, z1, y0, height, mat, collide=True,
             skip=(), mesh=None, tess=None):
    """A low kerb on the floor over the footprint (x0, z0, x1, z1); `skip` the
    face against the wall."""
    return box(g, sec, room, x0, y0, z0, x1, y0 + height, z1, mat, collide,
               bottom=False, skip=skip, mesh=mesh, tess=tess)


# ===========================================================================
# nav
# ===========================================================================
def nav_exclusions(boxes, cell=6.0, slack=1.0, origin=(0.0, 0.0), floor_y=None,
                   head=6.5):
    """Solid boxes as merged (x0, z0, x1, z1) grid rects; a cell is taken only
    where a box overlaps it by over `slack` both ways, so corner posts take none."""
    ox, oz = origin
    taken = set()
    for x0, y0, z0, x1, y1, z1 in boxes:
        if floor_y is not None and (y0 > floor_y + head or y1 < floor_y - 0.5):
            continue
        i0 = int(math.floor((x0 - ox) / cell))
        i1 = int(math.ceil((x1 - ox) / cell))
        j0 = int(math.floor((z0 - oz) / cell))
        j1 = int(math.ceil((z1 - oz) / cell))
        for i in range(i0, i1):
            cx0, cx1 = ox + i * cell, ox + (i + 1) * cell
            if min(x1, cx1) - max(x0, cx0) <= slack:
                continue
            for j in range(j0, j1):
                cz0, cz1 = oz + j * cell, oz + (j + 1) * cell
                if min(z1, cz1) - max(z0, cz0) <= slack:
                    continue
                taken.add((i, j))
    # runs along x, then equal runs stacked along z
    runs = []
    for (i, j) in sorted(taken, key=lambda ij: (ij[1], ij[0])):
        if runs and runs[-1][1] == j and runs[-1][2] == i:
            runs[-1][2] = i + 1
        else:
            runs.append([i, j, i + 1, j + 1])
    merged = []
    for r in runs:
        if merged and merged[-1][0] == r[0] and merged[-1][2] == r[2] \
                and merged[-1][3] == r[1]:
            merged[-1][3] = r[3]
        else:
            merged.append(list(r))
    return [(ox + i0 * cell, oz + j0 * cell, ox + i1 * cell, oz + j1 * cell)
            for i0, j0, i1, j1 in merged]

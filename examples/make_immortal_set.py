#!/usr/bin/env python3
"""
Author sets\\immortal.bst -- the Immortal Arena, a fully custom world.

  - 200x200ft plaza inside 20ft crenellated walls, thick floor slab
  - two chasms through the floor, each crossed by one railed causeway
  - east and west plateaus 10ft up, one 48ft ramp each at opposite corners
  - a five-step ziggurat, an open centre crossroads, glass screens, four
    24ft pillars carrying an overhead beam square
  - per-face grayscale vertex colours as fake AO
  - one render mesh per material, one collision BVT over everything in both
    windings (triangle tests are one-sided), one all-zero surface-table entry
  - the six section lights repositioned over the arena

The donor set contributes header, skybox, lights, probes, materials and nav.
--copy writes a byte-identical donor rebuild instead.

Usage: make_immortal_set.py <abyss.bst> -o <out.bst> [--copy]
"""
import os
import struct
import sys

# Run from a checkout without installing:  PYTHONPATH=src python3 examples/make_immortal_set.py
from gbtvgr.mesh import smb
from gbtvgr.sets import bst

FLOOR_Y = 199.0          # proven: hero at 200.6 lands here
HALF = 100.0             # arena half-size
WALL_T = 3.0             # perimeter wall thickness (inner face at +-97)
WALL_H = 20.0
PLAT_Y = 10.0            # plateau height above the floor; the ramps climb
                         # it over 48ft = 2.5ft per 12ft nav cell, which is
                         # the steepest the cliff-link selftest allows
ZIG_Z = -71.0            # ziggurat centre (base z -80..-62)
# Chasms cut clean through the slab, each 1ft inside its nav exclusion so the
# walkable polygon always ends on solid floor. Causeway decks likewise overhang.
CHASM_N = (-59.0, 37.0, 59.0, 47.0)
CHASM_S = (-59.0, -47.0, 59.0, -37.0)
BRIDGE_N = (-13.0, 37.0, 13.0, 47.0)
BRIDGE_S = (-13.0, -47.0, 13.0, -37.0)
# Only materials 1 and 2 are ever used by donor set meshes. The other two
# render nothing at all on set geometry.
MAT_ROCK, MAT_FLOOR = 2, 1

# Planar world->tile mapping. B matches the section bbox padding so the whole
# arena footprint lands inside [0,1] with margin.
LM_B = HALF + 20.0

def lightmap_uv(p):
    u = (p[0] + LM_B) / (2.0 * LM_B)
    v = (p[2] + LM_B) / (2.0 * LM_B)
    u = min(0.996, max(0.004, u))
    v = min(0.996, max(0.004, v))
    return u, v

# per-face grayscale "fake AO" by normal direction
def face_color(n):
    if n[1] > 0.5: g = 0xFF
    elif n[1] < -0.5: g = 0x90
    elif abs(n[0]) > 0.5: g = 0xC8
    else: g = 0xB0
    return 0xFF000000 | (g << 16) | (g << 8) | g


class Geo:
    """Render meshes bucketed per material + one shared collision soup."""
    def __init__(self):
        self.buckets = {}          # mat -> {'v': bytes, 'nv': int, 'i': [], 'pts': []}
        self.cverts, self.ctris = [], []

    def _bucket(self, mat):
        return self.buckets.setdefault(mat, {'v': b'', 'nv': 0, 'i': [], 'pts': []})

    def quad(self, mat, p0, p1, p2, p3, uvscale=0.125):
        """One quad, both windings for render AND collision."""
        import math
        e1 = [p1[k] - p0[k] for k in range(3)]
        e2 = [p3[k] - p0[k] for k in range(3)]
        n = [e1[1]*e2[2] - e1[2]*e2[1], e1[2]*e2[0] - e1[0]*e2[2],
             e1[0]*e2[1] - e1[1]*e2[0]]
        l = math.sqrt(sum(x*x for x in n)) or 1.0
        n = [x / l for x in n]
        col = face_color(n)
        ax = max(range(3), key=lambda k: abs(n[k]))
        ua, va = (0, 2) if ax == 1 else ((1, 2) if ax == 0 else (0, 1))
        b = self._bucket(mat)
        base = b['nv']
        for p in (p0, p1, p2, p3):
            b['v'] += struct.pack('<3f', *p)
            b['v'] += struct.pack('<3f', *n)
            b['v'] += struct.pack('<2H', smb.f32_to_half(p[ua] * uvscale),
                                  smb.f32_to_half(p[va] * uvscale))
            lu, lv = lightmap_uv(p)
            b['v'] += struct.pack('<2H', smb.f32_to_half(lu),
                                  smb.f32_to_half(lv))              # lightmap uv
            b['v'] += struct.pack('<3f', 1.0, 0.0, 0.0)            # tangent
            b['v'] += struct.pack('<3f', 0.0, 0.0, 1.0)            # binormal
            b['v'] += struct.pack('<4I', col, col, col, col)
            b['pts'].append(p)
        b['nv'] += 4
        for a, c, d in ((0, 1, 2), (0, 2, 3), (2, 1, 0), (3, 2, 0)):
            b['i'].append((base + a, base + c, base + d))
        cb = len(self.cverts)
        self.cverts += [tuple(p) for p in (p0, p1, p2, p3)]
        for a, c, d in ((0, 1, 2), (0, 2, 3), (2, 1, 0), (3, 2, 0)):
            self.ctris.append((cb + a, cb + c, cb + d))

    def box(self, mat, mn, mx, uvscale=0.125):
        x0, y0, z0 = mn; x1, y1, z1 = mx
        self.quad(mat, (x0,y1,z0), (x1,y1,z0), (x1,y1,z1), (x0,y1,z1), uvscale)  # top
        self.quad(mat, (x0,y0,z1), (x1,y0,z1), (x1,y0,z0), (x0,y0,z0), uvscale)  # bottom
        self.quad(mat, (x0,y0,z0), (x1,y0,z0), (x1,y1,z0), (x0,y1,z0), uvscale)  # -z
        self.quad(mat, (x1,y0,z1), (x0,y0,z1), (x0,y1,z1), (x1,y1,z1), uvscale)  # +z
        self.quad(mat, (x0,y0,z1), (x0,y0,z0), (x0,y1,z0), (x0,y1,z1), uvscale)  # -x
        self.quad(mat, (x1,y0,z0), (x1,y0,z1), (x1,y1,z1), (x1,y1,z0), uvscale)  # +x

    def wedge_x(self, mat, x0, x1, y0, y1, z0, z1):
        """Ramp rising along +x: top slopes y0@x0 -> y1@x1."""
        self.quad(mat, (x0,y0,z0), (x1,y1,z0), (x1,y1,z1), (x0,y0,z1))  # slope
        self.quad(mat, (x0,FLOOR_Y,z0), (x0,y0,z0), (x0,y0,z1), (x0,FLOOR_Y,z1))
        self.quad(mat, (x0,FLOOR_Y,z0), (x1,FLOOR_Y,z0), (x1,y1,z0), (x0,y0,z0))
        self.quad(mat, (x0,y0,z1), (x1,y1,z1), (x1,FLOOR_Y,z1), (x0,FLOOR_Y,z1))


def build_arena():
    """One section, one collision BVT, two set-mesh materials. Two invariants keep
    the nav safe: obstacles inset inside their exclusion, floors wider than their poly."""
    g = Geo()
    y = FLOOR_Y

    # ---- floor slab minus the chasms.  Band sweep over the hole z-edges;
    #      the two holes have disjoint bands, so this is seven boxes.
    edges = [-HALF, CHASM_S[1], CHASM_S[3], CHASM_N[1], CHASM_N[3], HALF]
    for z0, z1 in zip(edges, edges[1:]):
        spans = [(-HALF, HALF)]
        for hole in (CHASM_N, CHASM_S):
            if not (hole[1] < z1 and hole[3] > z0):
                continue
            cut = []
            for sx0, sx1 in spans:
                if hole[0] > sx0:
                    cut.append((sx0, hole[0]))
                if hole[2] < sx1:
                    cut.append((hole[2], sx1))
            spans = cut
        for x0, x1 in spans:
            g.box(MAT_FLOOR, (x0, y - 4.0, z0), (x1, y, z1))

    # ---- the two causeways: deck flush with the floor, 1ft of lip on each
    #      side of the 24ft nav corridor, 3ft rails standing on the lip
    for b in (BRIDGE_N, BRIDGE_S):
        g.box(MAT_FLOOR, (b[0], y - 1.5, b[1]), (b[2], y, b[3]))
        g.box(MAT_ROCK, (b[0], y, b[1]), (b[0] + 1.0, y + 3.0, b[3]))
        g.box(MAT_ROCK, (b[2] - 1.0, y, b[1]), (b[2], y + 3.0, b[3]))

    # ---- perimeter walls + crenellations (a merlon every 24ft)
    t = WALL_T
    g.box(MAT_ROCK, (-HALF, y, HALF - t), (HALF, y + WALL_H, HALF))       # N
    g.box(MAT_ROCK, (-HALF, y, -HALF), (HALF, y + WALL_H, -HALF + t))     # S
    g.box(MAT_ROCK, (HALF - t, y, -HALF), (HALF, y + WALL_H, HALF))       # E
    g.box(MAT_ROCK, (-HALF, y, -HALF), (-HALF + t, y + WALL_H, HALF))     # W
    m0, m1 = y + WALL_H, y + WALL_H + 4.0
    for k in range(-4, 4):
        a, b = k * 24.0 + 8.0, k * 24.0 + 16.0
        g.box(MAT_ROCK, (a, m0, HALF - t), (b, m1, HALF))
        g.box(MAT_ROCK, (a, m0, -HALF), (b, m1, -HALF + t))
        g.box(MAT_ROCK, (HALF - t, m0, a), (HALF, m1, b))
        g.box(MAT_ROCK, (-HALF, m0, a), (-HALF + t, m1, b))

    # ---- four pillars + the beam square they carry, 20ft overhead
    for sx in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            cx, cz = sx * 55.0, sz * 55.0
            g.box(MAT_ROCK, (cx - 3, y, cz - 3), (cx + 3, y + 24.0, cz + 3))
            g.box(MAT_ROCK, (cx - 5, y + 24.0, cz - 5),
                  (cx + 5, y + 26.5, cz + 5))
    for s in (-1.0, 1.0):
        g.box(MAT_ROCK, (-52.0, y + 20.0, s * 55.0 - 2.0),
              (52.0, y + 24.0, s * 55.0 + 2.0))
        g.box(MAT_ROCK, (s * 55.0 - 2.0, y + 20.0, -52.0),
              (s * 55.0 + 2.0, y + 24.0, 52.0))

    # ---- the ziggurat (south), five 3ft steps
    for i in range(5):
        hx, hz = 20.0 - 3.0 * i, 9.0 - 1.5 * i
        g.box(MAT_ROCK, (-hx, y + 3.0 * i, ZIG_Z - hz),
              (hx, y + 3.0 * (i + 1), ZIG_Z + hz))

    # ---- plateaus, ramps and ramp rails (nav islands, see ISLANDS)
    for isl in ISLANDS:
        px0, pz0, px1, pz1 = isl['plateau']
        up = isl['crest'] > isl['foot']
        # 1ft of lip on every free edge; the ramp side butts flush
        g.box(MAT_FLOOR, (px0 if up else px0 - 1.0, y - 4.0, pz0 - 1.0),
              (px1 + 1.0 if up else px1, y + PLAT_Y, pz1 + 1.0))
        rx0, rz0, rx1, rz1 = isl['ramp']
        xa, xb = min(isl['foot'], isl['crest']), max(isl['foot'], isl['crest'])
        g.wedge_x(MAT_FLOOR, xa, xb, ramp_h(isl, xa) + 0.4,
                  ramp_h(isl, xb), rz0 - 1.0, rz1 + 1.0)
        for k in range(int(round((xb - xa) / 12.0))):
            sa, sb = xa + 12.0 * k, xa + 12.0 * (k + 1)
            lo = min(ramp_h(isl, sa), ramp_h(isl, sb)) - 1.0
            hi = max(ramp_h(isl, sa), ramp_h(isl, sb)) + 3.0
            g.box(MAT_ROCK, (sa, lo, rz0 - 1.0), (sb, hi, rz0))
            g.box(MAT_ROCK, (sa, lo, rz1), (sb, hi, rz1 + 1.0))

    # ---- glass screens: cover either side of the centre line
    for gx0, gx1 in ((-35.0, -13.0), (13.0, 35.0)):
        g.box(MAT_ROCK, (gx0, y, 5.0), (gx1, y + 9.0, 6.0), uvscale=0.05)

    # ---- the centre stays empty: the crossroads must be walkable straight
    #      through, so the furnace there is coal props, not geometry.
    return g


# Grid-aligned so the overlap test removes exactly the intended cells and not
# one ring more; inflating these instead splits the nav in two.
NAV_EXCLUDE = [
    (-60.0, 36.0, -12.0, 48.0), (12.0, 36.0, 60.0, 48.0),        # N chasm
    (-60.0, -48.0, -12.0, -36.0), (12.0, -48.0, 60.0, -36.0),    # S chasm
    (48.0, 48.0, 60.0, 60.0), (-60.0, 48.0, -48.0, 60.0),        # pillars
    (48.0, -60.0, 60.0, -48.0), (-60.0, -60.0, -48.0, -48.0),
    (-24.0, -84.0, 24.0, -60.0),                                 # ziggurat
    (-36.0, 0.0, -12.0, 12.0), (12.0, 0.0, 36.0, 12.0),          # glass
]
# Nav islands: walkable nav at height, each a plateau plus its one ramp. Corner
# heights come from ramp_h(), so the link rule leaves the ramp foot as the only way up.
ISLANDS = [
    {'plateau': (60.0, -24.0, 84.0, 24.0),                       # east
     'ramp': (12.0, -24.0, 60.0, -12.0), 'foot': 12.0, 'crest': 60.0},
    {'plateau': (-84.0, -24.0, -60.0, 24.0),                     # west
     'ramp': (-60.0, 12.0, -12.0, 24.0), 'foot': -12.0, 'crest': -60.0},
]
NAV_LO, NAV_CELL, NAV_N = -96.0, 12.0, 16
NAV_CLAMP = 94.0   # walls at +-97; cells reaching +-96 made string-pulled
                   # paths hug the boundary (a mini ran INTO the wall)


def ramp_h(isl, x):
    """Island surface height at world x (ramp slope, then plateau top)."""
    t = (x - isl['foot']) / (isl['crest'] - isl['foot'])
    return FLOOR_Y + PLAT_Y * min(1.0, max(0.0, t))


def cell_island(cx0, cz0, cx1, cz1):
    """Index of the island a whole cell sits inside, or None for floor."""
    e = 0.01
    for k, isl in enumerate(ISLANDS):
        for r in (isl['plateau'], isl['ramp']):
            if (cx0 >= r[0] - e and cx1 <= r[2] + e and
                    cz0 >= r[1] - e and cz1 <= r[3] + e):
                return k
    return None


def nav_selftest(cells, isl, verts, meta, lo, cell):
    """Offline gate: the nav bugs found in-game (half-zones, wall-hugging,
    unreachable platform) must be impossible to ship again."""
    idx = {}
    for k, (key, cen, nb, ring) in enumerate(meta):
        idx[key] = k
    adj = {k: [n for n in meta[k][2] if n >= 0] for k in range(len(meta))}

    def bfs(start, allowed=None):
        seen, q = {start}, [start]
        while q:
            c = q.pop()
            for n in adj[c]:
                if n not in seen and (allowed is None or n in allowed):
                    seen.add(n); q.append(n)
        return seen

    def cell_at(x, z):
        key = (int((x - lo) // cell), int((z - lo) // cell))
        assert key in cells, 'no nav cell at (%g, %g)' % (x, z)
        return idx[key]

    # 1. one connected component (probes: the hero spawn, every corner, the
    #    two causeways, both ramps, both plateaus, every spawner position)
    home = cell_at(0.0, -30.0)
    comp = bfs(home)
    assert len(comp) == len(meta), \
        'nav split: %d of %d nodes reachable' % (len(comp), len(meta))
    for x, z in ((-88, -88), (88, -88), (-88, 88), (88, 88), (0, -88),
                 (0, 42), (0, -42), (30, -18), (-30, 18), (72, 0), (-72, 0),
                 (44, 0), (-44, 0), (75, 75), (-75, 75), (75, -75),
                 (-75, -75), (0, 82), (0, 0)):
        assert cell_at(x, z) in comp, 'probe (%g,%g) unreachable' % (x, z)

    # 1b. the centre is a CROSSROADS, not an obstacle: all four cells that
    #     meet at (0,0) exist and are linked to each other in a ring
    quad = [cell_at(-6.0, -6.0), cell_at(-6.0, 6.0),
            cell_at(6.0, 6.0), cell_at(6.0, -6.0)]
    for a in range(4):
        assert quad[(a + 1) % 4] in adj[quad[a]], 'the arena centre is walled'

    # 2. a real corridor SOUTH of the ziggurat: SW to SE staying below z=-77
    south = {k for k in range(len(meta)) if meta[k][1][2] < -77.0}
    sw, se = cell_at(-40.0, -88.0), cell_at(40.0, -88.0)
    assert se in bfs(sw, allowed=south), 'no corridor south of the ziggurat'

    # 2b. both causeways are the ONLY way across their chasm at |x| < 60:
    #     the cells flanking each deck must be gone
    for x, z in ((-30, 42), (30, 42), (-30, -42), (30, -42)):
        key = (int((x - lo) // cell), int((z - lo) // cell))
        assert key not in cells, 'chasm cell (%g,%g) is still walkable' % (x, z)

    # 3. every vertex inside the +-94 wall margin
    for v in verts:
        assert abs(v[0]) <= NAV_CLAMP + 0.001 and abs(v[2]) <= NAV_CLAMP + 0.001

    # 4. corridor width: every used link's shared edge is >= 8ft long
    for k, (key, cen, nb, ring) in enumerate(meta):
        for a in range(4):
            if nb[a] < 0:
                continue
            e0, e1 = verts[ring[a]], verts[ring[(a + 1) % 4]]
            w = max(abs(e0[0] - e1[0]), abs(e0[2] - e1[2]))
            assert w >= 8.0, 'link %d->%d edge %.1fft' % (k, nb[a], w)

    # 5. no link may bridge a >2ft step, and 3.0 is the tightest bound that
    #    still admits the ramp itself
    for k, (key, cen, nb, ring) in enumerate(meta):
        for n in nb:
            if n >= 0:
                assert abs(meta[n][1][1] - cen[1]) < 3.0, \
                    'cliff link %d->%d' % (k, n)
    # 6. each island is entered from the floor exactly once
    for w in range(len(ISLANDS)):
        feet = 0
        for k, (key, cen, nb, ring) in enumerate(meta):
            if isl[key] != w:
                continue
            for n in nb:
                if n >= 0 and isl[meta[n][0]] is None:
                    feet += 1
        assert feet == 1, 'island %d has %d floor links (want 1)' % (w, feet)
    n_isl = sum(1 for key in cells if isl[key] is not None)
    print('nav selftest OK: %d nodes (%d island), %d verts, one component, '
          'south corridor, 2 causeways, %d ramp feet, +-%g bounds' %
          (len(meta), n_isl, len(verts), len(ISLANDS), NAV_CLAMP))


def build_nav():
    """A walkable grid for the arena floor. Floor and ceiling plane heights are
    absolute, not relative -- get them wrong and every containment test fails."""
    y = FLOOR_Y
    lo, cell, ncell = NAV_LO, NAV_CELL, NAV_N

    def blocked(cx0, cz0, cx1, cz1):
        for ex0, ez0, ex1, ez1 in NAV_EXCLUDE:
            if cx0 < ex1 and cx1 > ex0 and cz0 < ez1 and cz1 > ez0:
                return True
        return False

    cells, isl = {}, {}
    for i in range(ncell):
        for j in range(ncell):
            x0, z0 = lo + i * cell, lo + j * cell
            w = cell_island(x0, z0, x0 + cell, z0 + cell)
            if w is not None or not blocked(x0, z0, x0 + cell, z0 + cell):
                isl[(i, j)] = w
                cells[(i, j)] = len(cells)

    def cornh(key, i, j):     # corner height as seen by cell `key`
        w = isl[key]
        return y if w is None else ramp_h(ISLANDS[w], lo + i * cell)

    def clamp(v):
        return max(-NAV_CLAMP, min(NAV_CLAMP, v))

    vids, verts = {}, []
    def vid(key, i, j):
        hh = cornh(key, i, j)
        k = (i, j, round(hh * 8))
        if k not in vids:
            vids[k] = len(verts)
            verts.append((clamp(lo + i * cell), hh, clamp(lo + j * cell)))
        return vids[k]

    # A link is real only when both cells agree on both shared corner heights,
    # which cuts every link across a ledge and up a ramp's rising sides.
    EDGE_CORNERS = {(-1, 0): ((0, 0), (0, 1)), (0, 1): ((0, 1), (1, 1)),
                    (1, 0): ((1, 0), (1, 1)), (0, -1): ((0, 0), (1, 0))}
    def link(key, d):
        i, j = key
        nk = (i + d[0], j + d[1])
        if nk not in cells:
            return -1
        for ci, cj in EDGE_CORNERS[d]:
            if abs(cornh(key, i + ci, j + cj) -
                   cornh(nk, i + ci, j + cj)) > 0.5:
                return -1
        return cells[nk]

    nodes, meta = [], []
    for key in cells:
        i, j = key
        # CW ring in xz: (x0,z0) (x0,z1) (x1,z1) (x1,z0)
        ring = [vid(key, i, j), vid(key, i, j + 1),
                vid(key, i + 1, j + 1), vid(key, i + 1, j)]
        nb = [link(key, (-1, 0)), link(key, (0, 1)),
              link(key, (1, 0)), link(key, (0, -1))]
        pts = [verts[v] for v in ring]
        cx = sum(p[0] for p in pts) / 4.0
        cy = sum(p[1] for p in pts) / 4.0
        cz = sum(p[2] for p in pts) / 4.0
        f4 = min(p[1] for p in pts) - 1.0
        f5 = max(p[1] for p in pts) + 60.0
        hdr = struct.pack('<I3f2f6f', 0, cx, cy, cz, f4, f5,
                          0.0, 1.0, 0.0, 0.0, 1.0, 0.0)
        nodes.append({'hdr': hdr,
                      'v': struct.pack('<4I', *ring),
                      'nb': struct.pack('<4i', *nb),
                      'd0': struct.pack('<4I', 0, 0, 0, 0),
                      'f15c': struct.pack('<I', 0)})
        meta.append((key, (cx, cy, cz), nb, ring))
    nav_selftest(cells, isl, verts, meta, lo, cell)
    # The node tail u32 is a group index into 'extra', not a self index: leaving
    # extra empty while indices are nonzero writes into a 0-byte allocation.
    extra = [{'a': struct.pack('<I', 0x1234ABCD), 'b': struct.pack('<I', 3),
              'c': struct.pack('<f', 1.0), 'rows': b''}]
    return {'empty': False, 'f': 2, 'nverts': len(verts), 'nnodes': len(nodes),
            'nextra': 1, 'nparts': 0,
            'verts': b''.join(struct.pack('<3f', *v) for v in verts),
            'nodes': nodes, 'extra': extra, 'parts': []}


def one_leaf_bsp():
    return struct.pack('<4f', 0, 0, 0, 0) + \
        struct.pack('<2h', 0, -1) + struct.pack('<2h', -1, 0) + \
        struct.pack('<6f', -99999, -99999, -99999, 99999, 99999, 99999)


def main():
    args = sys.argv[1:]
    src = args[0]
    out = args[args.index('-o') + 1]

    d = open(src, 'rb').read()
    m = bst.parse(d)

    if '--copy' in args:
        blob = bst.build(m)
        assert blob == d, 'copy mode must rebuild byte-identical'
        open(out, 'wb').write(blob)
        print('%s: byte-identical abyss rebuild' % out)
        return

    sec = m['sections'][0]
    donor = next(me for me in sec['meshes']
                 if 'pkt' in me and smb.decl_stride(me['pkt']['decl']) == 72)
    # h70 is the material index; f6c is the geometry-type field, constant across
    # a donor section's meshes.
    decl, donor_f6c = donor['pkt']['decl'], donor['f6c']
    # names3 filetag, copied verbatim from a real lit section: constant per .bst,
    # role unresolved, and not validated at load.
    donor_filetag = m['sections'][1]['names3'][0x5C:0x64]

    g = build_arena()
    meshes, blob = [], b''
    for mat in sorted(g.buckets):
        b = g.buckets[mat]
        idata = b''.join(struct.pack('<3H', *t) for t in b['i'])
        xs = [p[0] for p in b['pts']]; ys = [p[1] for p in b['pts']]
        zs = [p[2] for p in b['pts']]
        pkt = {'datasize': len(b['v']) + len(idata), 'decl': decl,
               'nverts': b['nv'], 'nprims': len(b['i']), 'f18': 0, 'morphs': [],
               'bbox': struct.pack('<6f', min(xs), min(ys), min(zs),
                                   max(xs), max(ys), max(zs)),
               'vdata': b['v'], 'idata': idata, 'mdata': []}
        name = ('immortal_m%d' % mat).encode()
        meshes.append({'h70': struct.pack('<H', mat),
                       'name': name + b'\0' * (0x20 - len(name)),
                       'flags': 0x4, 'f6c': donor_f6c, 'pkt': pkt})
        blob += b['v'] + idata
    blob += b'\0' * ((-len(blob)) % 32)

    m['sections'] = [sec]
    sec['meshes'] = meshes
    sec['bbox'] = struct.pack('<6f', -HALF - 20, FLOOR_Y - 60, -HALF - 20,
                              HALF + 20, FLOOR_Y + 120, HALF + 20)

    # Three lit slots naming our own tiles under a set-derived path, section
    # index 0 -- the rebuilt .bst has only the one section.
    def tex_slot(path, filetag):
        name = path.encode() + b'\0'
        assert len(name) <= 0x40, 'lightmap path too long for STexSlot.name'
        s = b'\xCD' * 8 + name + b'\xCD' * (0x40 - len(name))
        s += struct.pack('<5I', 3, 256, 256, 0, 0)   # STexDesc: fmt3, 256x256
        s += filetag
        assert len(s) == 100
        return s

    sec['names3'] = b''.join(
        tex_slot('lightmap\\immortal\\0_%d.tga' % i, donor_filetag)
        for i in range(3))

    # Dep entries for the new tiles. The hash role is unresolved, so each donor
    # tile's own 16 bytes are copied through rather than invented.
    lm_deps = [(n, h) for n, h in m['deps'] if n.startswith(b'art\\lightmap\\')]
    if lm_deps:
        for i, (_, h) in enumerate(lm_deps[:3]):
            m['deps'].append((('art\\lightmap\\immortal\\0_%d.tex' % i).encode(), h))
    else:
        print('note: donor deps table has no lightmap entries -- skipped append')

    root, arena = smb.build_bvt(g.cverts, g.ctris, [0] * len(g.ctris))
    def inflate(n):
        b = list(struct.unpack('<6f', n['bbox']))
        n['bbox'] = struct.pack('<6f', b[0]-1, b[1]-1, b[2]-1,
                                b[3]+1, b[4]+1, b[5]+1)
        for k in n.get('kids', ()): inflate(k)
    inflate(root)
    sec['bvtflag'] = arena
    sec['bvt'] = root

    # One all-zero surface entry, so every collision triangle indexes 0; a
    # nonzero second u32 would be an exclusion mask against the query channel.
    name = b'immortal_floor'
    sec['lrefs'] = [{'name': name + b'\0' * (0x20 - len(name)),
                     'v': struct.pack('<2I6f', 0, 0,
                                      -HALF - 2, FLOOR_Y - 6, -HALF - 2,
                                      HALF + 2, FLOOR_Y + 40, HALF + 2)}]

    # reposition the six lights the kept section references (arr818 =
    # 1,2,6,7,8,9): four corner-pile lights, one center, one furnace-north
    for li, pos in zip((1, 2, 6, 7, 8, 9),
                       ((-75, 216, -75), (75, 216, -75), (-75, 216, 75),
                        (75, 216, 75), (0, 216, 0), (0, 210, 42))):
        m['lights'][li]['pos'] = struct.pack('<3f', *pos)

    m['nav'] = build_nav()
    m['bsp'] = one_leaf_bsp()
    m['bspglob'] = struct.pack('<I', 0)
    m['watervis'] = m['watervis'][:1]
    sec['fa20'] = len(blob)
    m['datasize'] = len(blob)
    sec['blob'] = blob
    sec['dataofs'] = 0
    sec['datagap'] = b''
    pre = len(bst.build(m)) - len(blob)
    sec['dataofs'] = pre + ((-pre) % 32)
    sec['datagap'] = b'\0' * (sec['dataofs'] - pre)
    final = bst.build(m)

    m2 = bst.parse(final)
    s2 = m2['sections'][0]
    nv = sum(me['pkt']['nverts'] for me in s2['meshes'])
    nt = sum(me['pkt']['nprims'] for me in s2['meshes'])
    open(out, 'wb').write(final)
    print('%s: THE IMMORTAL ARENA -- %d bytes, %d meshes (%d verts, %d tris), '
          '%d collision tris, %d lights repositioned' %
          (out, len(final), len(s2['meshes']), nv, nt, len(g.ctris), 6))


if __name__ == '__main__':
    main()

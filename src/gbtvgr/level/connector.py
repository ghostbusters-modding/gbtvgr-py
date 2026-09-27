"""A generated corridor between two host sections, appended as a new section: a loft
between the rooms' real mouths, or a plain box.
Copyright (C) 2026 Colin Sullivan and contributors
SPDX-License-Identifier: GPL-2.0-only
"""
import struct

import numpy as np

from ..mesh.bvt import bvt_bbox, bvt_collect, union_bbox
from ..sets import geom
from . import caps
from . import geometry_primitives as primitives
from .geometry_mesh import EditMesh
from . import lighting
from . import lights as glights
from . import sections as gsec
from . import transplant as gtr
from .mathutil import F32
from .navmesh import GROUP_TAG
from .scene import Document, MeshNode, SectionNode

AX = {'x': 0, 'y': 1, 'z': 2}
SECTION_PAD = 8.0          # matches setfile.py's own shipped-section padding
END_FACES = {'x': (6, 7, 10, 11), 'z': (4, 5, 8, 9)}     # box() quads to drop for an open end
PROFILES = ('loft', 'box', 'capped')
PROFILE_ALIASES = {'mask': 'loft'}     # build scripts written for the capped box get the loft
RING_LIP = 0.02            # the end ring sits this far inside the rock line, never outside it
RING_FLANGE = 0.4          # and its flange reaches this far out behind the rock
NAV_LIFT = 0.5             # the nav strip is as wide as the loft this far above its floor


FLOOR_INSETS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)


def _probe_floor(bvt, plane, sign, cross, axis):
    """Floor near a room's face, stepping further inward until something is hit:
    a room's real floor commonly stops well short of its (padded, decorated)
    stored bbox, so a single fixed inset can probe empty air."""
    for inset in FLOOR_INSETS:
        p = plane - sign * inset
        y = floor_height(bvt, p, cross) if axis == 'x' else floor_height(bvt, cross, p)
        if y is not None:
            return y
    return None


def floor_height(bvt, x, z):
    """Lowest surface a vertical line at (x, z) crosses: the floor, since
    nothing here ships geometry under its own floor."""
    if bvt is None:
        return None
    verts, tris, _surf, _flags = bvt_collect(bvt)
    best = None
    for a, b, c in tris:
        ax_, ay, az = verts[a]
        bx, by, bz = verts[b]
        cx, cy, cz = verts[c]
        d = (bz - cz) * (ax_ - cx) + (cx - bx) * (az - cz)
        if abs(d) < 1e-9:
            continue
        w0 = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) / d
        w1 = ((cz - az) * (x - cx) + (ax_ - cx) * (z - cz)) / d
        w2 = 1.0 - w0 - w1
        if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6:
            continue
        y = w0 * ay + w1 * by + w2 * cy
        if best is None or y < best:
            best = y
    return best


def _bbox(sec):
    lo_x, lo_y, lo_z, hi_x, hi_y, hi_z = struct.unpack('<6f', sec['bbox'])
    return (lo_x, lo_y, lo_z), (hi_x, hi_y, hi_z)


def _face_plane(bb, axis, sign):
    lo, hi = bb
    return hi[AX[axis]] if sign > 0 else lo[AX[axis]]


def _cross_span(a_bb, b_bb, ca, width):
    """The doorway's cross-axis range: centred on where the two rooms actually
    overlap, since averaging their centres can miss a small room entirely."""
    a_lo, a_hi = a_bb[0][ca], a_bb[1][ca]
    b_lo, b_hi = b_bb[0][ca], b_bb[1][ca]
    lo_ov, hi_ov = max(a_lo, b_lo), min(a_hi, b_hi)
    center = (lo_ov + hi_ov) / 2.0 if lo_ov < hi_ov else \
        ((a_lo + a_hi) / 2.0 + (b_lo + b_hi) / 2.0) / 2.0
    return center - width / 2.0, center + width / 2.0


def _material_entry(ref):
    """Same wire shape as setfile.py's own _material_entry (kept local: that
    module belongs to another agent this wave)."""
    b = ref.replace('/', '\\').encode('latin1')
    return {'ref': (b, b'\0' * ((-(len(b) + 1)) % 4)), 'embedded': None}


def _corridor_mesh(axis, la_lo, la_hi, ca_lo, ca_hi, floor_y, height, uv_scale):
    """An open-ended box: primitives.box() minus the two end caps on `axis`."""
    length = la_hi - la_lo
    if axis == 'x':
        m = primitives.box(length, height, ca_hi - ca_lo, uv_scale=uv_scale, inside=True)
    else:
        m = primitives.box(ca_hi - ca_lo, height, length, uv_scale=uv_scale, inside=True)
    m.delete_faces(list(END_FACES[axis]))
    if axis == 'x':
        m.verts[:, 0] += (la_lo + la_hi) / 2.0
        m.verts[:, 2] += (ca_lo + ca_hi) / 2.0
    else:
        m.verts[:, 0] += (ca_lo + ca_hi) / 2.0
        m.verts[:, 2] += (la_lo + la_hi) / 2.0
    m.verts[:, 1] += floor_y
    return m


def _loft_mesh(axis, rim_lo, rim_hi, uv_scale, lip=RING_LIP, flange=RING_FLANGE):
    """Quads between two rims of equal n, faces inward; each end ring runs on into its
    room to the depth it was probed at and ends in a flange behind the rock.
    Returns (EditMesh, per-corner lightmap uv)."""
    if rim_lo.n != rim_hi.n:
        raise ValueError('rims of %d and %d points cannot be lofted' % (rim_lo.n, rim_hi.n))
    n = rim_lo.n
    ax, ca = AX[axis], AX['z' if axis == 'x' else 'x']
    rings, arcs, centres, bands = [], [], [], []

    def ring(rim, inset=0.0, push=0.0):
        rings.append(rim.points3(inset, push))
        arcs.append(rim.arcs)
        centres.append(rim.centre)
        return len(rings) - 1

    for rim in (rim_lo, rim_hi):
        # the end ring sits a lip inside the rock line and the flange reaches out past it:
        # rays from either side then meet the loft before the rock's ragged end
        inner = ring(rim, rim.depth, -lip)
        plane = ring(rim)
        out = ring(rim, rim.depth, np.where(rim.clamped, 0.0, flange))
        bands.append((inner, plane, None))
        bands.append((inner, out, -rim.sign))
        if rim is rim_lo:
            lo_plane = plane
        else:
            bands.append((lo_plane, plane, None))
    verts = np.concatenate(rings)
    t_lo, t_hi = verts[:, ax].min(), verts[:, ax].max()
    faces, uvs, lmuv = [], [], []
    for r0, r1, facing in bands:
        cen = np.zeros(3)
        cen[ca] = (centres[r0][0] + centres[r1][0]) / 2.0
        cen[1] = (centres[r0][1] + centres[r1][1]) / 2.0
        for i in range(n):
            j = (i + 1) % n
            a, b = r0 * n + i, r0 * n + j
            c, d = r1 * n + j, r1 * n + i
            # (vertex, ring, arc index): the wrap corner takes the perimeter, not 0
            quad = [(a, r0, i), (b, r0, i + 1), (c, r1, i + 1), (d, r1, i)]
            for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                vi = [q[0] for q in tri]
                nrm = np.cross(verts[vi[1]] - verts[vi[0]], verts[vi[2]] - verts[vi[0]])
                if np.linalg.norm(nrm) < 1e-10:
                    continue
                if facing is None:
                    cen[ax] = verts[vi].mean(axis=0)[ax]
                    flip = np.dot(nrm, cen - verts[vi].mean(axis=0)) < 0
                else:
                    flip = nrm[ax] * facing < 0
                if flip:
                    tri = (tri[0], tri[2], tri[1])
                faces.append([q[0] for q in tri])
                uvs.append([(verts[q[0]][ax] / uv_scale, arcs[q[1]][q[2]] / uv_scale) for q in tri])
                lmuv.append([((verts[q[0]][ax] - t_lo) / max(t_hi - t_lo, 1e-6), q[2] / float(n))
                             for q in tri])
    mesh = EditMesh(verts.astype(F32), np.array(faces, np.int32), np.array(uvs, F32),
                    np.zeros(len(faces), np.int32), smooth=True)
    return mesh, np.array(lmuv, F32)


def _section_record(mesh, material_name, name, lmuv=None):
    """mesh -> one host-section-record dict, the shape append_section expects
    (compile_set's own per-section rec, built here for a single ad-hoc section)."""
    doc = Document('connector')
    sec = SectionNode(name)
    lo, hi = mesh.bbox()
    pad = np.array([2.0, 2.0, 2.0], F32)
    sec.set_bounds(lo - pad, hi + pad)
    doc.add_node(sec, doc.sections)
    node = MeshNode(name + '_geo')
    node.mesh = mesh
    node.props['materials'] = [material_name]
    sec.add(node)

    meshes = gsec.gather_meshes(sec, None if lmuv is None else {node.id: lmuv})
    mesh_recs, blob, mesh_boxes = [], b'', []
    for mo in meshes:
        rec, box = gsec.mesh_record(mo, 0, lightmapped=lmuv is not None)
        mesh_recs.append(rec)
        blob += rec['pkt']['vdata'] + rec['pkt']['idata']
        mesh_boxes.append(box)
    verts, tris, surf, flags, entries = gsec.collect_collision(doc, sec)
    root, arena = gsec.build_section_bvt(verts, tris, surf, flags)
    slo, shi = np.array(sec.lo, np.float64), np.array(sec.hi, np.float64)
    for blo, bhi in mesh_boxes:
        slo, shi = np.minimum(slo, blo), np.maximum(shi, bhi)
    if root is not None:
        rb = bvt_bbox(root)
        slo, shi = np.minimum(slo, rb[:3]), np.maximum(shi, rb[3:])
    slo, shi = slo - SECTION_PAD, shi + SECTION_PAD
    blob += b'\0' * ((-len(blob)) % 32)
    nb = name.encode('latin1')[:0x3F]
    return {
        'name': nb + b'\0' * (0x40 - len(nb)),
        'b1': 1, 'b2': 0, 'b3': 1, 'f3b4': 0, 'f3b8': 0, 'skip': 0, 'f3c0': 0,
        'names3': glights.tex_slot_empty() * 3,
        'extnames': [],
        'portidx': b'',
        'meshes': mesh_recs,
        'lrefs': gsec.surface_entries(entries, (slo, shi)),
        'bvtflag': arena, 'bvt': root,
        'arr4f4': b'', 'arr818': b'', 'arr470': b'',
        'dataofs': 0, 'fa20': len(blob),
        'bbox': struct.pack('<6f', *slo, *shi),
        'blob': blob, 'datagap': b'',
    }


def _dedupe_material(host_m, material_name):
    """append_section always appends a fresh material; reuse an existing one
    of the same name instead of shipping a duplicate."""
    new_idx = len(host_m['materials']) - 1
    canon = material_name.replace('/', '\\').lower()
    for i in range(new_idx):
        ref = host_m['materials'][i]['ref'][0].decode('latin1').lower()
        if ref == canon:
            for me in host_m['sections'][-1]['meshes']:
                if struct.unpack('<H', me['h70'])[0] == new_idx:
                    me['h70'] = struct.pack('<H', i)
            del host_m['materials'][new_idx]
            host_m['mat16'] = host_m['mat16'][:new_idx * 2]
            return


# --- BSP: insert_bsp_leaf only wraps the tree as a bookend (new leaf entirely
# outside every existing section's bbox). A corridor between two rooms is never
# that: it shares its meeting faces with sections that are themselves part of
# the tree's bounding envelope, so no separating plane exists on any axis. This
# carves the corridor's own slab out of the root instead of trying to bookend.
def _bsp_wedge_split(host_m, new_idx, axis, lo_plane, hi_plane):
    ax = AX[axis]
    nodes = geom.bsp_nodes(host_m)
    for _plane, sec, _f, _b, bb in nodes:
        if sec < 0:
            continue
        if bb[ax] >= lo_plane and bb[ax + 3] <= hi_plane:
            raise ValueError(
                "section %d (%s) sits inside the corridor's own span on %s "
                "[%.2f, %.2f]; a straight corridor cannot pass through it"
                % (sec, geom.name_str(host_m['sections'][sec]['name']), axis,
                   lo_plane, hi_plane))
    root_bb = nodes[0][4]
    sec_bb = struct.unpack('<6f', host_m['sections'][new_idx]['bbox'])
    top = union_bbox([root_bb, sec_bb])
    n_old = len(nodes)
    old = host_m['bsp']
    shifted = bytearray()
    for i in range(n_old):
        rec = bytearray(old[i * gtr.BSP_NODE:(i + 1) * gtr.BSP_NODE])
        sec, f, b, pad = struct.unpack_from('<4h', rec, 16)
        struct.pack_into('<4h', rec, 16, sec, f + 2 if f >= 0 else f,
                         b + 2 if b >= 0 else b, pad)
        shifted += rec

    def pack(normal, d, sec, front, back, bb):
        return struct.pack('<4f4h6f', *(tuple(normal) + (d, sec, front, back, 0) + tuple(bb)))

    n_hi, n_lo = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    n_hi[ax], n_lo[ax] = -1.0, 1.0
    old_root = 2                                   # old index 0, now shifted by 2
    new_leaf = n_old + 2
    node_a = pack(n_hi, -hi_plane, -1, 1, old_root, top)          # front: x < hi_plane
    node_b = pack(n_lo, lo_plane, -1, new_leaf, old_root, top)    # front: x > lo_plane
    leaf = pack((0.0, 0.0, 0.0), 0.0, new_idx, -1, -1, sec_bb)
    host_m['bsp'] = bytes(node_a) + bytes(node_b) + bytes(shifted) + leaf


def _ensure_extra(nav):
    if nav['nextra'] > 0:
        return 0
    nav['extra'] = list(nav.get('extra', [])) + [
        {'a': struct.pack('<I', GROUP_TAG), 'b': struct.pack('<I', 3),
         'c': struct.pack('<f', 1.0), 'rows': b''}]
    nav['nextra'] = 1
    return 0


def _append_corridor_nav(host_m, axis, la_lo, la_hi, ca_lo, ca_hi, floor_y, cell, end=None):
    """A straight chain of quad nav cells the length of the corridor, appended
    to the host's own nav. Links are symmetric within the chain only: no host
    link, LINK adds that in wave B. end=(ca_lo, ca_hi, floor_y) at la_hi grades a loft."""
    nav = host_m['nav']
    if nav['empty']:
        nav.update({'empty': False, 'f': 2, 'nverts': 0, 'nnodes': 0,
                    'nextra': 0, 'nparts': 0, 'verts': b'', 'nodes': [],
                    'extra': [], 'parts': []})
    extra_idx = _ensure_extra(nav)
    n = max(1, round((la_hi - la_lo) / cell))
    e_lo, e_hi, e_y = end if end is not None else (ca_lo, ca_hi, floor_y)
    pts = []
    for i in range(n + 1):
        f = i / float(n)
        t = la_lo + (la_hi - la_lo) * f
        lo, hi = ca_lo + (e_lo - ca_lo) * f, ca_hi + (e_hi - ca_hi) * f
        y = floor_y + (e_y - floor_y) * f
        if axis == 'x':
            pts.append((t, y, lo))
            pts.append((t, y, hi))
        else:
            # hi first keeps the ring clockwise in x-z, the winding link.py assumes
            pts.append((hi, y, t))
            pts.append((lo, y, t))
    base_v, base_n = nav['nverts'], nav['nnodes']
    nav['verts'] += b''.join(struct.pack('<3f', *p) for p in pts)
    nav['nverts'] = base_v + len(pts)
    nodes = []
    for i in range(n):
        v0a, v0b = base_v + 2 * i, base_v + 2 * i + 1
        v1a, v1b = base_v + 2 * (i + 1), base_v + 2 * (i + 1) + 1
        ring = (v0a, v0b, v1b, v1a)
        p0, p1, p2, p3 = pts[2 * i], pts[2 * i + 1], pts[2 * i + 3], pts[2 * i + 2]
        cx = sum(p[0] for p in (p0, p1, p2, p3)) / 4.0
        cy = sum(p[1] for p in (p0, p1, p2, p3)) / 4.0
        cz = sum(p[2] for p in (p0, p1, p2, p3)) / 4.0
        prev_ = base_n + i - 1 if i > 0 else -1
        next_ = base_n + i + 1 if i < n - 1 else -1
        hdr = struct.pack('<I3f2f6f', 0, cx, cy, cz, cy - 1.0, cy + 30.0,
                          0.0, 1.0, 0.0, 0.0, 1.0, 0.0)
        nodes.append({'hdr': hdr, 'v': struct.pack('<4I', *ring),
                      'nb': struct.pack('<4i', prev_, -1, next_, -1),
                      'd0': struct.pack('<4I', 0, 0, 0, 0),
                      'f15c': struct.pack('<I', extra_idx)})
    nav['nodes'] = nav['nodes'] + nodes
    nav['nnodes'] = base_n + n
    return base_n, n


def make_connector(host_m, a_index, a_face, b_index, b_face, width, height,
                   material_name, uv_scale=8.0, cell=6.0, name=None, split=None,
                   span=None, light=False, profile='loft', mask_cell=0.5, rim_n=caps.RIM_N):
    """host_m + a straight corridor between section a's and b's named faces -> host_m,
    with the new section appended last. `a_face`/`b_face` are (axis, sign), e.g. ('x', 1).
    span=(lo, hi) fixes the cross-axis range (an opening's), overriding width.
    light=True, or a dict of lighting.light_connector keywords, lights it too.
    profile: 'loft' follows the rooms' real mouths, 'box' is the plain box over
    span x height, 'capped' the box shrunk to the mouths and walled at both ends."""
    axis, a_sign = a_face
    axis_b, b_sign = b_face
    if axis != axis_b:
        raise ValueError('a straight corridor needs the same axis on both faces: %r vs %r'
                         % (a_face, b_face))
    if axis not in ('x', 'z'):
        raise NotImplementedError("only 'x'/'z' corridor axes are supported "
                                  "('y' is a shaft, not a corridor)")
    profile = PROFILE_ALIASES.get(profile, profile)
    if profile not in PROFILES:
        raise ValueError('profile must be one of %s, not %r' % (PROFILES, profile))
    ca = AX['z' if axis == 'x' else 'x']

    a_bb, b_bb = _bbox(host_m['sections'][a_index]), _bbox(host_m['sections'][b_index])
    a_plane = _face_plane(a_bb, axis, a_sign)
    b_plane = _face_plane(b_bb, axis, b_sign)
    la_lo, la_hi = (a_plane, b_plane) if a_plane < b_plane else (b_plane, a_plane)
    if la_hi - la_lo < 1.0:
        raise ValueError('sections %d and %d faces are only %.2f apart on %s'
                         % (a_index, b_index, la_hi - la_lo, axis))
    ca_lo, ca_hi = span if span is not None else _cross_span(a_bb, b_bb, ca, width)
    cross_center = (ca_lo + ca_hi) / 2.0

    fa = _probe_floor(host_m['sections'][a_index]['bvt'], a_plane, a_sign, cross_center, axis)
    fb = _probe_floor(host_m['sections'][b_index]['bvt'], b_plane, b_sign, cross_center, axis)
    candidates = [f for f in (fa, fb) if f is not None]
    if not candidates:
        candidates = [a_bb[0][1], b_bb[0][1]]      # no BVT to probe: bbox floors
    floor_y = min(candidates)
    rect_lo, rect_hi = [0.0, floor_y, 0.0], [0.0, floor_y + height, 0.0]
    rect_lo[ca], rect_hi[ca] = ca_lo, ca_hi

    masks, nav_end, lmuv = None, None, None
    name = name or 'connector%d_%d' % (a_index, b_index)
    if profile == 'loft':
        centre = [(rect_lo[k] + rect_hi[k]) / 2.0 for k in range(3)]
        rims = [caps.mouth_rim(host_m, i, p, centre, axis, rim_n, rect=(rect_lo, rect_hi), sign=sg)
                for i, p, sg in ((a_index, a_plane, a_sign), (b_index, b_plane, b_sign))]
        rim_lo, rim_hi = rims if a_plane < b_plane else rims[::-1]
        mesh, lmuv = _loft_mesh(axis, rim_lo, rim_hi, uv_scale)
        # the nav strip runs along the loft's bottom, as wide as the loft a step above it
        y_lo, y_hi = rim_lo.lowest[1], rim_hi.lowest[1]
        c_lo, c_hi = rim_lo.chord(y_lo + NAV_LIFT), rim_hi.chord(y_hi + NAV_LIFT)
        if c_lo is None or c_hi is None:
            raise ValueError('a mouth of sections %d and %d is under %.1f ft tall'
                             % (a_index, b_index, NAV_LIFT))
        ca_lo, ca_hi, floor_y = c_lo[0], c_lo[1], y_lo
        nav_end = (c_hi[0], c_hi[1], y_hi)
    else:
        if profile == 'capped':
            masks = (caps.mouth_mask(host_m, a_index, a_plane, axis, a_sign, rect_lo, rect_hi,
                                     mask_cell),
                     caps.mouth_mask(host_m, b_index, b_plane, axis, b_sign, rect_lo, rect_hi,
                                     mask_cell))
            rect = caps.open_union(masks)
            if rect is None:
                raise ValueError('sections %d and %d show no open cell on their %s faces'
                                 % (a_index, b_index, axis))
            # a cell wider than the hole, never narrower: a rim outside the corridor shows
            # the void from the room. The floor stays probed: a raised slab opens a slit.
            ca_lo, ca_hi = max(ca_lo, rect[0] - mask_cell), min(ca_hi, rect[2] + mask_cell)
            height = min(height, rect[3] + mask_cell - floor_y)
        mesh = _corridor_mesh(axis, la_lo, la_hi, ca_lo, ca_hi, floor_y, height, uv_scale)
    # an unlit loft stays per-vertex like the box; lit, it is born with (t, i/n) lightmap uvs
    sec_rec = _section_record(mesh, material_name, name, lmuv=lmuv if light else None)

    donor = {'sections': [sec_rec], 'materials': [_material_entry(material_name)],
             'mat16': b'\0\0', 'watervis': b'\x01', 'portals': [], 'lights': [],
             'datasize': 0}
    try:
        host_m, _staged = gtr.append_section(host_m, donor, 0, (0.0, 0.0, 0.0),
                                             lightmap_dir='connector', split=split)
    except ValueError as e:
        # only insert_bsp_leaf's own "no bookend gap" refusal falls back; any
        # other ValueError (a real bug elsewhere in append_section) propagates.
        if 'overlaps the host volume' not in str(e) and 'does not separate' not in str(e):
            raise
        new_idx = len(host_m['sections']) - 1     # append_section had already appended it
        _bsp_wedge_split(host_m, new_idx, axis, la_lo, la_hi)
    _dedupe_material(host_m, material_name)
    _append_corridor_nav(host_m, axis, la_lo, la_hi, ca_lo, ca_hi, floor_y, cell, end=nav_end)
    if masks is not None:
        new_idx = len(host_m['sections']) - 1
        caps.cap_mouth(host_m, new_idx, a_index, masks[0], uv_scale)
        caps.cap_mouth(host_m, new_idx, b_index, masks[1], uv_scale)
    if light:
        host_m['light_staged'] = lighting.light_connector(
            host_m, len(host_m['sections']) - 1, a_index, b_index, axis,
            **(dict(light) if isinstance(light, dict) else {}))
    # both branches above may have shifted the materials table or grown nav,
    # which changes the fixed part's length: relayout the blob offsets last.
    geom.relayout(host_m)
    return host_m

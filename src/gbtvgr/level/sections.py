"""One authored section -> the set's mesh records, surface table and BVT.

Render meshes use the two vertex families the corpus ships: the stride-72
declaration with four packed colours for a per-vertex bake (geometry type 2),
or the stride-64 one with a lightmap uv in slot 12 (geometry type 1).
"""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import heapq
import math
import struct

import numpy as np

from gbtvgr.mesh.bvt import bvt_build
from gbtvgr.mesh.smb import decl_stride
from gbtvgr.sets.geom import DECL72, make_decl

from .geometry_mesh import EditMesh
from .mathutil import F32, transform_normals, transform_points

DECL_LM64 = make_decl([(0, 0, 3), (1, 12, 3), (2, 24, 5), (3, 28, 5), (5, 32, 3),
                       (8, 44, 3), (11, 60, 9), (12, 56, 5)])
GEOM_VERTEX, GEOM_LIGHTMAP = 2, 1
MESH_FLAGS = 0x4
MAX_VERTS = 65535
DEFAULT_MATERIAL = 'graveyard\\pavementwet'


class MeshOut:
    __slots__ = ('name', 'material', 'positions', 'normals', 'uvs', 'lmuv', 'colors', 'indices')

    def __init__(self, name, material):
        self.name = name
        self.material = material
        self.positions, self.normals, self.uvs = [], [], []
        self.lmuv, self.colors, self.indices = [], [], []

    @property
    def nverts(self):
        return sum(len(p) for p in self.positions)

    def add(self, pos, nrm, uv, lm, col, idx):
        base = self.nverts
        self.positions.append(pos)
        self.normals.append(nrm)
        self.uvs.append(uv)
        self.lmuv.append(lm)
        self.colors.append(col)
        self.indices.append(idx + base)

    def arrays(self):
        cat = np.concatenate
        lm = None if any(l is None for l in self.lmuv) else cat(self.lmuv)
        return (cat(self.positions), cat(self.normals), cat(self.uvs), lm,
                cat(self.colors), cat(self.indices).astype(np.uint32))


def _chunk_faces(idx, max_verts):
    """Face index groups whose vertex sets fit the cap, in face order."""
    out, start, n = [], 0, len(idx)
    while start < n:
        end = n
        while True:
            used = np.unique(idx[start:end])
            if len(used) <= max_verts or end - start <= 1:
                break
            end = start + max(1, (end - start) // 2)
        out.append(np.arange(start, end))
        start = end
    return out


def _node_geometry(node):
    if node.kind == 'mesh':
        return node.mesh
    if node.kind == 'terrain' and node.terrain is not None:
        return node.terrain.mesh(node.props.get('uv_scale', 8.0))
    return None


def _node_colors(node, geo):
    if geo.colors is not None and len(geo.colors) == geo.nverts:
        return geo.colors
    if node.kind == 'terrain' and node.terrain is not None:
        c = getattr(node.terrain, 'colors', None)
        if c is not None and len(c) == geo.nverts:
            return c
    return None


def tessellate(pos, nrm, uv, col, idx, max_edge):
    """Bisect the longest edge until none exceeds max_edge, both faces on an edge at one
    midpoint so nothing cracks. Also returns each new face's source face index."""
    pos, nrm, uv = list(map(tuple, pos)), list(map(tuple, nrm)), list(map(tuple, uv))
    col = [tuple(int(c) for c in v) for v in col]
    faces = [list(map(int, f)) for f in idx]
    origin = list(range(len(faces)))
    edges = {}
    for fi, f in enumerate(faces):
        for k in range(3):
            edges.setdefault((min(f[k], f[k - 2]), max(f[k], f[k - 2])), set()).add(fi)

    def len2(a, b):
        return sum((pos[a][k] - pos[b][k]) ** 2 for k in range(3))

    limit2 = max_edge * max_edge
    heap = [(-len2(*e), e) for e in edges]
    heapq.heapify(heap)
    while heap:
        neg, (a, b) = heapq.heappop(heap)
        if -neg <= limit2:
            break
        owners = edges.pop((a, b), None)
        if owners is None:
            continue
        m = len(pos)
        pos.append(tuple((pos[a][k] + pos[b][k]) / 2.0 for k in range(3)))
        n = [nrm[a][k] + nrm[b][k] for k in range(3)]
        ln = math.sqrt(sum(c * c for c in n)) or 1.0
        nrm.append(tuple(c / ln for c in n))
        uv.append(((uv[a][0] + uv[b][0]) / 2.0, (uv[a][1] + uv[b][1]) / 2.0))
        col.append(tuple((col[a][k] + col[b][k] + 1) // 2 for k in range(4)))
        for fi in owners:
            f = faces[fi]
            k = next(k for k in range(3) if {f[k], f[(k + 1) % 3]} == {a, b})
            p, q, r = f[k], f[(k + 1) % 3], f[(k + 2) % 3]
            nf = len(faces)
            faces[fi] = [p, m, r]
            faces.append([m, q, r])
            origin.append(origin[fi])
            qr = (min(q, r), max(q, r))
            edges[qr].discard(fi)
            edges[qr].add(nf)
            for e, owner in (((p, m), fi), ((m, q), nf), ((m, r), fi), ((m, r), nf)):
                key = (min(e), max(e))
                if key not in edges:
                    heapq.heappush(heap, (-len2(*key), key))
                edges.setdefault(key, set()).add(owner)
    return (np.array(pos, F32), np.array(nrm, F32), np.array(uv, F32),
            np.array(col, np.uint8).reshape(-1, 4), np.array(faces, np.uint32).reshape(-1, 3),
            np.array(origin, np.int64))


def tessellate_mesh(geo, max_edge):
    """An EditMesh cut to max_edge, for grids that must exist before the vertex bake."""
    pos, nrm, uv, idx, _cmap, _first = geo.split_corners()
    col = np.zeros((len(pos), 4), np.uint8)
    pos, _nrm, uv, _col, idx, origin = tessellate(pos, nrm, uv, col, idx, max_edge)
    return EditMesh(pos, idx.astype(np.int32), uv[idx], geo.face_mat[origin], geo.smooth)


def gather_meshes(section, lmuv_by_node=None, default_material=DEFAULT_MATERIAL, max_edge=None):
    """Every visible mesh under the section, bucketed by material and kept under
    the vertex cap. Returns [MeshOut]."""
    buckets = {}
    order = []
    for node in section.iter_descendants():
        geo = _node_geometry(node)
        if geo is None or geo.nfaces == 0 or not node.visible:
            continue
        mats = list(node.props.get('materials') or [])
        wm = node.world_matrix()
        lm_all = (lmuv_by_node or {}).get(node.id)
        colors = _node_colors(node, geo)
        for mat_i in np.unique(geo.face_mat):
            faces = np.where(geo.face_mat == mat_i)[0]
            ref = mats[mat_i] if mat_i < len(mats) and mats[mat_i] else default_material
            sub = geo.subset(faces)
            pos, nrm, uv, idx, cmap, first = sub.split_corners()
            pos = transform_points(wm, pos)
            nrm = transform_normals(wm, nrm)
            if colors is not None:
                col = np.asarray(colors, np.uint8)[np.unique(geo.faces[faces].ravel(), return_inverse=False)]
                # subset() compacted the vertices in ascending original order
                col = col[cmap]
            else:
                col = np.full((len(pos), 4), 255, np.uint8)
            lm = None
            if lm_all is not None:
                lm = np.asarray(lm_all, F32)[faces].reshape(-1, 2)[first]
            elif max_edge:
                pos, nrm, uv, col, idx = tessellate(pos, nrm, uv, col, idx, max_edge)[:5]
            for chunk in _chunk_faces(idx, MAX_VERTS):
                cidx = idx[chunk]
                used = np.unique(cidx)
                remap = np.full(len(pos), -1, np.int64)
                remap[used] = np.arange(len(used))
                piece_idx = remap[cidx].astype(np.uint32)
                key = ref.lower()
                if key not in buckets or buckets[key][-1].nverts + len(used) > MAX_VERTS:
                    name = '%s_%s_%d' % (section.name[:10], ref.replace('/', '\\').split('\\')[-1][:12],
                                         len(buckets.get(key, [])))
                    buckets.setdefault(key, []).append(MeshOut(name[:31], ref))
                    if key not in order:
                        order.append(key)
                buckets[key][-1].add(pos[used], nrm[used], uv[used],
                                     None if lm is None else lm[used], col[used], piece_idx)
    return [mo for key in order for mo in buckets[key]]


def tangents(pos, nrm, uv, idx):
    """Per-vertex tangent and binormal from the uv gradient, like the codec's."""
    n = len(pos)
    t = np.zeros((n, 3), np.float64)
    b = np.zeros((n, 3), np.float64)
    if len(idx):
        p0, p1, p2 = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
        u0, u1, u2 = uv[idx[:, 0]], uv[idx[:, 1]], uv[idx[:, 2]]
        e1, e2 = (p1 - p0).astype(np.float64), (p2 - p0).astype(np.float64)
        d1, d2 = (u1 - u0).astype(np.float64), (u2 - u0).astype(np.float64)
        det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(np.abs(det) > 1e-12, 1.0 / det, 0.0)
        ft = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
        fb = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r[:, None]
        for k in range(3):
            np.add.at(t, idx[:, k], ft)
            np.add.at(b, idx[:, k], fb)
    nn = nrm.astype(np.float64)
    t -= nn * np.einsum('ij,ij->i', nn, t)[:, None]
    lt = np.linalg.norm(t, axis=1)
    bad = lt < 1e-9
    t[bad] = np.where(np.abs(nn[bad][:, 1:2]) < 0.9, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    lb = np.linalg.norm(b, axis=1)
    bad = lb < 1e-9
    b[bad] = np.cross(nn[bad], t[bad])
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return t.astype(F32), b.astype(F32)


def _packed_colors(col):
    """(n, 4) RGBA -> (n, 4) bytes laid down B, G, R, A."""
    return np.ascontiguousarray(col[:, [2, 1, 0, 3]], dtype=np.uint8)


def pack_vertices(pos, nrm, uv, lm, col, idx, lightmapped):
    decl = DECL_LM64 if lightmapped else DECL72
    st = decl_stride(decl)
    n = len(pos)
    buf = np.zeros((n, st), np.uint8)
    t, b = tangents(pos, nrm, uv, idx)
    buf[:, 0:12] = pos.astype('<f4').view(np.uint8).reshape(n, 12)
    buf[:, 12:24] = nrm.astype('<f4').view(np.uint8).reshape(n, 12)
    h = uv.astype('<f2').view(np.uint8).reshape(n, 4)
    buf[:, 24:28] = h
    buf[:, 28:32] = h
    buf[:, 32:44] = t.astype('<f4').view(np.uint8).reshape(n, 12)
    buf[:, 44:56] = b.astype('<f4').view(np.uint8).reshape(n, 12)
    c = _packed_colors(col)
    if lightmapped:
        lmv = (lm if lm is not None else uv).astype('<f2').view(np.uint8).reshape(n, 4)
        buf[:, 56:60] = lmv
        buf[:, 60:64] = c
    else:
        for off in (56, 60, 64, 68):
            buf[:, off:off + 4] = c
    return decl, buf.tobytes()


def mesh_record(mo, mat_index, lightmapped=False):
    pos, nrm, uv, lm, col, idx = mo.arrays()
    if len(pos) > MAX_VERTS:
        raise ValueError('%s: %d vertices exceed the 16-bit index cap' % (mo.name, len(pos)))
    decl, vdata = pack_vertices(pos, nrm, uv, lm, col, idx, lightmapped)
    idata = np.ascontiguousarray(idx, dtype='<u2').tobytes()
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    pkt = {'datasize': len(vdata) + len(idata), 'decl': decl, 'nverts': len(pos),
           'nprims': len(idx), 'f18': 0, 'morphs': [],
           'bbox': struct.pack('<6f', *lo, *hi), 'vdata': vdata, 'idata': idata, 'mdata': []}
    name = mo.name.encode('latin1')[:0x1F]
    return {'h70': struct.pack('<H', mat_index), 'name': name + b'\0' * (0x20 - len(name)),
            'flags': MESH_FLAGS, 'f6c': struct.pack('<I', GEOM_LIGHTMAP if lightmapped else GEOM_VERTEX),
            'pkt': pkt}, (lo, hi)


def collect_collision(doc, section, pad=1.0):
    """World-space collision triangles that touch the section box, both windings,
    with a surface-table index per triangle from each node's surface mask."""
    verts, tris, owners = doc.world_triangles(collide_only=True)
    entries = [('%s_solid' % section.name[:24], 0)]
    if len(tris) == 0:
        return np.zeros((0, 3), F32), np.zeros((0, 3), np.int64), [], [], entries
    lo, hi = section.lo - pad, section.hi + pad
    p = verts[tris]
    tlo, thi = p.min(axis=1), p.max(axis=1)
    keep = np.all(thi >= lo, axis=1) & np.all(tlo <= hi, axis=1)
    masks = {}
    for n in doc.mesh_nodes():
        masks[n.id] = int(n.props.get('surface_mask', 0) or 0)
    owner_arr = np.array([masks.get(o, 0) for o in owners], np.int64)
    kept = np.where(keep)[0]
    surf = []
    index_of = {0: 0}
    for m in owner_arr[kept]:
        m = int(m)
        if m not in index_of:
            index_of[m] = len(entries)
            entries.append(('%s_mask%d' % (section.name[:20], m), m))
        surf.append(index_of[m])
    t = tris[kept]
    flipped = t[:, [0, 2, 1]]
    both = np.concatenate([t, flipped])
    surf = surf + surf
    flags = [0] * len(both)
    return verts, both, surf, flags, entries


def build_section_bvt(verts, tris, surf, flags, inflate=1.0):
    """(root node, arena size) over the collision soup; returns (None, 0) when empty."""
    if len(tris) == 0:
        return None, 0
    used = np.unique(np.asarray(tris).ravel())
    remap = np.full(len(verts), -1, np.int64)
    remap[used] = np.arange(len(used))
    v = [tuple(float(c) for c in verts[i]) for i in used]
    t = [tuple(int(x) for x in remap[tri]) for tri in tris]
    return bvt_build(v, t, list(surf), list(flags), inflate=inflate)


def surface_entries(entries, bbox):
    out = []
    for name, mask in entries:
        nb = name.encode('latin1')[:0x1F]
        out.append({'name': nb + b'\0' * (0x20 - len(nb)),
                    'v': struct.pack('<2I6f', 0, mask, *bbox[0], *bbox[1])})
    return out

"""EditMesh: triangles with per-corner uvs and a material index per face.

Kept deliberately small. Vertices are shared, uvs live on face corners the way
Blender's loops do, and export splits a vertex only where its corners disagree.
Winding is right-handed CCW with the normal facing out, the OBJ convention the
game's own meshes follow.
"""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import numpy as np

from .mathutil import F32, face_normals, vertex_normals, ray_triangles
from .scene import deserialize_array, serialize_array

MAX_VERTS = 65535        # 16-bit index buffer, and the exporter refuses beyond it


class EditMesh:
    def __init__(self, verts=None, faces=None, uvs=None, face_mat=None, smooth=False):
        self.verts = np.zeros((0, 3), F32) if verts is None else np.asarray(verts, F32).reshape(-1, 3)
        self.faces = np.zeros((0, 3), np.int32) if faces is None else np.asarray(faces, np.int32).reshape(-1, 3)
        n = len(self.faces)
        self.uvs = np.zeros((n, 3, 2), F32) if uvs is None else np.asarray(uvs, F32).reshape(n, 3, 2)
        self.face_mat = np.zeros(n, np.int32) if face_mat is None else np.asarray(face_mat, np.int32).reshape(n)
        self.smooth = smooth
        self.colors = None            # (N, 4) uint8 when baked

    # -- basics ---------------------------------------------------------------------
    def copy(self):
        m = EditMesh(self.verts.copy(), self.faces.copy(), self.uvs.copy(),
                     self.face_mat.copy(), self.smooth)
        m.colors = None if self.colors is None else self.colors.copy()
        return m

    @property
    def nverts(self):
        return len(self.verts)

    @property
    def nfaces(self):
        return len(self.faces)

    def bbox(self):
        if not len(self.verts):
            return np.zeros(3, F32), np.zeros(3, F32)
        return self.verts.min(axis=0), self.verts.max(axis=0)

    def face_normals(self):
        return face_normals(self.verts, self.faces) if len(self.faces) else np.zeros((0, 3), F32)

    def vertex_normals(self):
        return vertex_normals(self.verts, self.faces) if len(self.faces) else np.zeros((0, 3), F32)

    def face_centers(self):
        return self.verts[self.faces].mean(axis=1) if len(self.faces) else np.zeros((0, 3), F32)

    def to_dict(self):
        d = {'verts': serialize_array(self.verts, '<f4'), 'faces': serialize_array(self.faces, '<i4'),
             'uvs': serialize_array(self.uvs, '<f4'), 'face_mat': serialize_array(self.face_mat, '<i4'),
             'smooth': self.smooth}
        if self.colors is not None:
            d['colors'] = serialize_array(self.colors, 'u1')
        return d

    @classmethod
    def from_dict(cls, d):
        m = cls(deserialize_array(d['verts']), deserialize_array(d['faces']),
                deserialize_array(d['uvs']), deserialize_array(d['face_mat']), d.get('smooth', False))
        if 'colors' in d:
            m.colors = deserialize_array(d['colors'])
        return m

    # -- building ---------------------------------------------------------------------
    def append(self, other, mat_offset=0):
        base = len(self.verts)
        self.verts = np.concatenate([self.verts, other.verts]) if len(self.verts) else other.verts.copy()
        self.faces = np.concatenate([self.faces, other.faces + base]) if len(self.faces) else other.faces.copy()
        self.uvs = np.concatenate([self.uvs, other.uvs]) if len(self.uvs) else other.uvs.copy()
        fm = other.face_mat + mat_offset
        self.face_mat = np.concatenate([self.face_mat, fm]) if len(self.face_mat) else fm.copy()
        self.colors = None
        return self

    @classmethod
    def from_quads(cls, quads, mat=0, uv_scale=1.0):
        """quads: list of 4 points each, CCW seen from outside. UVs are planar by
        the dominant normal axis, world units over uv_scale."""
        verts, faces, uvs = [], [], []
        for q in quads:
            base = len(verts)
            verts.extend(q)
            faces.append((base, base + 1, base + 2))
            faces.append((base, base + 2, base + 3))
        m = cls(np.array(verts, F32).reshape(-1, 3), np.array(faces, np.int32),
                None, np.full(len(faces), mat, np.int32))
        m.planar_uvs(uv_scale)
        return m

    def planar_uvs(self, scale=1.0, faces=None):
        """Box-project uvs by each face's dominant normal axis. Walls get height in v."""
        if not len(self.faces):
            return
        idx = np.arange(len(self.faces)) if faces is None else np.asarray(faces)
        n = self.face_normals()[idx]
        ax = np.argmax(np.abs(n), axis=1)
        p = self.verts[self.faces[idx]]            # (k, 3, 3)
        u_axis = np.where(ax == 0, 2, 0)
        v_axis = np.where(ax == 1, 2, 1)
        u = np.take_along_axis(p, u_axis[:, None, None].repeat(3, 1), axis=2)[:, :, 0]
        v = np.take_along_axis(p, v_axis[:, None, None].repeat(3, 1), axis=2)[:, :, 0]
        self.uvs[idx, :, 0] = u / scale
        self.uvs[idx, :, 1] = v / scale

    # -- queries ----------------------------------------------------------------------
    def pick(self, origin, direction, cull=False):
        """(distance, face index) for the nearest hit in mesh space."""
        if not len(self.faces):
            return None, -1
        v = self.verts[self.faces]
        return ray_triangles(origin, direction, v[:, 0], v[:, 1], v[:, 2], cull)

    def edges(self):
        f = self.faces
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        e.sort(axis=1)
        return np.unique(e, axis=0)

    def face_edges_selected(self, faces):
        """Boundary edges of a face set (edges used once within it)."""
        f = self.faces[np.asarray(faces)]
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        key = np.sort(e, axis=1)
        _, idx, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
        return e[idx[counts == 1]]

    # -- edits (each returns the affected face indices or None) --------------------------
    def translate_verts(self, vidx, delta):
        self.verts[np.asarray(vidx)] += np.asarray(delta, F32)

    def transform_verts(self, vidx, m):
        from .mathutil import transform_points
        vidx = np.asarray(vidx)
        self.verts[vidx] = transform_points(m, self.verts[vidx])

    def delete_faces(self, faces):
        keep = np.ones(len(self.faces), bool)
        keep[np.asarray(faces)] = False
        self.faces = self.faces[keep]
        self.uvs = self.uvs[keep]
        self.face_mat = self.face_mat[keep]
        self.compact()

    def compact(self):
        """Drop vertices no face uses."""
        used = np.zeros(len(self.verts), bool)
        if len(self.faces):
            used[self.faces.ravel()] = True
        remap = np.cumsum(used) - 1
        self.verts = self.verts[used]
        if self.colors is not None:
            self.colors = self.colors[used]
        if len(self.faces):
            self.faces = remap[self.faces].astype(np.int32)

    def merge_by_distance(self, dist=1e-4):
        """Weld coincident vertices; degenerate faces go."""
        if not len(self.verts):
            return
        q = np.round(self.verts / max(dist, 1e-9)).astype(np.int64)
        _, first, inverse = np.unique(q, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
        self.verts = self.verts[first]
        if self.colors is not None:
            self.colors = self.colors[first]
        self.faces = inverse[self.faces].astype(np.int32)
        ok = (self.faces[:, 0] != self.faces[:, 1]) & (self.faces[:, 1] != self.faces[:, 2]) \
            & (self.faces[:, 0] != self.faces[:, 2])
        self.faces, self.uvs, self.face_mat = self.faces[ok], self.uvs[ok], self.face_mat[ok]

    def flip_faces(self, faces=None):
        idx = slice(None) if faces is None else np.asarray(faces)
        self.faces[idx] = self.faces[idx][:, [0, 2, 1]]
        self.uvs[idx] = self.uvs[idx][:, [0, 2, 1]]

    def extrude_faces(self, faces, distance):
        """Push a face region out along its average normal, walling its boundary.
        Returns the indices of the moved faces."""
        faces = np.asarray(faces, np.int64)
        if not len(faces):
            return faces
        n = self.face_normals()[faces].mean(axis=0)
        ln = np.linalg.norm(n)
        n = n / ln if ln > 1e-9 else np.array([0, 1, 0], F32)
        boundary = self.face_edges_selected(faces)
        region_verts = np.unique(self.faces[faces].ravel())
        base = len(self.verts)
        remap = {int(v): base + i for i, v in enumerate(region_verts)}
        new_verts = self.verts[region_verts] + (n * distance).astype(F32)
        self.verts = np.concatenate([self.verts, new_verts])
        if self.colors is not None:
            self.colors = np.concatenate([self.colors, self.colors[region_verts]])
        moved = self.faces[faces].copy()
        for k in range(3):
            moved[:, k] = [remap[int(v)] for v in moved[:, k]]
        self.faces[faces] = moved
        walls, wall_uv, wall_mat = [], [], []
        mat = int(self.face_mat[faces[0]])
        for a, b in boundary:
            a2, b2 = remap[int(a)], remap[int(b)]
            walls.append((int(a), int(b), b2))
            walls.append((int(a), b2, a2))
            wall_uv.append([[0, 0], [1, 0], [1, 1]])
            wall_uv.append([[0, 0], [1, 1], [0, 1]])
            wall_mat.extend([mat, mat])
        if walls:
            self.faces = np.concatenate([self.faces, np.array(walls, np.int32)])
            self.uvs = np.concatenate([self.uvs, np.array(wall_uv, F32)])
            self.face_mat = np.concatenate([self.face_mat, np.array(wall_mat, np.int32)])
            self.planar_uvs(1.0, np.arange(len(self.faces) - len(walls), len(self.faces)))
        return faces

    def subdivide_faces(self, faces=None):
        """Each face into four by edge midpoints (shared midpoints welded)."""
        idx = np.arange(len(self.faces)) if faces is None else np.asarray(faces, np.int64)
        if not len(idx):
            return
        mid = {}
        verts = [self.verts]
        extra = []

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in mid:
                mid[key] = len(self.verts) + len(extra)
                extra.append((self.verts[a] + self.verts[b]) / 2.0)
            return mid[key]

        new_faces, new_uvs, new_mat = [], [], []
        keep = np.ones(len(self.faces), bool)
        keep[idx] = False
        for fi in idx:
            a, b, c = (int(v) for v in self.faces[fi])
            ua, ub, uc = self.uvs[fi]
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            uab, ubc, uca = (ua + ub) / 2, (ub + uc) / 2, (uc + ua) / 2
            for f, u in (((a, ab, ca), (ua, uab, uca)), ((ab, b, bc), (uab, ub, ubc)),
                         ((ca, bc, c), (uca, ubc, uc)), ((ab, bc, ca), (uab, ubc, uca))):
                new_faces.append(f)
                new_uvs.append(u)
                new_mat.append(self.face_mat[fi])
        if extra:
            verts.append(np.array(extra, F32))
        self.verts = np.concatenate(verts)
        if self.colors is not None:
            self.colors = None
        self.faces = np.concatenate([self.faces[keep], np.array(new_faces, np.int32)])
        self.uvs = np.concatenate([self.uvs[keep], np.array(new_uvs, F32)])
        self.face_mat = np.concatenate([self.face_mat[keep], np.array(new_mat, np.int32)])

    def set_material(self, faces, mat):
        self.face_mat[np.asarray(faces)] = mat

    # -- export -----------------------------------------------------------------------
    def split_corners(self):
        """Per-vertex arrays, splitting a vertex only where its corners disagree.
        Returns (positions, normals, uvs, indices, corner->vertex map, corner index)."""
        n = len(self.faces)
        if n == 0:
            z = np.zeros((0, 3), F32)
            return (z, z, np.zeros((0, 2), F32), np.zeros((0, 3), np.uint32),
                    np.zeros(0, np.int64), np.zeros(0, np.int64))
        corner_v = self.faces.reshape(-1)
        corner_uv = self.uvs.reshape(-1, 2)
        if self.smooth:
            vn = self.vertex_normals()
            corner_n = vn[corner_v]
        else:
            fn = self.face_normals()
            corner_n = np.repeat(fn, 3, axis=0)
        key = np.concatenate([corner_v[:, None].astype(np.float64),
                              np.round(corner_uv, 5).astype(np.float64),
                              np.round(corner_n, 3).astype(np.float64)], axis=1)
        _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
        pos = self.verts[corner_v[first]]
        nrm = corner_n[first].astype(F32)
        uv = corner_uv[first].astype(F32)
        idx = inverse.reshape(n, 3).astype(np.uint32)
        return pos, nrm, uv, idx, corner_v[first], first

    def export_vertex_count(self):
        return len(self.split_corners()[0])

    def chunks(self, max_verts=MAX_VERTS):
        """Split into meshes whose exported vertex count stays under the cap,
        by walking faces in spatial order."""
        if self.export_vertex_count() <= max_verts:
            return [self]
        centers = self.face_centers()
        lo, hi = self.bbox()
        ax = int(np.argmax(hi - lo))
        order = np.argsort(centers[:, ax], kind='stable')
        out, start = [], 0
        step = max(1, len(order) // 8)
        while start < len(order):
            end = min(len(order), start + step)
            piece = self.subset(order[start:end])
            while piece.export_vertex_count() > max_verts and end - start > 1:
                end = start + max(1, (end - start) // 2)
                piece = self.subset(order[start:end])
            out.append(piece)
            start = end
        return out

    def subset(self, faces):
        faces = np.asarray(faces)
        m = EditMesh(self.verts, self.faces[faces], self.uvs[faces], self.face_mat[faces], self.smooth)
        m.colors = self.colors
        m.compact()
        return m

    def by_material(self):
        """{material index: EditMesh} for meshes that carry several."""
        out = {}
        for mat in np.unique(self.face_mat):
            out[int(mat)] = self.subset(np.where(self.face_mat == mat)[0])
        return out

    def triangle_soup(self):
        return self.verts[self.faces]

    def double_sided(self):
        """A copy with every face present in both windings."""
        m = self.copy()
        flipped = self.copy()
        flipped.flip_faces()
        m.append(flipped)
        return m

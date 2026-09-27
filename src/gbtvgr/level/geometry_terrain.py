"""Heightmap terrain: a grid of heights and per-vertex material weights, with
sculpt brushes, and an export that chunks under the 16-bit index cap."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import numpy as np

from .mathutil import F32
from .scene import deserialize_array, serialize_array
from .geometry_mesh import EditMesh, MAX_VERTS

MAX_LAYERS = 4


class Terrain:
    def __init__(self, nx=33, nz=33, cell=4.0, origin=(0.0, 0.0, 0.0)):
        self.nx, self.nz = int(nx), int(nz)
        self.cell = float(cell)
        self.origin = np.array(origin, F32)
        self.heights = np.zeros((self.nx, self.nz), F32)
        self.weights = np.zeros((self.nx, self.nz, MAX_LAYERS), F32)
        self.weights[:, :, 0] = 1.0
        self._mesh = None

    def copy(self):
        t = Terrain(self.nx, self.nz, self.cell, self.origin)
        t.heights = self.heights.copy()
        t.weights = self.weights.copy()
        return t

    def to_dict(self):
        return {'nx': self.nx, 'nz': self.nz, 'cell': self.cell,
                'origin': [float(v) for v in self.origin],
                'heights': serialize_array(self.heights, '<f4'),
                'weights': serialize_array(self.weights, '<f4')}

    @classmethod
    def from_dict(cls, d):
        t = cls(d['nx'], d['nz'], d['cell'], d['origin'])
        t.heights = deserialize_array(d['heights']).reshape(t.nx, t.nz).astype(F32)
        t.weights = deserialize_array(d['weights']).reshape(t.nx, t.nz, MAX_LAYERS).astype(F32)
        return t

    # -- geometry ---------------------------------------------------------------------
    def grid_positions(self):
        xs = self.origin[0] + np.arange(self.nx, dtype=F32) * self.cell
        zs = self.origin[2] + np.arange(self.nz, dtype=F32) * self.cell
        gx, gz = np.meshgrid(xs, zs, indexing='ij')
        return np.stack([gx, self.origin[1] + self.heights, gz], axis=-1)

    def size(self):
        return (self.nx - 1) * self.cell, (self.nz - 1) * self.cell

    def height_at(self, x, z):
        """Bilinear height in local space; None outside the patch."""
        u = (x - self.origin[0]) / self.cell
        v = (z - self.origin[2]) / self.cell
        if u < 0 or v < 0 or u > self.nx - 1 or v > self.nz - 1:
            return None
        i, j = min(int(u), self.nx - 2), min(int(v), self.nz - 2)
        fu, fv = u - i, v - j
        h = self.heights
        top = h[i, j] * (1 - fu) + h[i + 1, j] * fu
        bot = h[i, j + 1] * (1 - fu) + h[i + 1, j + 1] * fu
        return float(self.origin[1] + top * (1 - fv) + bot * fv)

    def invalidate(self):
        self._mesh = None

    def mesh(self, uv_scale=8.0):
        """The whole patch as one EditMesh, faces tagged with the dominant layer."""
        if self._mesh is not None:
            return self._mesh
        pos = self.grid_positions().reshape(-1, 3)
        nx, nz = self.nx, self.nz
        i, j = np.meshgrid(np.arange(nx - 1), np.arange(nz - 1), indexing='ij')
        a = (i * nz + j).ravel()
        b = a + 1
        c = a + nz
        d = c + 1
        faces = np.stack([np.stack([a, b, d], 1), np.stack([a, d, c], 1)], 1).reshape(-1, 3)
        # a cell takes the layer with the most weight over its four corners
        w = self.weights
        cell_w = w[:-1, :-1] + w[1:, :-1] + w[:-1, 1:] + w[1:, 1:]
        mat = np.argmax(cell_w.reshape(-1, MAX_LAYERS), axis=1)
        face_mat = np.repeat(mat, 2).astype(np.int32)
        m = EditMesh(pos, faces.astype(np.int32), None, face_mat, smooth=True)
        m.planar_uvs(uv_scale)
        self._mesh = m
        return m

    def export_chunks(self, uv_scale=8.0):
        """Per-material meshes, each chunked under the vertex cap."""
        out = []
        for mat, m in self.mesh(uv_scale).by_material().items():
            for piece in m.chunks(MAX_VERTS):
                out.append((mat, piece))
        return out

    # -- brushes ----------------------------------------------------------------------
    def _falloff(self, x, z, radius, hardness=0.5):
        gx = self.origin[0] + np.arange(self.nx, dtype=F32) * self.cell
        gz = self.origin[2] + np.arange(self.nz, dtype=F32) * self.cell
        dx = (gx[:, None] - x) / max(radius, 1e-6)
        dz = (gz[None, :] - z) / max(radius, 1e-6)
        d = np.sqrt(dx * dx + dz * dz)
        t = np.clip((d - hardness) / max(1e-6, 1.0 - hardness), 0.0, 1.0)
        return (1.0 - t * t * (3.0 - 2.0 * t)).astype(F32) * (d <= 1.0)

    def raise_(self, x, z, radius, amount, hardness=0.5):
        self.heights += self._falloff(x, z, radius, hardness) * amount
        self.invalidate()

    def lower(self, x, z, radius, amount, hardness=0.5):
        self.raise_(x, z, radius, -amount, hardness)

    def flatten(self, x, z, radius, target=None, strength=1.0, hardness=0.5):
        f = self._falloff(x, z, radius, hardness) * strength
        if target is None:
            h = self.height_at(x, z)
            target = (h - self.origin[1]) if h is not None else 0.0
        self.heights += (target - self.heights) * f
        self.invalidate()

    def smooth(self, x, z, radius, strength=0.5, hardness=0.5):
        f = self._falloff(x, z, radius, hardness) * strength
        h = np.pad(self.heights, 1, mode='edge')
        blur = (h[:-2, 1:-1] + h[2:, 1:-1] + h[1:-1, :-2] + h[1:-1, 2:] + 4 * h[1:-1, 1:-1]) / 8.0
        self.heights += (blur - self.heights) * f
        self.invalidate()

    def noise(self, x, z, radius, amount, hardness=0.5, seed=None):
        f = self._falloff(x, z, radius, hardness)
        rng = np.random.default_rng(seed)
        n = rng.standard_normal(self.heights.shape).astype(F32)
        n = (np.pad(n, 1, mode='edge')[:-2, 1:-1] + np.pad(n, 1, mode='edge')[2:, 1:-1]
             + np.pad(n, 1, mode='edge')[1:-1, :-2] + np.pad(n, 1, mode='edge')[1:-1, 2:] + n) / 5.0
        self.heights += n * f * amount
        self.invalidate()

    def paint(self, x, z, radius, layer, strength=0.5, hardness=0.5):
        f = self._falloff(x, z, radius, hardness) * strength
        w = self.weights
        w[:, :, layer] += f
        s = w.sum(axis=2, keepdims=True)
        s[s < 1e-6] = 1.0
        self.weights = (w / s).astype(F32)
        self.invalidate()

    def vertex_tints(self, layer_colors):
        """Blend weights -> RGBA tint per grid vertex, for the colour channel."""
        cols = np.array(layer_colors, F32).reshape(-1, 3)[:MAX_LAYERS]
        w = self.weights[:, :, :len(cols)]
        rgb = (w @ cols).reshape(-1, 3)
        out = np.empty((rgb.shape[0], 4), np.uint8)
        out[:, :3] = np.clip(rgb, 0, 255)
        out[:, 3] = 255
        return out

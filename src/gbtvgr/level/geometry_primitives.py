"""Parametric primitives as EditMeshes, normals out, CCW, uvs in world units."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import math

import numpy as np

from .mathutil import F32
from .geometry_mesh import EditMesh


def _mesh(verts, faces, uv_scale, mat=0, smooth=False):
    m = EditMesh(np.array(verts, F32).reshape(-1, 3), np.array(faces, np.int32).reshape(-1, 3),
                 None, np.full(len(faces), mat, np.int32), smooth)
    m.planar_uvs(uv_scale)
    return m


def plane(width=10.0, depth=10.0, segments=1, uv_scale=8.0, mat=0):
    """A floor in XZ, normal +Y, centred on the origin."""
    n = max(1, int(segments))
    xs = np.linspace(-width / 2.0, width / 2.0, n + 1)
    zs = np.linspace(-depth / 2.0, depth / 2.0, n + 1)
    gx, gz = np.meshgrid(xs, zs, indexing='ij')
    verts = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1)
    faces = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            b = a + 1
            c = a + (n + 1)
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    return _mesh(verts, faces, uv_scale, mat)


def box(width=10.0, height=10.0, depth=10.0, uv_scale=8.0, mat=0, inside=False):
    """An axis-aligned box on the origin, its base at y = 0. inside=True flips it
    into a room whose faces look inward."""
    x, z = width / 2.0, depth / 2.0
    y0, y1 = 0.0, height
    p = [(-x, y0, -z), (x, y0, -z), (x, y0, z), (-x, y0, z),
         (-x, y1, -z), (x, y1, -z), (x, y1, z), (-x, y1, z)]
    quads = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    faces = []
    for a, b, c, d in quads:
        faces.append((a, b, c))
        faces.append((a, c, d))
    m = _mesh(p, faces, uv_scale, mat)
    if inside:
        m.flip_faces()
    return m


def cylinder(radius=5.0, height=10.0, segments=16, uv_scale=8.0, mat=0, cap=True):
    n = max(3, int(segments))
    ring0, ring1 = [], []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        ring0.append((radius * math.cos(a), 0.0, radius * math.sin(a)))
        ring1.append((radius * math.cos(a), height, radius * math.sin(a)))
    verts = ring0 + ring1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, n + j, j))
        faces.append((i, n + i, n + j))
    if cap:
        verts.append((0.0, 0.0, 0.0))
        verts.append((0.0, height, 0.0))
        cb, ct = 2 * n, 2 * n + 1
        for i in range(n):
            j = (i + 1) % n
            faces.append((cb, i, j))
            faces.append((ct, n + j, n + i))
    m = _mesh(verts, faces, uv_scale, mat, smooth=True)
    return m


def cone(radius=5.0, height=10.0, segments=16, uv_scale=8.0, mat=0):
    n = max(3, int(segments))
    verts = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        verts.append((radius * math.cos(a), 0.0, radius * math.sin(a)))
    verts.append((0.0, height, 0.0))
    verts.append((0.0, 0.0, 0.0))
    apex, base = n, n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, apex, j))
        faces.append((base, i, j))
    return _mesh(verts, faces, uv_scale, mat, smooth=True)


def sphere(radius=5.0, segments=16, rings=8, uv_scale=8.0, mat=0):
    n, r = max(3, int(segments)), max(2, int(rings))
    verts = [(0.0, radius, 0.0)]
    for k in range(1, r):
        phi = math.pi * k / r
        y = radius * math.cos(phi)
        s = radius * math.sin(phi)
        for i in range(n):
            a = 2.0 * math.pi * i / n
            verts.append((s * math.cos(a), y, s * math.sin(a)))
    verts.append((0.0, -radius, 0.0))
    bottom = len(verts) - 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((0, 1 + j, 1 + i))
    for k in range(r - 2):
        a0, b0 = 1 + k * n, 1 + (k + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces.append((a0 + i, a0 + j, b0 + j))
            faces.append((a0 + i, b0 + j, b0 + i))
    last = 1 + (r - 2) * n
    for i in range(n):
        j = (i + 1) % n
        faces.append((last + i, last + j, bottom))
    m = _mesh(verts, faces, uv_scale, mat, smooth=True)
    return m


def wall(length=10.0, height=8.0, thickness=0.0, uv_scale=8.0, mat=0):
    """A wall along +X from the origin. Zero thickness is one two-sided quad."""
    if thickness <= 0.0:
        quad = [(0.0, 0.0, 0.0), (length, 0.0, 0.0), (length, height, 0.0), (0.0, height, 0.0)]
        m = EditMesh.from_quads([quad], mat, uv_scale)
        return m.double_sided()
    b = box(length, height, thickness, uv_scale, mat)
    b.verts[:, 0] += length / 2.0
    return b


def ramp(length=12.0, width=6.0, rise=4.0, uv_scale=8.0, mat=0):
    """A sloped deck along +Z, rising `rise` over `length`."""
    hw = width / 2.0
    quad = [(-hw, 0.0, 0.0), (-hw, rise, length), (hw, rise, length), (hw, 0.0, 0.0)]
    return EditMesh.from_quads([quad], mat, uv_scale)


PRIMITIVES = {'plane': plane, 'box': box, 'cylinder': cylinder, 'cone': cone,
              'sphere': sphere, 'wall': wall, 'ramp': ramp}

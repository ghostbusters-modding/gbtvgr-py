"""Vectors, matrices and the game's orientation convention, on numpy.

The game is Y-up right-handed and an actor's `orient` is (yaw, pitch, roll) in
degrees, applied roll about Z, then pitch about X, then yaw about Y. That is
the order the placement code in TheHarbor proved in-game; nothing here invents
a second one.
"""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import math

import numpy as np

F32 = np.float32


def vec3(x=0.0, y=0.0, z=0.0):
    return np.array([x, y, z], dtype=F32)


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return (v / n).astype(F32) if n > 1e-12 else np.zeros_like(v, dtype=F32)


def identity():
    return np.identity(4, dtype=F32)


def translation(t):
    m = identity()
    m[:3, 3] = t
    return m


def scaling(s):
    m = identity()
    s = np.broadcast_to(np.asarray(s, dtype=F32), (3,))
    m[0, 0], m[1, 1], m[2, 2] = s
    return m


def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = identity()
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = identity()
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def rot_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = identity()
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def orient_matrix(orient):
    """(yaw, pitch, roll) degrees -> 4x4 rotation, game order."""
    yaw, pitch, roll = orient
    return rot_y(yaw) @ rot_x(pitch) @ rot_z(roll)


def matrix_to_orient(m):
    """Inverse of orient_matrix for the yaw/pitch/roll it can represent."""
    r = np.asarray(m, dtype=np.float64)[:3, :3]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, r[2, 1]))))
    if abs(r[2, 1]) < 0.9999:
        yaw = math.degrees(math.atan2(-r[2, 0], r[2, 2]))
        roll = math.degrees(math.atan2(-r[0, 1], r[1, 1]))
    else:
        yaw = math.degrees(math.atan2(r[0, 2], r[0, 0]))
        roll = 0.0
    return (yaw, pitch, roll)


def compose(pos, orient, scale=(1.0, 1.0, 1.0)):
    return translation(pos) @ orient_matrix(orient) @ scaling(scale)


def transform_points(m, pts):
    pts = np.asarray(pts, dtype=F32).reshape(-1, 3)
    out = pts @ m[:3, :3].T + m[:3, 3]
    return out.astype(F32)


def transform_normals(m, nrm):
    r = np.asarray(m, dtype=np.float64)[:3, :3]
    inv_t = np.linalg.inv(r).T
    out = np.asarray(nrm, dtype=np.float64).reshape(-1, 3) @ inv_t.T
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return (out / n).astype(F32)


def perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=F32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(half_w, half_h, near, far):
    m = identity()
    m[0, 0] = 1.0 / half_w
    m[1, 1] = 1.0 / half_h
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    eye = np.asarray(eye, dtype=np.float64)
    f = normalize(np.asarray(target, dtype=np.float64) - eye).astype(np.float64)
    s = normalize(np.cross(f, np.asarray(up, dtype=np.float64))).astype(np.float64)
    if np.linalg.norm(s) < 1e-9:
        s = normalize(np.cross(f, (0.0, 0.0, 1.0))).astype(np.float64)
    u = np.cross(s, f)
    m = identity()
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -s @ eye, -u @ eye, f @ eye
    return m


def bbox_of(pts):
    pts = np.asarray(pts, dtype=F32).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros(3, F32), np.zeros(3, F32)
    return pts.min(axis=0), pts.max(axis=0)


def ray_aabb(origin, direction, lo, hi):
    """Slab test; returns the entry distance or None."""
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / direction
        t0 = (np.asarray(lo, dtype=np.float64) - origin) * inv
        t1 = (np.asarray(hi, dtype=np.float64) - origin) * inv
    tmin = np.nanmax(np.minimum(t0, t1))
    tmax = np.nanmin(np.maximum(t0, t1))
    if tmax < max(tmin, 0.0):
        return None
    return max(tmin, 0.0)


def ray_triangles(origin, direction, v0, v1, v2, cull=False):
    """Moeller-Trumbore over arrays of triangles; returns (t, index) of the
    nearest hit or (None, -1)."""
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    e1 = np.asarray(v1, dtype=np.float64) - v0
    e2 = np.asarray(v2, dtype=np.float64) - v0
    p = np.cross(direction, e2)
    det = np.einsum('ij,ij->i', e1, p)
    if cull:
        ok = det > 1e-9
    else:
        ok = np.abs(det) > 1e-9
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / det
        tv = origin - v0
        u = np.einsum('ij,ij->i', tv, p) * inv
        q = np.cross(tv, e1)
        v = np.einsum('j,ij->i', direction, q) * inv
        t = np.einsum('ij,ij->i', e2, q) * inv
        ok &= (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-6)
    if not ok.any():
        return None, -1
    idx = np.where(ok)[0]
    best = idx[np.argmin(t[idx])]
    return float(t[best]), int(best)


def face_normals(verts, faces):
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln < 1e-20] = 1.0
    return (n / ln).astype(F32)


def vertex_normals(verts, faces):
    """Area-weighted smooth normals."""
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    out = np.zeros_like(v)
    for k in range(3):
        np.add.at(out, f[:, k], n)
    ln = np.linalg.norm(out, axis=1, keepdims=True)
    ln[ln < 1e-20] = 1.0
    return (out / ln).astype(F32)


def pack_color(rgba):
    """(r, g, b, a) 0..255 -> D3DCOLOR u32 (stored B, G, R, A in memory)."""
    r, g, b, a = (int(max(0, min(255, c))) for c in rgba)
    return (a << 24) | (r << 16) | (g << 8) | b


def unpack_color(c):
    return ((c >> 16) & 255, (c >> 8) & 255, c & 255, (c >> 24) & 255)

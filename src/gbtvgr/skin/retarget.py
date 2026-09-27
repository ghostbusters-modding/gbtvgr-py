"""Re-pose a skinned source onto a target rig: each source bone is moved onto the target's
joint, turned to the target's bone direction and stretched to its length (RIGID bones are not)."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import numpy as np

from ..mesh import bfm
from .rigs import AIM, RIGID


def rot_between(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


class Rig:
    """A shipped mesh and its skeleton, with bind-pose joint positions from block A."""
    def __init__(self, bfm_bytes, skb_bytes):
        self.m = bfm.parse(bfm_bytes)
        self.bones = bfm.read_skb_bones(skb_bytes)
        self.index = {b['name']: i for i, b in enumerate(self.bones)}
        A = np.frombuffer(self.m['bone_a'], '<f4').reshape(-1, 3).astype(float)
        self.joints = np.zeros_like(A)
        for i, b in enumerate(self.bones):
            self.joints[i] = A[i] + (self.joints[b['parent']] if b['parent'] >= 0 else 0)


def fit(skel, prims, rig):
    """Source skeleton in common bone names -> (X, basis, per-joint affine, per-joint rotation, scale)."""
    names = skel['names']; J = skel['bind'][:, :3, 3]; ix = {n: i for i, n in enumerate(names)}
    GW, gidx = rig.joints, rig.index
    left = J[ix['bone_L_upperarm']] - J[ix['bone_R_upperarm']]; left /= np.linalg.norm(left)
    up = J[ix['bone_head']] - J[ix['bone_hips']]; up -= left * (up @ left); up /= np.linalg.norm(up)
    fwd = np.cross(left, up)
    Bm = np.array([-left, up, fwd])       # game +x is the character's right
    floor = (np.vstack([p['pos'] for p in prims]) @ Bm.T)[:, 1].min()
    Jg = J @ Bm.T
    s = (GW[gidx['bone_head']][1] - 0.0) / (Jg[ix['bone_head']][1] - floor)
    t = np.array([GW[gidx['bone_hips']][0] - s * Jg[ix['bone_hips']][0], -s * floor,
                  GW[gidx['bone_hips']][2] - s * Jg[ix['bone_hips']][2]])
    X = lambda p: s * (p @ Bm.T) + t
    MJ = X(J)
    kids = {}
    for i, p in enumerate(skel['parent']):
        kids.setdefault(p, []).append(i)
    M = [np.eye(4) for _ in names]; Rj = [np.eye(3) for _ in names]
    for n, i in ix.items():
        if n not in gidx: continue
        gb = GW[gidx[n]]
        if n in RIGID:
            gb = M[skel['parent'][i]][:3, :3] @ MJ[i] + M[skel['parent'][i]][:3, 3]
        aim = AIM.get(n, 'x')
        if aim == 'x':
            ch = [c for c in kids.get(i, []) if names[c] in gidx]
            aim = names[ch[0]] if len(ch) == 1 else None
        R3 = np.eye(3); S3 = np.eye(3)
        if aim:
            a = MJ[ix[aim]] - MJ[i]; b = GW[gidx[aim]] - gb
            R3 = rot_between(a, b)
            ah = a / np.linalg.norm(a)
            if n not in RIGID:
                S3 = np.eye(3) + (np.linalg.norm(b) / np.linalg.norm(a) - 1) * np.outer(ah, ah)
        L = np.eye(4); L[:3, :3] = R3 @ S3; L[:3, 3] = gb - R3 @ S3 @ MJ[i]
        M[i] = L; Rj[i] = R3
    return X, Bm, M, Rj, s


def pose(skel, prims, rig):
    """Every primitive re-posed into the rig's bind pose, with weights as rig bone indices."""
    X, Bm, M, Rj, s = fit(skel, prims, rig)
    names = skel['names']; out = []
    for pr in prims:
        P = X(pr['pos']); N = pr['nrm'] @ Bm.T
        P2 = np.zeros_like(P); N2 = np.zeros_like(N)
        for k in range(4):
            w = pr['weights'][:, k:k+1]; jj = pr['joints'][:, k]
            Ms = np.array([M[j] for j in jj]); Rs = np.array([Rj[j] for j in jj])
            P2 += w * (np.einsum('nij,nj->ni', Ms[:, :3, :3], P) + Ms[:, :3, 3])
            N2 += w * np.einsum('nij,nj->ni', Rs, N)
        N2 /= np.linalg.norm(N2, axis=1, keepdims=True)
        gbj = np.vectorize(lambda j: rig.index.get(names[j], -1))(pr['joints'])
        if (gbj[pr['weights'] > 0] < 0).any():
            bad = sorted({names[j] for j in pr['joints'][pr['weights'] > 0] if names[j] not in rig.index})
            raise ValueError('%s: weights on bones the rig lacks: %s' % (pr['mat'], ', '.join(bad)))
        out.append(dict(pr, P=P2, N=N2, gbj=gbj))
    return out, s

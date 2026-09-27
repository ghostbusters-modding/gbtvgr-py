"""Held props (.smb): an outside model into a shipped prop's frame, keeping its parts,
attach points and poses so the game's by-name and by-index lookups still land."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import struct
import numpy as np

from ..mesh import smb
from ..smb import NO_PARENT, pose_nodes
from ..wire import f32_to_half
from . import emit, source
from .autorig import AXES


def _basis(fwd, up):
    """Rows: the source's left, up and forward, with left = up x forward."""
    f = np.array(AXES[fwd], float); u = np.array(AXES[up], float)
    return np.array([np.cross(u, f), u, f])


def place(P, N, cfg):
    """Source points into the target frame: the game draws mirrored, so source left lands on -left."""
    S = _basis(cfg['forward'], cfg.get('up', '+y'))
    T = _basis(cfg.get('target_forward', '+z'), cfg.get('target_up', '+y'))
    M = np.array([-T[0], T[1], T[2]]).T @ S          # source xyz -> target xyz
    return P @ M.T, N @ M.T


def _world(m, pose=0):
    W = pose_nodes(m, pose)
    return [(np.array(R), np.array(t)) for R, t in W]


def _part_points(p):
    st = smb.decl_stride(p['decl'])
    return np.array([struct.unpack_from('<3f', p['vdata'], i * st) for i in range(p['nverts'])])


def _collapse(p):
    st = smb.decl_stride(p['decl']); v = bytearray(p['vdata'])
    for i in range(p['nverts']): v[i * st:i * st + 12] = b'\0' * 12
    p['vdata'] = bytes(v)


def _part(name, mat_idx, P, N, UV, tris):
    verts = [(tuple(map(float, a)), tuple(map(float, b)), (float(c[0]), float(c[1]))) for a, b, c in zip(P, N, UV)]
    tb = smb.compute_tangents(verts, [tuple(map(int, t)) for t in tris])
    vd = bytearray()
    for (pos, nrm, uv), (t, b) in zip(verts, tb):
        vd += smb.pack_geo(pos, nrm, uv)
        vd += struct.pack('<3f3f2HI', *t, *b, f32_to_half(uv[0]), f32_to_half(uv[1]), 0xFFFFFFFF)
    idata = b''.join(struct.pack('<3H', *map(int, t)) for t in tris)
    lo, hi = P.min(0), P.max(0)
    return {'name': name, 'mat_idx': mat_idx, 'bbox': struct.pack('<6f', *lo, *hi),
            'datasize': len(vd) + len(idata), 'decl': list(smb.DECL60), 'nverts': len(P),
            'nprims': len(tris), 'f18': 0, 'morphs': [], 'vdata': bytes(vd), 'idata': idata, 'mdata': []}


def build(data, prims, cfg, mats):
    """data: the shipped .smb; mats: source material -> (material ref, uv function)."""
    m = smb.parse(data)
    main = cfg.get('main_part', 0)
    fit = cfg.get('fit_parts', [main])
    W0 = _world(m)
    box = np.vstack([_part_points(m['parts'][i]) @ W0[i][0].T + W0[i][1] for i in fit])
    lo, hi = box.min(0), box.max(0)

    P = np.vstack([q['pos'] for q in prims]); N = np.vstack([q['nrm'] for q in prims])
    P, N = place(P, N, cfg)
    fwd = int(np.argmax(np.abs(AXES[cfg.get('fit_axis', cfg.get('target_forward', '+z'))])))
    s = (hi[fwd] - lo[fwd]) / np.ptp(P[:, fwd]) * cfg.get('scale', 1.0)
    P = P * s
    ctr = (lo + hi) / 2 - (P.min(0) + P.max(0)) / 2
    ctr[fwd] = hi[fwd] - P[:, fwd].max()   # the far end (muzzle, antenna) stays where the old one was
    P = P + ctr
    nlo, nhi = P.min(0), P.max(0)

    # new geometry lives in the main part's frame
    R, t = W0[main]
    local = (P - t) @ R
    Nl = N @ R
    name = m['parts'][main]['name']
    refs = {}
    for ref, _ in mats.values():
        if ref not in refs:
            m['materials'].append({'ref': ref.encode(), 'refpad': b'\0' * ((-(len(ref) + 1)) % 4), 'embedded': None})
            refs[ref] = len(m['materials']) - 1
    new, a = [], 0
    by_ref = {}
    for q in prims:
        n = len(q['pos']); ref, uvf = mats[q['mat']]
        by_ref.setdefault(ref, []).append((slice(a, a + n), q, uvf(q['uv']))); a += n
    for ref, items in by_ref.items():
        Pp = np.vstack([local[sl] for sl, _, _ in items]); Np = np.vstack([Nl[sl] for sl, _, _ in items])
        UV = np.vstack([uv for _, _, uv in items])
        tris, b = [], 0
        for _sl, q, _uv in items:
            tris.append(q['tris'][:, ::-1] + b)   # the mirror in place() turns the winding over
            b += len(q['pos'])
        new.append(_part(name, refs[ref], Pp, Np, UV, np.vstack(tris)))

    for i, p in enumerate(m['parts']): _collapse(p)
    first, extra = new[0], new[1:]
    m['parts'][main] = first
    nold = len(m['parts'])
    if extra:
        # extra parts go after the old ones; nodes behind them shift down
        par = m['parents']
        if par is not None:
            shift = lambda x: x + len(extra) if x != NO_PARENT and x >= nold else x
            par = [shift(x) for x in par]
            par = par[:nold] + [par[main]] * len(extra) + par[nold:]
            m['parents'] = par
            m['poses'] = [rows[:nold] + [rows[main]] * len(extra) + rows[nold:] for rows in m['poses']]
        m['parts'] += extra

    # attach points keep their place relative to the box they sat in
    naux = len(m['auxnames'])
    if naux and m['poses']:
        base = len(m['parts']) + len(m['collisions'])
        for k in range(m['nposes']):
            Wk = _world(m, k)
            for j in range(naux):
                node = base + j
                wp = Wk[node][1]
                frac = (wp - lo) / np.where(hi - lo == 0, 1, hi - lo)
                target = nlo + frac * (nhi - nlo)
                pi = m['parents'][node]
                if pi == NO_PARENT:
                    tl = target
                else:
                    Rp, tp = Wk[pi]
                    tl = (target - tp) @ Rp
                q, _ = m['poses'][k][node]
                m['poses'][k][node] = (q, struct.pack('<3f', *tl))
    allb = np.vstack([np.frombuffer(m['bbox'], '<f4').reshape(2, 3), nlo[None], nhi[None]])
    m['bbox'] = struct.pack('<6f', *allb.min(0), *allb.max(0))
    m['hdrpad'] = b''
    probe = smb.build(m)
    datalen = sum(len(p['vdata']) + len(p['idata']) + sum(len(x) for x in p['mdata']) for p in m['parts'])
    m['hdrpad'] = b'\0' * ((-(len(probe) - datalen)) % 16)
    out = smb.build(m)
    if smb.build(smb.parse(out)) != out:
        raise AssertionError('rebuilt prop does not round-trip')
    return out, dict(scale=round(float(s), 5), verts=int(sum(p['nverts'] for p in new)),
                     tris=int(sum(p['nprims'] for p in new)), parts=len(m['parts']), materials=len(refs))


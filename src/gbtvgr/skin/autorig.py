"""Rig an unrigged human: joints from body landmarks, weights from distance to bone segments.
Guesses are meant to be checked on the render; recipe [autorig.landmarks] overrides any joint."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import numpy as np

AXES = {'+x': (1, 0, 0), '-x': (-1, 0, 0), '+y': (0, 1, 0), '-y': (0, -1, 0), '+z': (0, 0, 1), '-z': (0, 0, -1)}

# joint -> parent, in the common bone names; left is +x in the working frame
PARENT = {'bone_hips': None, 'bone_spine1': 'bone_hips', 'bone_spine2': 'bone_spine1',
          'bone_spine3': 'bone_spine2', 'bone_neck': 'bone_spine3', 'bone_head': 'bone_neck',
          'bone_head_top': 'bone_head'}
for _s in 'LR':
    PARENT.update({'bone_%s_clavicle1' % _s: 'bone_spine3', 'bone_%s_upperarm' % _s: 'bone_%s_clavicle1' % _s,
                   'bone_%s_forearm' % _s: 'bone_%s_upperarm' % _s, 'bone_%s_hand' % _s: 'bone_%s_forearm' % _s,
                   'bone_%s_middle1' % _s: 'bone_%s_hand' % _s, 'bone_%s_middle2' % _s: 'bone_%s_middle1' % _s,
                   'bone_%s_thigh' % _s: 'bone_hips', 'bone_%s_calf' % _s: 'bone_%s_thigh' % _s,
                   'bone_%s_ankle' % _s: 'bone_%s_calf' % _s, 'bone_%s_foot' % _s: 'bone_%s_ankle' % _s,
                   'bone_%s_toe' % _s: 'bone_%s_foot' % _s})
# bones that take weight, each a segment to the listed child
SEGMENT = {'bone_hips': 'bone_spine1', 'bone_spine1': 'bone_spine2', 'bone_spine2': 'bone_spine3',
           'bone_spine3': 'bone_neck', 'bone_neck': 'bone_head', 'bone_head': 'bone_head_top'}
for _s in 'LR':
    SEGMENT.update({'bone_%s_clavicle1' % _s: 'bone_%s_upperarm' % _s,
                    'bone_%s_upperarm' % _s: 'bone_%s_forearm' % _s,
                    'bone_%s_forearm' % _s: 'bone_%s_hand' % _s, 'bone_%s_hand' % _s: 'bone_%s_middle2' % _s,
                    'bone_%s_thigh' % _s: 'bone_%s_calf' % _s, 'bone_%s_calf' % _s: 'bone_%s_ankle' % _s,
                    'bone_%s_ankle' % _s: 'bone_%s_foot' % _s, 'bone_%s_foot' % _s: 'bone_%s_toe' % _s})


def frame(P, cfg):
    """Working frame: y up, z forward, x the character's left; floor at 0, centred on x and z."""
    up = np.array(AXES[cfg.get('up', '+y')], float)
    fwd = cfg.get('forward')
    if fwd is None:
        # toes stick out past the ankles: the lowest band leans forward
        h = P @ up; lo = h.min(); H = h.max() - lo
        rest = [a for a in ('+x', '+z', '-x', '-z', '+y', '-y') if abs(np.array(AXES[a]) @ up) < 0.5]
        feet = P[h < lo + 0.03 * H]; shins = P[(h > lo + 0.1 * H) & (h < lo + 0.25 * H)]
        fwd = max(rest, key=lambda a: (feet @ np.array(AXES[a])).mean() - (shins @ np.array(AXES[a])).mean())
    f = np.array(AXES[fwd], float)
    left = np.cross(up, f)
    B = np.array([left, up, f])
    Q = P @ B.T
    Q -= [(Q[:, 0].min() + Q[:, 0].max()) / 2, Q[:, 1].min(), 0]
    return B, Q


def _centroid(Q, lo, hi, xmin=-1e9, xmax=1e9):
    # sparse rips leave bands empty: widen until something is in them
    for _ in range(6):
        s = Q[(Q[:, 1] >= lo) & (Q[:, 1] < hi) & (Q[:, 0] >= xmin) & (Q[:, 0] <= xmax)]
        if len(s): return s.mean(0)
        lo, hi = lo - (hi - lo) / 2, hi + (hi - lo) / 2
    return None


def _crotch(Q, T, H):
    """Lowest height where the surface crosses the centre line: below it the legs stand apart."""
    tx, ty = Q[T, 0], Q[T, 1]
    cross = (tx.min(1) < 0) & (tx.max(1) > 0)
    lo, hi = ty.min(1)[cross], ty.max(1)[cross]
    for y in np.arange(0.1 * H, 0.6 * H, 0.005 * H):
        if ((lo <= y) & (hi >= y)).any():
            # a dress hem closes the gap far too low: fall back to a human crotch
            return y if y >= 0.33 * H else None
    return None


def _neck(Q, H):
    """Narrowest slice of the central column between chest and chin."""
    best = None
    for y in np.arange(0.70 * H, 0.90 * H, 0.005 * H):
        s = Q[(Q[:, 1] > y - 0.015 * H) & (Q[:, 1] < y + 0.015 * H) & (np.abs(Q[:, 0]) < 0.15 * H)]
        if len(s) < 3: continue
        w = np.ptp(s[:, 0])
        if best is None or w < best[0]: best = (w, y)
    return best[1] if best else 0.843 * H


# heights along hips -> neck, and neck -> crown, measured on the rookie
SPINE_T = {'bone_spine1': 0.179, 'bone_spine2': 0.374, 'bone_spine3': 0.656}
SHOULDER_T, CLAVICLE_T, HEAD_T = 0.813, 0.861, 0.31


def landmarks(Q, T, cfg=None):
    cfg = cfg or {}
    H = Q[:, 1].max()
    J = {}
    yc = cfg['crotch'] * H if 'crotch' in cfg else (_crotch(Q, T, H) or 0.47 * H)
    yn = cfg['neck'] * H if 'neck' in cfg else _neck(Q, H)
    yh = yc + 0.05 * H
    tz = _centroid(Q, yh, yn, -0.08 * H, 0.08 * H)[2]
    J['bone_hips'] = np.array([0, yh, tz])
    for name, t in SPINE_T.items():
        y = yh + t * (yn - yh)
        c = _centroid(Q, y - 0.02 * H, y + 0.02 * H, -0.07 * H, 0.07 * H)
        J[name] = np.array([0, y, c[2]])
    J['bone_neck'] = np.array([0, yn, _centroid(Q, yn - 0.02 * H, yn + 0.02 * H, -0.07 * H, 0.07 * H)[2]])
    yhd = yn + HEAD_T * (H - yn)
    J['bone_head'] = np.array([0, yhd, _centroid(Q, yhd - 0.03 * H, yhd + 0.03 * H, -0.1 * H, 0.1 * H)[2]])
    J['bone_head_top'] = np.array([0, H, J['bone_head'][2]])
    ys = yh + SHOULDER_T * (yn - yh)
    chest = Q[(Q[:, 1] > J['bone_spine3'][1] - 0.01 * H) & (Q[:, 1] < J['bone_spine3'][1] + 0.01 * H)
              & (np.abs(Q[:, 0]) < 0.2 * H)]
    sx = min(0.094 * H, 0.85 * np.abs(chest[:, 0]).max()) if len(chest) else 0.094 * H
    upper = Q[(Q[:, 1] > yc + 0.1 * H) & (Q[:, 1] < yn + 0.05 * H)]
    half_span = np.abs(upper[:, 0]).max()
    for s, sg in (('L', 1), ('R', -1)):
        box = (0, 0.3 * H) if sg > 0 else (-0.3 * H, 0)
        th = _centroid(Q, yc - 0.06 * H, yc - 0.02 * H, *box)
        kn = _centroid(Q, 0.525 * yc - 0.02 * H, 0.525 * yc + 0.02 * H, *box)
        an = _centroid(Q, 0.035 * H, 0.06 * H, *box)
        J['bone_%s_thigh' % s] = np.array([th[0], yc, th[2]])
        J['bone_%s_calf' % s] = kn
        J['bone_%s_ankle' % s] = an
        side = Q[(Q[:, 1] < 0.04 * H) & (sg * Q[:, 0] > 0)]
        tip = side[side[:, 2].argmax()]
        J['bone_%s_toe' % s] = np.array([an[0], 0.01 * H, tip[2]])
        J['bone_%s_foot' % s] = np.array([an[0], 0.015 * H, an[2] + 0.7 * (tip[2] - an[2])])
        S = np.array([sg * sx, ys, J['bone_spine3'][2]])
        J['bone_%s_clavicle1' % s] = np.array([sg * 0.2 * sx, yh + CLAVICLE_T * (yn - yh), J['bone_spine3'][2] + 0.01 * H])
        J['bone_%s_upperarm' % s] = S
        if half_span > 0.3 * H:
            arm = upper[sg * upper[:, 0] > 0.15 * H]
            tip = arm[(sg * arm[:, 0]).argmax()]
        else:
            cand = Q[(sg * Q[:, 0] > 1.5 * sx) & (Q[:, 1] > 0.3 * H) & (Q[:, 1] < ys)]
            tip = cand[np.linalg.norm(cand - S, axis=1).argmax()]
        W = S + 0.78 * (tip - S)
        J['bone_%s_forearm' % s] = S + 0.5 * (W - S)
        J['bone_%s_hand' % s] = W
        J['bone_%s_middle1' % s] = W + 0.45 * (tip - W)
        J['bone_%s_middle2' % s] = tip
    return J, H


def _seg_dist(P, a, b):
    ab = b - a; t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


ARM = ('upperarm', 'forearm', 'hand')


def weights(Q, J, H, arm_rows=None):
    names = list(SEGMENT)
    D = np.column_stack([_seg_dist(Q, J[n], J[SEGMENT[n]]) for n in names])
    if arm_rows is not None:
        # arms hanging against the body: only the pieces named as arms may follow them
        is_arm = np.array([n.endswith(ARM) for n in names])
        D[np.ix_(~arm_rows, is_arm)] = np.inf
        D[np.ix_(arm_rows, ~is_arm & ~np.array([n.endswith('clavicle1') for n in names]))] = np.inf
    # never across the body: a vertex left of centre takes no right-side bone
    side = np.array([1 if '_L_' in n else -1 if '_R_' in n else 0 for n in names])
    far = np.outer(np.sign(Q[:, 0]) * (np.abs(Q[:, 0]) > 0.02 * H), np.ones(len(names))) * side < 0
    D[far] = np.inf
    k = np.argsort(D, axis=1)[:, :3]
    d = np.take_along_axis(D, k, axis=1) + 1e-4 * H
    w = 1 / d ** 4
    w[~np.isfinite(d)] = 0
    w /= w.sum(1, keepdims=True)
    return names, k, w


def rig(prims, cfg):
    """-> (skeleton in common names, prims with joints/weights), all in source space."""
    P = np.vstack([p['pos'] for p in prims])
    B, Q = frame(P, cfg)
    T, a = [], 0
    for p in prims:
        T.append(p['tris'] + a); a += len(p['pos'])
    J, H = landmarks(Q, np.vstack(T), cfg)
    for n, v in cfg.get('landmarks', {}).items():
        J[n] = np.array(v, float) * H     # overrides are fractions of the height, in the working frame
    names = list(PARENT)
    offset = P @ B.T - Q   # working frame back to source: p = B.T (q + offset)
    o = offset[0]
    to_src = lambda q: (q + o) @ B
    bind = np.array([np.eye(4) for _ in names])
    for i, n in enumerate(names): bind[i][:3, 3] = to_src(J[n])
    skel = {'names': names, 'parent': [names.index(PARENT[n]) if PARENT[n] else -1 for n in names],
            'bind': bind, 'frame': B, 'height': H, 'joints_frame': J}
    arm_rows = None
    if cfg.get('arm_materials'):
        arm_rows = np.concatenate([np.full(len(p['pos']), p['mat'] in cfg['arm_materials']) for p in prims])
    seg_names, k, w = weights(Q, J, H, arm_rows)
    idx = np.array([names.index(n) for n in seg_names])[k]
    out, a = [], 0
    for p in prims:
        n = len(p['pos'])
        jj = np.zeros((n, 4), int); ww = np.zeros((n, 4))
        jj[:, :3] = idx[a:a + n]; ww[:, :3] = w[a:a + n]; jj[:, 3] = jj[:, 0]
        out.append(dict(p, joints=jj, weights=ww)); a += n
    return skel, out

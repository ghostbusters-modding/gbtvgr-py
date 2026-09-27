"""The top-level collision BSP: a k-d tree over the section boxes.

The shipped trees are axis-aligned k-d splits whose leaf boxes nest inside
their parents and whose front child is the side where dot(normal, p) >= d
(670 of 698 shipped section centres resolve to their own section that way).
Every point in space reaches a leaf, so nothing is ever outside the world.
"""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import struct

import numpy as np

NODE = 0x30
HUGE = 99999.0
MAX_DEPTH = 24


class Box:
    __slots__ = ('index', 'lo', 'hi')

    def __init__(self, index, lo, hi):
        self.index = index
        self.lo = np.array(lo, dtype=np.float64)
        self.hi = np.array(hi, dtype=np.float64)

    def volume(self):
        return float(np.prod(np.maximum(self.hi - self.lo, 1e-6)))

    def overlaps(self, lo, hi):
        return bool(np.all(self.lo < hi) and np.all(self.hi > lo))

    def contains(self, p):
        return bool(np.all(p >= self.lo) and np.all(p <= self.hi))


def _pack_node(plane, sec, front, back, lo, hi):
    return (struct.pack('<4f', *plane) + struct.pack('<4h', sec, front, back, 0)
            + struct.pack('<6f', *lo, *hi))


def _best_split(boxes, lo, hi):
    """(axis, position) separating the overlapping boxes best, or None."""
    best = None
    for axis in range(3):
        extent = hi[axis] - lo[axis]
        if extent <= 1e-3:
            continue
        cands = set()
        for b in boxes:
            for v in (b.lo[axis], b.hi[axis]):
                if lo[axis] + 1e-3 < v < hi[axis] - 1e-3:
                    cands.add(round(float(v), 3))
        for v in cands:
            front = sum(1 for b in boxes if b.hi[axis] > v)
            back = sum(1 for b in boxes if b.lo[axis] < v)
            both = sum(1 for b in boxes if b.lo[axis] < v and b.hi[axis] > v)
            if front == len(boxes) and back == len(boxes):
                continue
            score = (both, abs(front - back), -extent)
            if best is None or score < best[0]:
                best = (score, axis, v)
    return None if best is None else (best[1], best[2])


def _leaf_section(boxes, lo, hi, all_boxes):
    c = (lo + hi) / 2.0
    inside = [b for b in boxes if b.contains(c)]
    if inside:
        return min(inside, key=Box.volume).index
    if boxes:
        return max(boxes, key=lambda b: _overlap_volume(b, lo, hi)).index
    d = [np.linalg.norm(np.maximum(np.maximum(b.lo - c, 0.0), c - b.hi)) for b in all_boxes]
    return all_boxes[int(np.argmin(d))].index


def _overlap_volume(b, lo, hi):
    o = np.minimum(b.hi, hi) - np.maximum(b.lo, lo)
    return float(np.prod(np.maximum(o, 0.0)))


def build_bsp(sections, pad=64.0):
    """sections: [(lo, hi)] in index order -> (node bytes, glob bytes, node list).
    The root cell is the union of the boxes padded on every side."""
    boxes = [Box(i, lo, hi) for i, (lo, hi) in enumerate(sections)]
    if not boxes:
        raise ValueError('a set needs at least one section')
    nodes = []
    if len(boxes) == 1:
        nodes.append((np.zeros(4), 0, -1, -1, np.full(3, -HUGE), np.full(3, HUGE)))
        return _emit(nodes)
    lo = np.min([b.lo for b in boxes], axis=0) - pad
    hi = np.max([b.hi for b in boxes], axis=0) + pad

    def rec(lo, hi, depth):
        here = [b for b in boxes if b.overlaps(lo, hi)]
        idx = len(nodes)
        nodes.append(None)
        distinct = {b.index for b in here}
        split = _best_split(here, lo, hi) if len(distinct) > 1 and depth < MAX_DEPTH else None
        if split is None:
            nodes[idx] = (np.zeros(4), _leaf_section(here, lo, hi, boxes), -1, -1, lo, hi)
            return idx
        axis, v = split
        plane = np.zeros(4)
        plane[axis] = 1.0
        plane[3] = v
        flo, fhi = lo.copy(), hi.copy()
        flo[axis] = v
        blo, bhi = lo.copy(), hi.copy()
        bhi[axis] = v
        front = rec(flo, fhi, depth + 1)
        back = rec(blo, bhi, depth + 1)
        nodes[idx] = (plane, -1, front, back, lo, hi)
        return idx

    rec(lo, hi, 0)
    return _emit(nodes)


def _emit(nodes):
    out = b''.join(_pack_node(p, s, f, b, lo, hi) for p, s, f, b, lo, hi in nodes)
    return out, struct.pack('<I', 0), nodes


def parse_bsp(blob):
    nodes = []
    for i in range(len(blob) // NODE):
        rec = blob[i * NODE:(i + 1) * NODE]
        plane = np.array(struct.unpack_from('<4f', rec, 0), dtype=np.float64)
        sec, front, back, _pad = struct.unpack_from('<4h', rec, 16)
        bb = struct.unpack_from('<6f', rec, 24)
        nodes.append((plane, sec, front, back, np.array(bb[:3]), np.array(bb[3:])))
    return nodes


def query(nodes, p):
    """The section index a point resolves to, walking planes to a leaf."""
    p = np.asarray(p, dtype=np.float64)
    i, steps = 0, 0
    while steps < 512:
        plane, sec, front, back, _lo, _hi = nodes[i]
        if sec >= 0:
            return sec
        side = plane[0] * p[0] + plane[1] * p[1] + plane[2] * p[2] >= plane[3]
        nxt = front if side else back
        if nxt < 0:
            return -1
        i = nxt
        steps += 1
    return -1


def verify(nodes, sections, samples_per_section=64, seed=1):
    """Sample points inside every section box and count the ones that resolve to
    a different section. Overlapping boxes are reported, not counted as errors."""
    rng = np.random.default_rng(seed)
    boxes = [Box(i, lo, hi) for i, (lo, hi) in enumerate(sections)]
    problems, tested = [], 0
    for b in boxes:
        pts = rng.uniform(b.lo, b.hi, size=(samples_per_section, 3))
        for p in pts:
            got = query(nodes, p)
            tested += 1
            if got == b.index:
                continue
            if 0 <= got < len(boxes) and boxes[got].contains(p):
                continue
            problems.append((b.index, got, [round(float(v), 2) for v in p]))
    return tested, problems


def leaf_count(nodes):
    return sum(1 for n in nodes if n[1] >= 0)

"""Bounding-volume trees: the collision structure .smb models and .bst sections share."""
import struct


def bbox_of(points):
    xs, ys, zs = zip(*points)
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))

def pack_bbox(bb):
    return struct.pack('<6f', *bb)

def union_bbox(boxes):
    bs = [struct.unpack('<6f', b) if isinstance(b, (bytes, bytearray)) else b
          for b in boxes]
    return tuple([min(b[k] for b in bs) for k in range(3)]
                 + [max(b[k] for b in bs) for k in range(3, 6)])


# --- collision (CCollisionMesh + CBoundingVolumeTree) -------------------------
def read_bvt_node(r):
    n = {'bbox': r.take(24)}
    if r.u32() == 0:
        nv, nt = r.u16(), r.u16()
        n['verts'] = r.take(nv * 12)
        n['tris'] = r.take(nt * 4)
    else:
        n['kids'] = [read_bvt_node(r), read_bvt_node(r)]
    return n

def write_bvt_node(w, n):
    w.w(n['bbox'])
    if 'kids' in n:
        w.u32(1)
        write_bvt_node(w, n['kids'][0]); write_bvt_node(w, n['kids'][1])
    else:
        w.u32(0); w.u16(len(n['verts']) // 12); w.u16(len(n['tris']) // 4)
        w.w(n['verts']); w.w(n['tris'])

def bvt_arena(n):
    """Engine arena bytes: internal node = 0x38+0x38 headroom, leaf = 0x18."""
    if 'kids' in n:
        return 0x70 + bvt_arena(n['kids'][0]) + bvt_arena(n['kids'][1])
    return 0x18

def build_bvt(verts, tris, flags):
    """verts: [(x,y,z)], tris: [(a,b,c)], flags: per-tri byte. Real recursive
    median-split tree; leaf <=16 tris and <=31 local verts (5-bit indices)."""
    def centroid(ti):
        return tuple(sum(verts[i][k] for i in tris[ti]) / 3.0 for k in range(3))
    def make(idxs):
        pts = [verts[i] for t in idxs for i in tris[t]]
        bb = bbox_of(pts)
        lverts = sorted(set(i for t in idxs for i in tris[t]))
        if len(idxs) <= 16 and len(lverts) <= 31:
            lmap = {g: l for l, g in enumerate(lverts)}
            vblob = b''.join(struct.pack('<3f', *verts[g]) for g in lverts)
            keyed = []
            for t in idxs:
                a, b, c = (lmap[i] for i in tris[t])
                keyed.append(flags[t] << 15 | c << 10 | b << 5 | a)
            tblob = b''.join(struct.pack('<I', k) for k in keyed)
            return {'bbox': pack_bbox(bb), 'verts': vblob, 'tris': tblob}
        ax = max(range(3), key=lambda k: bb[k+3] - bb[k])
        order = sorted(idxs, key=lambda t: centroid(t)[ax])
        h = len(order) // 2
        return {'bbox': pack_bbox(bb), 'kids': [make(order[:h]), make(order[h:])]}
    def arena(n):
        if 'kids' in n: return 0x70 + arena(n['kids'][0]) + arena(n['kids'][1])
        return 0x18 + len(n['verts']) + len(n['tris'])
    root = make(list(range(len(tris))))
    return root, 0x38 + arena(root)


# --- collision: BVT <-> triangle soup ------------------------------------------
# Leaf triangles pack into one u32: three 5-bit leaf-local vertex indices, so a
# leaf holds at most 32 vertices, then surface flags and a surface-table index.
TRI_SURF_SHIFT = 21

TRI_FLAG_SHIFT = 15

TRI_FLAG_MASK = 0x3F

def bvt_collect(node):
    """BVT -> (verts, tris, surf, tflags), vertices merged across leaves.
    Merging is bit-exact, so collecting the same tree twice gives the same lists."""
    verts, tris, surf, tflags = [], [], [], []
    pool = {}

    def walk(n):
        if 'kids' in n:
            for k in n['kids']:
                walk(k)
            return
        local = []
        for i in range(len(n['verts']) // 12):
            p = struct.unpack_from('<3f', n['verts'], i * 12)
            key = struct.pack('<3f', *p)
            gi = pool.get(key)
            if gi is None:
                gi = pool[key] = len(verts)
                verts.append(p)
            local.append(gi)
        for i in range(len(n['tris']) // 4):
            k = struct.unpack_from('<I', n['tris'], i * 4)[0]
            tris.append((local[k & 31], local[(k >> 5) & 31],
                         local[(k >> 10) & 31]))
            tflags.append((k >> TRI_FLAG_SHIFT) & TRI_FLAG_MASK)
            surf.append(k >> TRI_SURF_SHIFT)

    if node is not None:
        walk(node)
    return verts, tris, surf, tflags

def bvt_build(verts, tris, surf, tflags, inflate=1.0):
    """Returns (node, arena size). Undersizing the arena silently corrupts the
    tree and the section stops colliding; boxes are inflated so flat geometry works."""
    if not tris:
        return None, 0
    packed = [((s & 0x7FF) << 6) | (f & TRI_FLAG_MASK)
              for s, f in zip(surf, tflags)]
    root, arena = build_bvt(verts, tris, packed)
    if inflate:
        _inflate(root, inflate)
    return root, arena

def _inflate(n, d):
    b = struct.unpack('<6f', n['bbox'])
    n['bbox'] = struct.pack('<6f', b[0] - d, b[1] - d, b[2] - d,
                            b[3] + d, b[4] + d, b[5] + d)
    for k in n.get('kids', ()):
        _inflate(k, d)

def bvt_bbox(node):
    return struct.unpack('<6f', node['bbox']) if node else None

"""Source models: GLB (skinned or static) and OBJ+MTL, as primitives plus an optional skeleton."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import json, os, re, struct
import numpy as np

CT = {5120: 'i1', 5121: 'u1', 5122: 'i2', 5123: 'u2', 5125: 'u4', 5126: 'f4'}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def node_matrix(nd):
    if 'matrix' in nd: return np.array(nd['matrix'], float).reshape(4, 4).T
    m = np.eye(4)
    x, y, z, w = nd.get('rotation', [0, 0, 0, 1])
    m[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    m[:3, :3] *= nd.get('scale', [1, 1, 1])
    m[:3, 3] = nd.get('translation', [0, 0, 0])
    return m


def load_glb(path):
    b = open(path, 'rb').read()
    n = struct.unpack_from('<I', b, 12)[0]
    j = json.loads(b[20:20 + n])
    o = 20 + n
    blob = b[o + 8:o + 8 + struct.unpack_from('<I', b, o)[0]]

    def view(i):
        bv = j['bufferViews'][i]; off = bv.get('byteOffset', 0)
        return blob[off:off + bv['byteLength']]

    def acc(i):
        a = j['accessors'][i]; k = NC[a['type']]; dt = np.dtype('<' + CT[a['componentType']])
        bv = j['bufferViews'][a['bufferView']]; stride = bv.get('byteStride', 0) or dt.itemsize * k
        raw = np.frombuffer(view(a['bufferView']), np.uint8, a['count'] * stride, a.get('byteOffset', 0))
        v = raw.reshape(a['count'], stride)[:, :dt.itemsize * k].copy().view(dt).reshape(a['count'], k)
        if a.get('normalized') and dt.kind in 'iu':
            v = v / float(np.iinfo(dt).max)
        return v

    images = [view(im['bufferView']) if 'bufferView' in im else None for im in j.get('images', [])]
    world = {}
    def walk(ni, parent):
        world[ni] = parent @ node_matrix(j['nodes'][ni])
        for c in j['nodes'][ni].get('children', []): walk(c, world[ni])
    for ni in j['scenes'][j.get('scene', 0)]['nodes']: walk(ni, np.eye(4))

    prims = []
    for ni in sorted(world):
        nd = j['nodes'][ni]
        if 'mesh' not in nd: continue
        skinned = 'skin' in nd
        # a skinned mesh lives in the space its inverse bind matrices expect, not its node's
        M = np.eye(4) if skinned else world[ni]
        for pr in j['meshes'][nd['mesh']]['primitives']:
            at = pr['attributes']; mat = j['materials'][pr['material']]
            pbr = mat.get('pbrMetallicRoughness', {})
            tex = pbr.get('baseColorTexture')
            P = acc(at['POSITION']).astype(float) @ M[:3, :3].T + M[:3, 3]
            N = acc(at['NORMAL']).astype(float) @ np.linalg.inv(M[:3, :3])
            if not skinned: N /= np.linalg.norm(N, axis=1, keepdims=True)
            prims.append({'mat': mat['name'], 'color': pbr.get('baseColorFactor', [1, 1, 1, 1]),
                          'image': images[j['textures'][tex['index']]['source']] if tex else None,
                          'pos': P, 'nrm': N,
                          'uv': acc(at['TEXCOORD_0']).astype(float) if 'TEXCOORD_0' in at else np.zeros((len(P), 2)),
                          'tris': acc(pr['indices']).reshape(-1, 3).astype(int),
                          'joints': acc(at['JOINTS_0']).astype(int) if skinned else None,
                          'weights': acc(at['WEIGHTS_0']).astype(float) if skinned else None})
    skel = None
    if j.get('skins'):
        skin = j['skins'][0]
        names = [re.sub(r'^mixamorig:|_\d+$', '', j['nodes'][i]['name']) for i in skin['joints']]
        ibm = acc(skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
        bind = np.linalg.inv(ibm)
        parent = {}
        for ni, nd in enumerate(j['nodes']):
            for c in nd.get('children', []): parent[c] = ni
        jidx = {n: i for i, n in enumerate(skin['joints'])}
        jparent = [jidx.get(parent.get(n), -1) for n in skin['joints']]
        used = set()
        for pr in prims:
            if pr['joints'] is not None: used |= set(pr['joints'][pr['weights'] > 0].ravel().tolist())
        # end joints carry no weights and a placeholder IBM: place them by their node offset
        for k, ni in enumerate(skin['joints']):
            if k not in used and jparent[k] >= 0 and not j['nodes'][ni].get('children'):
                bind[k] = bind[jparent[k]] @ node_matrix(j['nodes'][ni])
        skel = {'names': names, 'parent': jparent, 'bind': bind}
    return {'prims': prims, 'skel': skel}


def _find(base, name):
    """A texture the MTL names, matched without case: rips made on Windows mix it freely."""
    want = os.path.basename(name.replace('\\', '/')).lower()
    for root, _dirs, files in os.walk(base):
        for f in files:
            if f.lower() == want: return os.path.join(root, f)
    return None


def load_obj(path):
    """OBJ + MTL: one primitive per material, vertices unshared across materials."""
    base = os.path.dirname(path)
    V, VT, VN, faces = [], [], [], {}
    mtllib, cur = None, None
    for line in open(path, encoding='latin1'):
        t = line.split()
        if not t: continue
        if t[0] == 'v': V.append([float(x) for x in t[1:4]])
        elif t[0] == 'vt': VT.append([float(x) for x in t[1:3]])
        elif t[0] == 'vn': VN.append([float(x) for x in t[1:4]])
        elif t[0] == 'usemtl': cur = line.strip()[7:]
        elif t[0] == 'mtllib': mtllib = line.strip()[7:]
        elif t[0] == 'f':
            idx = []
            for c in t[1:]:
                p = (c.split('/') + ['', ''])[:3]
                idx.append(tuple(int(x) if x else 0 for x in p))
            for k in range(1, len(idx) - 1):
                faces.setdefault(cur, []).append((idx[0], idx[k], idx[k + 1]))
    maps, kd = {}, {}
    if not (mtllib and os.path.exists(os.path.join(base, mtllib))):
        mtllib = os.path.splitext(os.path.basename(path))[0] + '.mtl'   # some rips drop the mtllib line
    if os.path.exists(os.path.join(base, mtllib)):
        m = None
        for line in open(os.path.join(base, mtllib), encoding='latin1'):
            t = line.split()
            if not t: continue
            if t[0] == 'newmtl': m = line.strip()[7:]
            elif t[0] == 'map_Kd': maps[m] = line.strip()[7:].strip()
            elif t[0] == 'Kd': kd[m] = [float(x) for x in t[1:4]] + [1]
    V, VT, VN = np.array(V), np.array(VT or [[0, 0]]), np.array(VN or [[0, 1, 0]])
    prims = []
    for mat, fs in faces.items():
        keys, tris = {}, []
        for f in fs:
            tri = []
            for c in f:
                if c not in keys: keys[c] = len(keys)
                tri.append(keys[c])
            tris.append(tri)
        ks = list(keys)
        pos = V[[k[0] - 1 if k[0] > 0 else len(V) + k[0] for k in ks]]
        uv = VT[[k[1] - 1 if k[1] > 0 else 0 for k in ks]] * [1, -1] + [0, 1]   # OBJ v runs up
        img = None
        if mat in maps:
            p = _find(base, maps[mat])
            if p: img = open(p, 'rb').read()
        tris = np.array(tris)
        if all(k[2] for k in ks):
            nrm = VN[[k[2] - 1 for k in ks]]
        else:
            nrm = np.zeros_like(pos)
            fn = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
            for c in range(3): np.add.at(nrm, tris[:, c], fn)
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        prims.append({'mat': mat or 'default', 'color': kd.get(mat, [1, 1, 1, 1]), 'image': img,
                      'image_name': maps.get(mat), 'pos': pos, 'nrm': nrm, 'uv': uv, 'tris': tris,
                      'joints': None, 'weights': None})
    return {'prims': prims, 'skel': None}


def load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.glb': return load_glb(path)
    if ext == '.obj': return load_obj(path)
    raise ValueError('%s: convert to .glb or .obj first' % path)

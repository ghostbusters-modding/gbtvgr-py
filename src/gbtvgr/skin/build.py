#!/usr/bin/env python3
"""
Skins: an outside model onto shipped characters, from a recipe (skin.toml).

Commands:
  build <skin.toml> [--game DIR] [--out DIR] [--obj DIR]   write the mod's meshes, materials, textures
  targets                                                  the characters a recipe can name

Recipe:
  source = "model.glb"          # .glb or .obj, relative to the recipe
  rig = "mixamo"                # the source's bone names; "auto" rigs an unrigged human
  targets = ["rookie", "peter"]
  [material]
  name = "hazmat"               # also the texture stem
  dir = 'char\\HZ\\hazmat'      # layer path; files land under art\\<dir>
  template = "jumpsuit_gloves"  # a shipped skinned material to clone
  mode = "swatch"               # flat colours from [material.colors], or "texture" to use the source's images
  [material.colors]
  "source material" = [[r, g, b], specular]
"""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import argparse, os, sys, tomllib
import numpy as np

from ..level.library import Library, default_game_dir
from . import emit, retarget, source
from .rigs import SOURCE_RIGS, TARGETS


def read_recipe(path):
    with open(path, 'rb') as fh:
        r = tomllib.load(fh)
    r['source'] = os.path.join(os.path.dirname(os.path.abspath(path)), r['source'])
    return r


def canonical(skel, rig_name):
    table = SOURCE_RIGS[rig_name]
    return dict(skel, names=[table.get(n, n) for n in skel['names']])


def materials(recipe, prims, lib):
    """-> {source material: (ref, uv function)}, {out path: bytes}."""
    mc = recipe['material']; files = {}
    template = lib.read('materials\\%s.mtb' % mc['template'])
    art = 'art\\' + mc['dir']
    if mc.get('mode', 'swatch') == 'swatch':
        if 'colors' not in mc:   # flat source colours, sRGB from the linear Kd
            mc['colors'] = {p['mat']: [[int(255 * min(1, c) ** (1 / 2.2)) for c in p['color'][:3]], 40]
                            for p in prims}
        order = list(mc['colors'])
        n = len(order)
        mats = {k: (mc['name'], (lambda uv, i=order.index(k): np.column_stack(
            [np.full(len(uv), (i + 0.5) / n), np.full(len(uv), 0.5)]))) for k in order}
        diff = [tuple(mc['colors'][k][0]) for k in order]
        spec = [(mc['colors'][k][1],) * 3 for k in order]
        files['materials\\%s.mtb' % mc['name']] = emit.material(template, mc['dir'], mc['name'])
        for kind, data in (('diff', emit.swatch_tex(diff, 50)), ('spec', emit.swatch_tex(spec, 43)),
                           ('aocc', emit.swatch_tex([(255, 255, 255)] * n, 43)), ('bump', emit.swatch_tex(None, 47))):
            files['%s\\%s_%s.tex' % (art, mc['name'], kind)] = data
        return mats, files
    mats, seen = {}, {}
    for p in prims:
        key = p['image'] if p['image'] is not None else p['mat']
        if key not in seen:
            name = '%s_%d' % (mc['name'], len(seen))
            seen[key] = name
            files['materials\\%s.mtb' % name] = emit.material(template, mc['dir'], name)
            rgb = tuple(int(255 * c ** (1 / 2.2)) for c in p['color'][:3])
            diff = emit.image_tex(p['image']) if p['image'] is not None else emit.swatch_tex([rgb], 50)
            for kind, data in (('diff', diff), ('spec', emit.swatch_tex([(40, 40, 40)], 43)),
                               ('aocc', emit.swatch_tex([(255, 255, 255)], 43)), ('bump', emit.swatch_tex(None, 47))):
                files['%s\\%s_%s.tex' % (art, name, kind)] = data
        mats[p['mat']] = (seen[key], _unshift)
    return mats, files


def _unshift(uv):
    # rips often park UVs whole tiles away; the sampler wraps, so pull them back to 0..1
    if np.ptp(uv[:, 0]) <= 1.001 and np.ptp(uv[:, 1]) <= 1.001:
        return uv - np.floor(uv.min(0) + 1e-6)
    return uv


def build(recipe, lib, obj_dir=None):
    src = source.load(recipe['source'])
    prims = [p for p in src['prims'] if p['mat'] not in recipe.get('drop_materials', [])]
    if recipe['rig'] == 'auto':
        from . import autorig
        skel, prims = autorig.rig(prims, recipe.get('autorig', {}))
    else:
        skel = canonical(src['skel'], recipe['rig'])
    mats, files = materials(recipe, prims, lib)
    report = {}
    for who in recipe['targets']:
        prof = TARGETS[who]
        for f in prof['files']:
            base = 'skeletal\\%s\\%s' % (prof['dir'], f)
            rig = retarget.Rig(lib.read(base + '.bfm'), lib.read(base + '.skb'))
            posed, scale = retarget.pose(skel, prims, rig)
            data, info = emit.build_mesh(rig, posed, prof, mats, recipe.get('compact_materials', False))
            files[base + '.bfm'] = data
            report[f] = dict(info, scale=round(float(scale), 4))
            if obj_dir: write_obj(os.path.join(obj_dir, f + '.obj'), data, set(r for r, _ in mats.values()))
    return files, report


def write_obj(path, data, refs):
    from ..mesh import bfm
    q = bfm.parse(data); off = 1
    names = [e['ref'].decode('latin1') for e in q['materials']]
    with open(path, 'w') as fh:
        for pk in q['packets'][0]:
            v = bfm.decode_vertices(pk)
            fh.write('o %s_%d\n' % ('skin' if names[pk['mat']] in refs else 'kept', pk['parts'][0]))
            for x in v['pos']: fh.write('v %f %f %f\n' % x)
            for t in bfm.decode_tris(pk): fh.write('f %d %d %d\n' % (t[0] + off, t[1] + off, t[2] + off))
            off += pk['nverts']


def build_prop(recipe, lib):
    from . import prop
    src = source.load(recipe['source'])
    prims = [p for p in src['prims'] if p['mat'] not in recipe.get('drop_materials', [])]
    mats, files = materials(recipe, prims, lib)
    report = {}
    for target in recipe['prop']['targets']:
        data, info = prop.build(lib.read(target), prims, recipe['prop'], mats)
        files[target] = data
        report[target.rsplit('\\', 1)[-1]] = info
    return files, report


def cmd_build(a):
    recipe = read_recipe(a.recipe)
    out = a.out or os.path.dirname(os.path.abspath(a.recipe))
    if a.obj: os.makedirs(a.obj, exist_ok=True)
    lib = Library(a.game or default_game_dir()).open()
    try:
        files, report = build_prop(recipe, lib) if 'prop' in recipe else build(recipe, lib, a.obj)
    finally:
        lib.close()
    for name, data in files.items():
        p = os.path.join(out, *name.split('\\'))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'wb') as fh: fh.write(data)
    for f, info in report.items():
        print('%-18s %s' % (f, ' '.join('%s=%s' % kv for kv in info.items())))
    print('%d file(s) written under %s' % (len(files), out))


def cmd_targets(a):
    for k, v in TARGETS.items():
        print('%-8s %s' % (k, ', '.join('skeletal\\%s\\%s.bfm' % (v['dir'], f) for f in v['files'])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('build'); p.add_argument('recipe'); p.add_argument('--game')
    p.add_argument('--out', help='default: the recipe folder'); p.add_argument('--obj', help='also dump OBJs here')
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser('targets'); p.set_defaults(fn=cmd_targets)
    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == '__main__':
    main()

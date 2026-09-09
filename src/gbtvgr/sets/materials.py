#!/usr/bin/env python3
"""
Set materials resolved through .mtb -> .tex -> RGBA.

A set mesh names a material index; the material names layer textures; a layer
texture resolves to an art path. This walks that chain straight out of the
shipped archives -- no multi-gigabyte extraction -- and decodes to plain RGBA.
The BC1/BC3 decoders here are pure python because the PNG path needs Pillow,
which Blender does not ship.

Commands:
  materials <file.bst> [--game DIR]    every material and the texture it resolves to
  preview <material-ref> -o out.png    decode one material's diffuse layer
"""
import argparse
import os
import struct
import sys

from .. import tex
from ..archive.assets import ArtIndex
from ..mesh.mtb import read_material
from ..wire import R

DIFFUSE_LAYER = 0        # .mtb layer type


def layer_name(raw):
    e = raw.find(b'\0')
    return raw[:e if e >= 0 else len(raw)].decode('latin1')


def material_layers(mat):
    """[(type, texture name)] for a parsed .mtb / embedded material record."""
    out = []
    for l in mat['layers']:
        name = layer_name(l['name'])
        if name:
            out.append((struct.unpack('<I', l['fac'])[0], name))
    return out


def tex_path(name):
    """A material layer's texture name -> its path in the art archives.
    The reader prefixes `art` and forces the extension to `.tex`."""
    name = name.replace('/', '\\').lstrip('\\')
    return 'art\\' + os.path.splitext(name)[0] + '.tex'


def resolve_material(entry, index, art):
    """One set material table entry -> (texture path, .tex bytes) or None.
    Handles both shapes: a reference to a shipped .mtb, and an embedded record."""
    mat = entry['embedded']
    if mat is None:
        ref = layer_name(entry['ref'][0])
        if not ref or art is None:
            return None
        blob = art.read('materials\\' + ref.replace('/', '\\') + '.mtb')
        if blob is None:
            return None
        try:
            mat = read_material(R(blob))
        except (ValueError, struct.error):
            return None
    layers = material_layers(mat)
    if not layers:
        return None
    diffuse = next((n for t, n in layers if t == DIFFUSE_LAYER), layers[0][1])
    path = tex_path(diffuse)
    blob = art.read(path) if art else None
    return (path, blob) if blob else (path, None)


def material_image(entry, index, art, max_size=256):
    """(name, w, h, RGBA) for a set material, or None if it cannot be resolved."""
    got = resolve_material(entry, index, art)
    if not got or got[1] is None:
        return None
    path, blob = got
    try:
        w, h, rgba = tex.decode(blob, max_size)
    except (ValueError, struct.error, tex.TexError):
        return None
    return (path, w, h, rgba)


# --- CLI ----------------------------------------------------------------------
def open_art(a):
    dirs = [d for d in (a.dir or []) if d]
    return ArtIndex(a.game, dirs)


def cmd_materials(a):
    from . import bst, geom
    art = open_art(a)
    if not art:
        sys.exit('no art source: pass --game <game dir> or --dir <extracted>')
    with open(a.file, 'rb') as fh:
        m = bst.parse(fh.read())
    used = set()
    for s in m['sections']:
        for me in s['meshes']:
            used.add(struct.unpack('<H', me['h70'])[0])
    for i, e in enumerate(m['materials']):
        got = resolve_material(e, i, art)
        mark = '*' if i in used else ' '
        if got is None:
            print('%s %-40s  (unresolved)' % (mark, geom.material_name(e, i)))
        else:
            path, blob = got
            note = ''
            if blob:
                try:
                    hh = tex.parse_header(blob)
                    note = '  %dx%d fmt %d' % (hh['w'], hh['h'], hh['fmt'])
                except (tex.TexError, struct.error):
                    note = '  (unreadable)'
            else:
                note = '  (not in the archives)'
            print('%s %-40s  %s%s'
                  % (mark, geom.material_name(e, i), path, note))
    print('* = used by a mesh')


def cmd_preview(a):
    art = open_art(a)
    blob = art.read('materials\\' + a.ref.replace('/', '\\') + '.mtb')
    if blob is None:
        sys.exit('no materials\\%s.mtb in the given sources' % a.ref)
    entry = {'ref': (a.ref.encode(), b''), 'embedded': read_material(R(blob))}
    got = material_image(entry, 0, art, a.size)
    if got is None:
        sys.exit('%s: could not resolve or decode a diffuse texture' % a.ref)
    path, w, h, rgba = got
    try:
        from PIL import Image
    except ImportError:
        sys.exit('%s -> %s (%dx%d); install Pillow to write a .png' %
                 (a.ref, path, w, h))
    Image.frombytes('RGBA', (w, h), rgba).save(a.outfile)
    print('%s -> %s -> %s (%dx%d)' % (a.ref, path, a.outfile, w, h))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--game', help='game directory holding the .POD archives')
    ap.add_argument('--dir', action='append',
                    help='directory of extracted art (repeatable, wins over PODs)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('materials')
    p.add_argument('file')
    p.set_defaults(fn=cmd_materials)
    p = sub.add_parser('preview')
    p.add_argument('ref', help=r'material ref, e.g. graveyard\cryptWall')
    p.add_argument('-o', dest='outfile', required=True)
    p.add_argument('--size', type=int, default=512)
    p.set_defaults(fn=cmd_preview)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()

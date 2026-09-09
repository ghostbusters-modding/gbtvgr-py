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
import zlib

from . import tex
from .pod import Pod
from .smb import R, read_material

# PATCH first: it is what the game itself loads first, so a modded texture wins
POD_ORDER = ('PATCH.POD', 'W64ART02.POD', 'W64ART.POD', 'W64MODEL.POD',
             'COMMON.POD')
DIFFUSE_LAYER = 0        # .mtb layer type


class ArtIndex(object):
    """Name -> bytes over the game's archives, extracted dirs winning.
    Only the indices are read, so nothing is ever unpacked to disk."""

    def __init__(self, gamedir=None, dirs=()):
        self.pods = []
        self.map = {}
        self.dirs = [d for d in dirs if d and os.path.isdir(d)]
        if gamedir and os.path.isdir(gamedir):
            for name in POD_ORDER:
                path = os.path.join(gamedir, name)
                if not os.path.isfile(path):
                    continue
                try:
                    p = Pod(path)
                except (OSError, ValueError):
                    continue
                pi = len(self.pods)
                self.pods.append(p)
                for e in p.entries:
                    self.map.setdefault(e['name'].lower().replace('/', '\\'),
                                        (pi, e))

    def __bool__(self):
        return bool(self.pods or self.dirs)

    __nonzero__ = __bool__

    def close(self):
        for p in self.pods:
            p.close()
        self.pods = []

    def read(self, name):
        """name like 'art\\graveyard\\wall_diff.tex'; None when absent."""
        key = name.lower().replace('/', '\\')
        for d in self.dirs:
            path = os.path.join(d, *key.split('\\'))
            if os.path.isfile(path):
                with open(path, 'rb') as fh:
                    return fh.read()
        hit = self.map.get(key)
        if hit is None:
            return None
        pi, e = hit
        try:
            return self.pods[pi].read(e)
        except (OSError, ValueError, zlib.error):
            return None


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


# --- .tex -> RGBA (pure python; gbtvgr.tex's PNG path needs Pillow) ---------
def _rgb565(v):
    r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
    # bit replication, the D3D expansion the hardware uses
    return ((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2))


def _bc_colors(c0, c1, punch):
    a, b = _rgb565(c0), _rgb565(c1)
    if c0 > c1 or not punch:
        return [a, b,
                tuple((2 * a[k] + b[k]) // 3 for k in range(3)),
                tuple((a[k] + 2 * b[k]) // 3 for k in range(3))], None
    return [a, b, tuple((a[k] + b[k]) // 2 for k in range(3)), (0, 0, 0)], 3


def _blocks(w, h):
    return (max(1, (w + 3) // 4), max(1, (h + 3) // 4))


def decode_bc(data, w, h, bc3):
    """BC1 (DXT1) or BC3 (DXT5) -> top-down RGBA bytes."""
    bw, bh = _blocks(w, h)
    out = bytearray(w * h * 4)
    stride = 16 if bc3 else 8
    for by in range(bh):
        for bx in range(bw):
            off = (by * bw + bx) * stride
            if bc3:
                a0, a1 = data[off], data[off + 1]
                abits = int.from_bytes(data[off + 2:off + 8], 'little')
                if a0 > a1:
                    alpha = [a0, a1] + [((7 - i) * a0 + i * a1) // 7
                                        for i in range(1, 7)]
                else:
                    alpha = [a0, a1] + [((5 - i) * a0 + i * a1) // 5
                                        for i in range(1, 5)] + [0, 255]
                coff = off + 8
            else:
                alpha = None
                coff = off
            c0, c1 = struct.unpack_from('<2H', data, coff)
            colors, clear = _bc_colors(c0, c1, not bc3)
            bits = struct.unpack_from('<I', data, coff + 4)[0]
            for py in range(4):
                y = by * 4 + py
                if y >= h:
                    break
                for px in range(4):
                    x = bx * 4 + px
                    if x >= w:
                        break
                    i = (py * 4 + px)
                    ci = (bits >> (2 * i)) & 3
                    r, g, b = colors[ci]
                    if alpha is not None:
                        a = alpha[(abits >> (3 * i)) & 7]
                    else:
                        a = 0 if ci == clear else 255
                    p = (y * w + x) * 4
                    out[p:p + 4] = bytes((r, g, b, a))
    return bytes(out)


def decode_raw(data, w, h, unit):
    out = bytearray(w * h * 4)
    if unit == 4:                                  # BGRA8
        for i in range(w * h):
            b, g, r, a = data[i * 4:i * 4 + 4]
            out[i * 4:i * 4 + 4] = bytes((r, g, b, a))
    else:                                          # RG8 tangent normal
        for i in range(w * h):
            x, y = data[i * 2], data[i * 2 + 1]
            nx, ny = x / 127.5 - 1.0, y / 127.5 - 1.0
            nz = max(0.0, 1.0 - nx * nx - ny * ny) ** 0.5
            out[i * 4:i * 4 + 4] = bytes((x, y, int((nz + 1.0) * 127.5), 255))
    return bytes(out)


def decode(data, max_size=256):
    """.tex bytes -> (w, h, top-down RGBA).  Picks the largest mip that fits
    max_size, so a preview costs a fraction of the full image."""
    h = tex.parse_header(data)
    fi = h['fmt_info']
    if fi['cube']:
        raise ValueError('cubemap (fmt %d) has no flat preview' % h['fmt'])
    levels = tex.mip_dims(h['w'], h['h'])[:max(1, h['mipcount'] or 1)]
    off = 0
    pick = None
    for lw, lh in levels:
        size = tex.level_size(fi, lw, lh)
        if pick is None and max(lw, lh) <= max_size:
            pick = (lw, lh, off, size)
        off += size
    if pick is None:                                # every mip is too big
        lw, lh = levels[-1]
        off -= tex.level_size(fi, lw, lh)
        pick = (lw, lh, off, tex.level_size(fi, lw, lh))
    lw, lh, off, size = pick
    blob = h['payload'][off:off + size]
    if len(blob) < size:
        raise ValueError('truncated mip (%d of %d bytes)' % (len(blob), size))
    if fi['kind'] == 'bc':
        return lw, lh, decode_bc(blob, lw, lh, h['fmt'] == 50)
    return lw, lh, decode_raw(blob, lw, lh, fi['unit'])


def material_image(entry, index, art, max_size=256):
    """(name, w, h, RGBA) for a set material, or None if it cannot be resolved."""
    got = resolve_material(entry, index, art)
    if not got or got[1] is None:
        return None
    path, blob = got
    try:
        w, h, rgba = decode(blob, max_size)
    except (ValueError, struct.error, tex.TexError):
        return None
    return (path, w, h, rgba)


# --- CLI ----------------------------------------------------------------------
def open_art(a):
    dirs = [d for d in (a.dir or []) if d]
    return ArtIndex(a.game, dirs)


def cmd_materials(a):
    from . import bst, bst_geom
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
            print('%s %-40s  (unresolved)' % (mark, bst_geom.material_name(e, i)))
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
                  % (mark, bst_geom.material_name(e, i), path, note))
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

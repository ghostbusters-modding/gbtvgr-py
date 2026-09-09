#!/usr/bin/env python3
"""
Hand-bake the test lightmap tiles for sets\\immortal.bst.

256x256 fmt3 (RAW_BGRA8), single mip. An unmistakable pattern rather than a flat
fill, so a screenshot proves the lightmapped shader path is live and not the
no-lightmap fallback: base gray 90, a bright warm disc at centre, corners down
to ~45, a 32px grid at 160. The shader doubles the value it reads, so a raw byte
of 128 is neutral -- 90 reads as a mild dim, 255 as full overbright.

Headers mirror a real donor tile, which is game content and so is not carried
here. Extract one and point $LIGHTMAP_DONOR at the directory:

    gbtvgr pod extract "<game>/W64ART02.POD" -o out/lightmap -f .tex
    LIGHTMAP_DONOR=out/lightmap/art/lightmap/abyss

ver/zero1/A/C/D come from it verbatim; fmt/w/h are set explicitly. The 16-byte
hash field's algorithm is unresolved, so it too is copied from the matching
donor tile rather than invented.

Usage: make_lightmap.py <out_dir>
Writes 0_0.tex, 0_1.tex, 0_2.tex -- identical pixels, donor-copied headers.
"""
import math
import os
import sys

# Run from a checkout without installing:  PYTHONPATH=src python3 examples/make_lightmap.py
from gbtvgr import tex

W = H = 256
# A donor tile supplies the header fields still unresolved. It is game content,
# so it is read from an installation you extracted yourself, never shipped here.
DONOR_DIR = os.environ.get('LIGHTMAP_DONOR', '')
DONOR_TILES = [os.path.join(DONOR_DIR, '1_%d.tex' % i) for i in range(3)]


def make_pixels():
    """BGRA8, row-major, one pixel = 4 bytes (B,G,R,A)."""
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    corner = math.hypot(cx, cy)
    radius = 96.0
    out = bytearray(W * H * 4)
    for y in range(H):
        for x in range(W):
            dx, dy = x - cx, y - cy
            dist = math.hypot(dx, dy)
            t = min(1.0, dist / corner)
            base = 90.0 * (1.0 - t) + 45.0 * t          # vignette to corners
            df = max(0.0, 1.0 - dist / radius)
            df = df * df * (3.0 - 2.0 * df)              # smoothstep falloff
            b = base * (1.0 - df) + 200.0 * df
            g = base * (1.0 - df) + 240.0 * df
            r = base * (1.0 - df) + 255.0 * df
            if x % 32 == 0 or y % 32 == 0:
                b = g = r = 160.0
            o = (y * W + x) * 4
            out[o + 0] = int(round(b))
            out[o + 1] = int(round(g))
            out[o + 2] = int(round(r))
            out[o + 3] = 255
    return bytes(out)


def build_tile(donor_path, payload):
    dh = tex.parse_header(open(donor_path, 'rb').read())
    hdr = tex.build_header(dh['ver'], dh['hash'], dh['zero1'], 3, W, H,
                            dh['A'], 1, dh['C'], dh['D'])
    data = hdr + payload
    assert len(data) == tex.HEADER_LEN + W * H * 4
    return data


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip().splitlines()[-3])
    if not DONOR_DIR or not os.path.isdir(DONOR_DIR):
        sys.exit("set $LIGHTMAP_DONOR to a directory of extracted donor tiles "
                 "(see this file's docstring); they are game content and are not "
                 "shipped here.")
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    payload = make_pixels()
    for i, donor in enumerate(DONOR_TILES):
        data = build_tile(donor, payload)
        outp = os.path.join(out_dir, '0_%d.tex' % i)
        open(outp, 'wb').write(data)
        print('%s: %d bytes (fmt3 %dx%d)' % (outp, len(data), W, H))


if __name__ == '__main__':
    main()

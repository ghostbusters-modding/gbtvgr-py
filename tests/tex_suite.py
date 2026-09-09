#!/usr/bin/env python3
"""tex_test - check gbtvgr.tex's pure-python .tex decoders.

The add-on cannot use gbtvgr.tex's PNG path (it needs Pillow, which Blender
does not ship), so tex.py decodes BC1/BC3/raw in pure python too.  This compares that
against gbtvgr.tex's own DDS output through Pillow, on real textures pulled
straight out of the shipped archives, and checks the mip picker.

  python3 tests/tex_suite.py [--game <game dir>]

Skips (exit 0) when Pillow or the game archives are unavailable.
"""
import argparse
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))

from gbtvgr import tex                                           # noqa: E402
from gbtvgr.archive import assets                                # noqa: E402

GAME = ""          # no default: pass --game or set $GAME_DIR
WANT = 4          # textures to check per format
FAILED = []


def check(name, cond, detail=""):
    print("%-4s %s%s" % ("ok" if cond else "FAIL", name,
                         "" if cond else "  -- " + str(detail)))
    if not cond:
        FAILED.append(name)


def sample(art, limit=400000):
    """A few small textures of each flat format, from the archives."""
    out = {}
    for key in art.map:
        if not key.endswith('.tex'):
            continue
        if all(len(v) >= WANT for v in out.values()) and len(out) >= 4:
            break
        pi, e = art.map[key]
        if e['usize'] > limit:
            continue
        blob = art.read(key)
        if blob is None:
            continue
        try:
            h = tex.parse_header(blob)
        except Exception:
            continue
        if h['fmt_info']['cube']:
            continue
        got = out.setdefault(h['fmt'], [])
        if len(got) < WANT:
            got.append((key, blob))
    return out


def main(game):
    try:
        from PIL import Image
    except ImportError:
        print("skip: Pillow not installed")
        return 0
    art = assets.ArtIndex(game)
    if not art:
        print("skip: no game archives at %s" % game)
        return 0

    found = sample(art)
    check("textures of several formats were found in the archives",
          len(found) >= 2, sorted(found))

    for fmt in sorted(found):
        for key, blob in found[fmt]:
            h = tex.parse_header(blob)
            w, ht, rgba = tex.decode(blob, max_size=1 << 20)   # top mip
            name = "fmt %d %s" % (fmt, os.path.basename(key))
            if (w, ht) != (h['w'], h['h']):
                check(name + " top mip has the header's size", False,
                      "%dx%d vs %dx%d" % (w, ht, h['w'], h['h']))
                continue
            if fmt == 47:
                # Pillow cannot read the DXGI R8G8 DDS tex.py emits, so check
                # the channels against the payload directly
                src = h['payload'][:w * ht * 2]
                ok = all(rgba[i * 4] == src[i * 2]
                         and rgba[i * 4 + 1] == src[i * 2 + 1]
                         and rgba[i * 4 + 3] == 255
                         for i in range(0, w * ht, 7))
                check(name + " keeps both channels", ok)
                continue
            try:
                ref = Image.open(io.BytesIO(tex.to_dds(blob))).convert('RGBA')
            except NotImplementedError as e:
                print("skip %s (%s)" % (name, e))
                continue
            same = ref.size == (w, ht) and ref.tobytes() == rgba
            if not same and ref.size == (w, ht):
                b = ref.tobytes()
                worst = max(abs(rgba[i] - b[i]) for i in range(len(rgba)))
                check(name + " decodes exactly like tex.py + Pillow", False,
                      "max channel difference %d" % worst)
            else:
                check(name + " decodes exactly like tex.py + Pillow", same,
                      "%s vs %dx%d" % (ref.size, w, ht))

    # mip picker: asking for a small preview must return a small mip
    big = next((b for items in found.values() for _k, b in items
                if tex.parse_header(b)['w'] >= 128), None)
    if big is not None:
        h = tex.parse_header(big)
        w, ht, rgba = tex.decode(big, max_size=64)
        check("the mip picker returns a mip that fits the budget",
              max(w, ht) <= 64 and len(rgba) == w * ht * 4,
              "%dx%d from %dx%d" % (w, ht, h['w'], h['h']))
        w2, ht2, _ = tex.decode(big, max_size=1)
        check("a budget below every mip falls back to the smallest",
              (w2, ht2) == (1, 1) or max(w2, ht2) <= max(w, ht),
              "%dx%d" % (w2, ht2))

    print()
    if FAILED:
        print("%d FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("all texture checks passed")
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--game', default=os.environ.get('GAME_DIR', GAME))
    a = ap.parse_args()
    sys.exit(main(a.game))

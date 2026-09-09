# examples

Four generators that build a shipped format from nothing, so you can see what
each codec wants without an installation to copy from. Run them from a checkout
with `PYTHONPATH=../src`, or after `pip install gbtvgr` with no path at all.

| | |
|---|---|
| `make_obelisk.py <out.obj>` | a mesh with a separate collision hull, ready for `gbtvgr smb from-obj` |
| `make_blank_lvl.py -o <out.lvl>` | an empty but loadable `.lvl` actor table |
| `make_immortal_set.py <src> <out.bst>` | a `.bst` set assembled from meshes |
| `make_lightmap.py <out_dir>` | three 256×256 `fmt3` lightmap tiles |

The mesh path is the one to read first — it round-trips end to end with
nothing installed:

```bash
python3 make_obelisk.py /tmp/ob.obj
gbtvgr smb from-obj /tmp/ob.obj /tmp/ob.smb --auto-collision
gbtvgr smb verify /tmp/ob.smb
```

`make_lightmap.py` is the exception: a tile header carries a 16-byte field (`gbtvgr-docs/formats/tex_format.md`), which
is copied from a donor tile rather than invented. Donor tiles are game content and are not
shipped here — extract your own and point `$LIGHTMAP_DONOR` at them:

```bash
gbtvgr pod extract "<game>/W64ART02.POD" -o out/lightmap -f .tex
LIGHTMAP_DONOR=out/lightmap/art/lightmap/abyss python3 make_lightmap.py out/tiles
```

# gbtvgr

Codecs for the asset formats of GBTVGR (archives, textures, meshes, sets, levels and characters) 
as one Python package and one `gbtvgr` command.

```
  .POD  ──▶ list / extract / create / chain      the archive and its mount order
  .tex  ──▶ DDS, PNG, and back                   BC1/BC3/raw, cubemaps, mipmaps
  .smb  ──▶ OBJ, and back                        static meshes + .mtb materials
  .bst  ──▶ geometry, collision, relayout        sets, the largest format here
  .lvl  ──▶ inspect, validate, diff, edit        the actor table for a level
  .cib  ──▶ inspect, clone, edit                 character / biped definitions
```

## Install

Python 3.9+, no required dependencies.

```bash
pip install -e .            # provides the `gbtvgr` command
pip install -e ".[test]"    # + pytest and Pillow, to run the suite
```

`Pillow` is optional and only used by `gbtvgr tex to-png`; every codec and every
round-trip works without it.

## Use it

```bash
gbtvgr                                        # the nine formats, one line each
gbtvgr pod --help                             # subcommands for one of them

# archives
gbtvgr pod list "<game>/COMMON.POD"
gbtvgr pod extract "<game>/COMMON.POD" -o out/scripts -f .dante
gbtvgr pod create out/IMMORTAL.POD -d mymod/files --next NEXT.POD
gbtvgr pod chain "<game>/PATCH.POD" IMMORTAL.POD       # what mounts after what

# shipping a mod as a chained archive, and taking it back off
gbtvgr patchpod content --game "<game>" --mods mymod/files -o out/IMMORTAL.POD
gbtvgr patchpod install --game "<game>" --pod out/IMMORTAL.POD
gbtvgr patchpod uninstall --game "<game>"

# textures, meshes, sets
gbtvgr tex to-png  art/ui/thing.tex thing.png
gbtvgr smb to-obj  models/prop.smb prop.obj      # writes a .smb.json sidecar
gbtvgr smb from-obj prop.obj prop.smb --sidecar prop.smb.json   # exact rebuild
gbtvgr bst info    sets/abyss.bst
gbtvgr bst-geom to-obj sets/abyss.bst abyss.obj --collision
gbtvgr bst-tex --game "<game>" materials sets/abyss.bst

# levels and characters
gbtvgr lvl actors  world/cemetery2.lvl
gbtvgr lvl schema  --dante-api ...             # per-class fields, from the API table
gbtvgr lvl add-actor world/cemetery2.lvl --class CGhost --name ghost_7 \
                     --pos 10 0 4 -o out/cemetery2.lvl
gbtvgr cib inspect data/biped1/fiend.cib
gbtvgr cib clone   data/biped1/fiend.cib new_fiend.cib
```

The same thing from Python:

```python
from gbtvgr.archive import pod
from gbtvgr.mesh import smb

p = pod.Pod("COMMON.POD")                       # a mounted archive
data = p.read("world\\cemetery2.lvl")           # entries keep the game's separator

mesh = smb.parse(open("prop.smb", "rb").read())
assert smb.build(mesh) == open("prop.smb", "rb").read()     # the standing bar
```

## The corpus

Fidelity is measured against the shipped game assets, so this repository does not carry them

Pull what a check needs out of your own installation:

```bash
gbtvgr pod extract "<game>/W64MODEL.POD" -o out/smb  -f .smb
gbtvgr pod extract "<game>/W64SET.POD"   -o out/sets -f .bst
gbtvgr pod extract "<game>/W64ART02.POD" -o out/mtb  -f .mtb
```

## What is verified

```bash
GAME_DIR="<game>" SMB_CORPUS=out/smb BST_CORPUS=out/sets pytest
pytest --game="<game>" --smb-corpus=out/smb        # note the `=` (tests/conftest.py)
pytest                                             # corpus tests skip; the rest run
```

| check | needs |
|---|---|
| POD6 header, chain field, legacy rewrite, `content`/`install`/`uninstall` | nothing — builds its own archives |
| every shipped `.POD` parses, chain field legible | `--game` |
| `.smb` → parse → rebuild, byte-identical, whole corpus | `--smb-corpus` |
| `.smb` → OBJ → `.smb`, byte-identical, 40-file sample | `--smb-corpus` |
| `.mtb` material tables, byte-identical | `--mtb-corpus` |
| `.bst` → parse → rebuild, byte-identical, every set | `--bst-corpus` |
| `.bst` per-mesh decode/encode **and** a full relayout, byte-identical | `--bst-corpus` |
| `bst_tex`'s pure-python BC1/BC3/raw decoders vs the Pillow path | `--game`, Pillow |

The install/uninstall checks build a fake game directory, run a real install into it, 
and assert that `PATCH.POD`'s content is byte-identical afterwards. 

## Layout

```
src/gbtvgr/
    wire.py           byte reader/writer, and the primitives records are built from
    archive/
        pod.py        POD6: read, write, and the chain field that sets mount order
        patchpod.py   building and installing a mod archive against an installation
        assets.py     name -> bytes across the mounted archives
    mesh/
        smb.py        static meshes, OBJ both ways
        mtb.py        .mtb material tables
        bvt.py        bounding-volume trees: model and section collision
    sets/
        bst.py        .bst sets -- the wire codec
        geom.py       .bst geometry, collision, and relayout
        materials.py  set materials resolved .mtb -> .tex -> RGBA
    tex.py            .tex textures, through DDS and PNG; also decodes without Pillow
    lvl.py            .lvl actor tables
    cib.py            .cib character definitions
    cli.py            the `gbtvgr` command
    data/             dante_api.json -- the script API table, for lvl's field schema
examples/             generators that build each format from nothing
tests/                pytest; the two hand-rolled suites keep their own output
```

`wire` and `archive.pod` are the roots. 

`mesh` builds on `wire`

`sets` builds on `mesh`

`lvl` and `archive.assets` build on the archive reader. 

## Data

`src/gbtvgr/data/dante_api.json` is the script API table recovered from the game with 
every registered class and its properties. 

`gbtvgr lvl schema` uses it to name the fields in an actor table. 
It can be easily replaced without touching code:

```bash
DANTE_API_JSON=/path/to/dante_api.json gbtvgr lvl schema ...
```

## Docs

The format specifications live in the
[gbtvgr-docs](https://github.com/ghostbusters-modding/gbtvgr-docs) repository:

| | |
|---|---|
| [pod_format.md](https://github.com/ghostbusters-modding/gbtvgr-docs/blob/main/formats/pod_format.md) | POD6: header, entries, compression, and the mount chain |
| [tex_format.md](https://github.com/ghostbusters-modding/gbtvgr-docs/blob/main/formats/tex_format.md) | `.tex`: formats, cubemaps, mipmaps, the unresolved hash field |
| [smb_format.md](https://github.com/ghostbusters-modding/gbtvgr-docs/blob/main/formats/smb_format.md) | `.smb`: vertex declarations, render packets, collision, BVT |
| [mtb_format.md](https://github.com/ghostbusters-modding/gbtvgr-docs/blob/main/formats/mtb_format.md) | `.mtb`: the material table `.smb` and `.bst` refer into |
| [bst_format.md](https://github.com/ghostbusters-modding/gbtvgr-docs/blob/main/formats/bst_format.md) | `.bst`: sections, meshes, breakers, events |
| [lvl_format.md](https://github.com/ghostbusters-modding/gbtvgr-docs/blob/main/formats/lvl_format.md) | `.lvl`: the actor table and its typed properties |
| [cib_format.md](https://github.com/ghostbusters-modding/gbtvgr-docs/blob/main/formats/cib_format.md) | `.cib`: character/biped definitions |
| [lightmap_format.md](https://github.com/ghostbusters-modding/gbtvgr-docs/blob/main/formats/lightmap_format.md) | how lightmap tiles are laid out and shaded |

"""
gbtvgr -- asset-format tools for *Ghostbusters: The Video Game Remastered*.

Each codec is verified by round-tripping the retail corpus back to
byte-identical output:

    wire               byte reader/writer, and the shared record primitives
    archive.pod        POD6 archives, and the mount chain that makes mods load
    archive.patchpod   building/installing a chained mod archive against an install
    archive.assets     name -> bytes across the mounted archives
    mesh.smb           static meshes
    mesh.mtb           .mtb material tables
    mesh.bvt           bounding-volume trees: model and section collision
    sets.bst           .bst sets -- the wire codec
    sets.geom          .bst geometry, collision and relayout
    sets.materials     set materials resolved through .mtb -> .tex -> RGBA
    tex                .tex textures  (DXT/BC, cubemaps, mipmaps)
    lvl                .lvl actor tables
    cib                .cib character/biped definitions

Every module is usable as a library and as a subcommand of the `gbtvgr`
command; `python -m gbtvgr <format> --help` lists any of them.  The CLI keeps
the old spelling: `gbtvgr bst-geom` reaches `sets.geom`.

    from gbtvgr.archive import pod
    p = pod.Pod("COMMON.POD")
    data = p.read("world\\\\hotel1a.dante")

This package ships no game content.  It reads the files of an installation you
already own, and the only data it carries is `data/dante_api.json`, the script
API table recovered from `ghost.exe`; set $DANTE_API_JSON to point at another
build's table.
"""

__version__ = "1.0.0"

_MODULES = ("archive", "mesh", "sets", "tex", "lvl", "cib", "wire")

__all__ = list(_MODULES) + ["__version__"]


def __getattr__(name):
    """Import subpackages on demand, so one question does not load every codec."""
    if name in _MODULES:
        import importlib
        mod = importlib.import_module("." + name, __name__)
        globals()[name] = mod
        return mod
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(__all__)

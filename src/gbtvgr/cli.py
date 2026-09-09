"""
The `gbtvgr` command: one entry point in front of the nine format modules.

    gbtvgr <format> <subcommand> [options]

Each format keeps its own argument parser -- this dispatcher only picks the
module and hands it the rest of the command line, so `gbtvgr pod extract ...`
and `python -m gbtvgr.pod extract ...` take exactly the same arguments.
"""

import importlib
import sys

# cli name -> (module path under gbtvgr, one-line summary). The CLI spelling
# uses hyphens; the module lives wherever the package layout puts it.
FORMATS = [
    ("pod",      "archive.pod",      "POD6 archives: list, extract, create, and the mount chain"),
    ("patchpod", "archive.patchpod", "mod archives: build a chained IMMORTAL.POD, install, roll back"),
    ("tex",      "tex",              ".tex textures: DDS/PNG both ways, cubemaps, mipmaps"),
    ("smb",      "mesh.smb",         "static meshes and .mtb material tables: OBJ both ways"),
    ("bst",      "sets.bst",         ".bst sets: the wire codec (parse, verify, rebuild)"),
    ("bst_geom", "sets.geom",        ".bst geometry: meshes, collision, relayout, OBJ export"),
    ("bst_tex",  "sets.materials",   "set materials resolved through .mtb -> .tex -> RGBA"),
    ("lvl",      "lvl",              ".lvl actor tables: inspect, validate, diff, add/remove actors"),
    ("cib",      "cib",              ".cib character definitions: inspect, clone, edit properties"),
]

_BY_CLI = {name.replace("_", "-"): mod for name, mod, _ in FORMATS}
_BY_CLI.update({name: mod for name, mod, _ in FORMATS})   # underscores accepted too


def _usage():
    from . import __version__
    w = max(len(n.replace("_", "-")) for n, _, _ in FORMATS)
    lines = [
        "usage: gbtvgr <format> <subcommand> [options]",
        "",
        "Asset-format tools for Ghostbusters: The Video Game Remastered.",
        "",
        "formats:",
    ]
    lines += ["  %-*s  %s" % (w, n.replace("_", "-"), d) for n, _, d in FORMATS]
    lines += [
        "",
        "  gbtvgr <format> --help   subcommands and options for one format",
        "",
        "gbtvgr %s" % __version__,
    ]
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    if argv[0] in ("-V", "--version"):
        from . import __version__
        print("gbtvgr %s" % __version__)
        return 0

    fmt = _BY_CLI.get(argv[0])
    if fmt is None:
        sys.stderr.write("gbtvgr: unknown format %r\n\n%s\n" % (argv[0], _usage()))
        return 2

    mod = importlib.import_module("." + fmt, __package__)

    # argparse takes `prog` from argv[0], so rewriting it makes a format's own
    # --help render as `gbtvgr pod ...` rather than the bare module file name.
    saved = sys.argv
    sys.argv = ["gbtvgr " + argv[0]] + argv[1:]
    try:
        return mod.main() or 0
    finally:
        sys.argv = saved


if __name__ == "__main__":
    raise SystemExit(main())

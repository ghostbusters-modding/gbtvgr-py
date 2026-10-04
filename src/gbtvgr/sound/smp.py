"""
smp.py - single tool for Ghostbusters The Video Game .smp sound files
(a 160-byte header followed by a plain Ogg Vorbis stream).

  info <file.smp>                  header + stream summary
  verify <file.smp ...>            rebuild header from the Ogg payload, must be byte-identical
  roundtrip-test <dir>             verify every .smp under dir
  to-ogg <in.smp> <out.ogg>        strip the header
  from-ogg <in.ogg> <out.smp>      build a new .smp (see the subcommand's --help)

SMP header (160 bytes, little-endian):
    0  u32  6                          (format/version?)
    4  10B  		               
   24  u32  (duration_ms + 275) * 44   (subtitle timing?)
   28  u32  160                        (header size / payload offset)
   32  u32  size of the Ogg stream in bytes
   36  u8   9
   44  u8   0x10
   48  u16  44100                      (sample rate; lip-sync?)
  other bytes are zero
  160+      Ogg Vorbis stream ("OggS...")
"""
import argparse
import os
import struct
import sys

HEADER_SIZE = 160
MARKER = b"gbmoddingC"
DEFAULT_RATE = 44100


# ----------------------------------------------------------------- Ogg parsing
def ogg_summary(data):
    """Return (channels, sample_rate, total_samples, duration_ms) of an Ogg Vorbis stream."""
    if data[:4] != b"OggS":
        raise ValueError("not an Ogg stream")
    channels = rate = None
    last_granule = 0
    pos, n = 0, len(data)
    first = True
    while pos < n:
        if pos + 27 > n or data[pos:pos + 4] != b"OggS":
            raise ValueError("corrupt Ogg page at 0x%x" % pos)
        granule = struct.unpack_from("<q", data, pos + 6)[0]
        nseg = data[pos + 26]
        if pos + 27 + nseg > n:
            raise ValueError("truncated Ogg page at 0x%x" % pos)
        body = sum(data[pos + 27:pos + 27 + nseg])
        start = pos + 27 + nseg
        if first:
            ident = data[start:start + 16]
            if ident[:7] != b"\x01vorbis" or len(ident) < 16:
                raise ValueError("first packet is not a Vorbis identification header")
            channels = ident[11]
            rate = struct.unpack_from("<I", ident, 12)[0]
            first = False
        if granule >= 0:
            last_granule = granule
        pos = start + body
    if not rate:
        raise ValueError("no sample rate found")
    return channels, rate, last_granule, last_granule * 1000 // rate


# ----------------------------------------------------------------- SMP header
def build_header(ogg, template=None, rate=DEFAULT_RATE):
    _, _, _, ms = ogg_summary(ogg)
    if template is not None:
        h = bytearray(template[:HEADER_SIZE])
    else:
        h = bytearray(HEADER_SIZE)
        h[0] = 0x06
        h[4:14] = MARKER
        struct.pack_into("<I", h, 28, HEADER_SIZE)
        h[36] = 0x09
        h[44] = 0x10
        struct.pack_into("<H", h, 48, rate)
    struct.pack_into("<I", h, 24, ((ms + 275) * 44) & 0xFFFFFFFF)
    struct.pack_into("<I", h, 32, len(ogg) & 0xFFFFFFFF)
    return bytes(h)


def split_smp(data):
    if len(data) < HEADER_SIZE + 4 or data[HEADER_SIZE:HEADER_SIZE + 4] != b"OggS":
        raise ValueError("not a valid SMP (no Ogg stream at offset 160)")
    return data[:HEADER_SIZE], data[HEADER_SIZE:]


def write_file(path, data):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def find_files(root, ext):
    out = []
    for dp, _, fns in os.walk(root):
        out += [os.path.join(dp, f) for f in fns if f.lower().endswith(ext)]
    return sorted(out)


# ----------------------------------------------------------------- commands
def cmd_info(a):
    data = open(a.file, "rb").read()
    hdr, ogg = split_smp(data)
    u32 = lambda o: struct.unpack_from("<I", hdr, o)[0]
    print("file          :", a.file)
    print("total size    :", len(data))
    print("magic u32 @0  :", u32(0))
    print("marker @4     :", hdr[4:14].rstrip(b"\0").decode("latin1"))
    print("timing @24    : %d (-> %.1f ms)" % (u32(24), u32(24) / 44 - 275))
    print("hdr size @28  :", u32(28))
    print("ogg size @32  : %d (actual %d)" % (u32(32), len(ogg)))
    print("byte @36      :", hdr[36])
    print("byte @44      : 0x%02x" % hdr[44])
    print("rate @48      :", struct.unpack_from("<H", hdr, 48)[0])
    try:
        ch, rate, total, ms = ogg_summary(ogg)
        print("ogg channels  :", ch)
        print("ogg rate      :", rate)
        print("ogg samples   :", total)
        print("ogg duration  : %d ms" % ms)
    except ValueError as e:
        print("ogg           : unreadable (%s)" % e)
    extra = [i for i in range(HEADER_SIZE) if hdr[i] and i not in
             (0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36, 44, 48, 49)]
    if extra:
        print("non-zero bytes outside known fields:", ", ".join(map(str, extra)))
    return 0


def verify_one(path):
    try:
        data = open(path, "rb").read()
        hdr, ogg = split_smp(data)
        diff = [i for i, (x, y) in enumerate(zip(hdr, build_header(ogg))) if x != y]
        return (not diff), ("header differs at bytes %s" % ",".join(map(str, diff))) if diff else ""
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, e)


def cmd_verify(a):
    bad = 0
    for p in a.files:
        ok, err = verify_one(p)
        print("%-4s %s %s" % ("OK" if ok else "FAIL", p, err))
        bad += not ok
    return 2 if bad else 0


def cmd_roundtrip(a):
    files = find_files(a.dir, ".smp")
    bad = 0
    for p in files:
        ok, err = verify_one(p)
        if not ok:
            bad += 1
            print("FAIL", p, err)
        elif a.verbose:
            print("OK  ", p)
    print("%d/%d byte-identical" % (len(files) - bad, len(files)))
    return 2 if bad else 0


def cmd_to_ogg(a):
    try:
        _, ogg = split_smp(open(a.input, "rb").read())
    except (ValueError, OSError) as e:
        print('* ERROR: "%s": %s' % (a.input, e), file=sys.stderr)
        return 3
    out = a.output or os.path.splitext(a.input)[0] + ".ogg"
    write_file(out, ogg)
    if not a.quiet:
        print("Conversion complete:", out)
    return 0


def cmd_from_ogg(a):
    try:
        ogg = open(a.input, "rb").read()
        if ogg[:4] != b"OggS":
            raise ValueError("not a valid OGG")
        tpl = open(a.template, "rb").read(HEADER_SIZE) if a.template else None
        hdr = build_header(ogg, tpl, a.rate)
    except (ValueError, OSError) as e:
        print('* ERROR: "%s": %s' % (a.input, e), file=sys.stderr)
        return 3
    out = a.output or os.path.splitext(a.input)[0] + ".smp"
    write_file(out, hdr + ogg)
    if not a.quiet:
        print("Conversion complete:", out)
    return 0


def build_parser():
    ap = argparse.ArgumentParser(
        description="Inspect, verify and convert GBTVG .smp files <-> Ogg Vorbis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("SMP header")[0].split("\n", 3)[3])
    sub = ap.add_subparsers(dest="cmd", metavar="<command>")
    sub.required = True

    p = sub.add_parser("info", help="header + stream summary")
    p.add_argument("file"); p.set_defaults(fn=cmd_info)

    p = sub.add_parser("verify", help="rebuild header from the Ogg payload, must be byte-identical")
    p.add_argument("files", nargs="+"); p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("roundtrip-test", help="verify every .smp under dir")
    p.add_argument("dir"); p.add_argument("-v", "--verbose", action="store_true", help="also print passing files")
    p.set_defaults(fn=cmd_roundtrip)

    p = sub.add_parser("to-ogg", help="strip the 160-byte header")
    p.add_argument("input"); p.add_argument("output", nargs="?", help="default: input with .ogg extension")
    p.add_argument("-q", "--quiet", action="store_true"); p.set_defaults(fn=cmd_to_ogg)

    p = sub.add_parser("from-ogg", help="build an .smp from an Ogg Vorbis file")
    p.add_argument("input"); p.add_argument("output", nargs="?", help="default: input with .smp extension")
    p.add_argument("--template", metavar="OLD.smp",
                   help="reuse this file's header and only update the timing and size fields (keeps unknown bytes)")
    p.add_argument("--rate", type=int, default=DEFAULT_RATE,
                   help="u16 written at byte 48 (default 44100, as ogg2smp; ignored with --template)")
    p.add_argument("-q", "--quiet", action="store_true"); p.set_defaults(fn=cmd_from_ogg)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

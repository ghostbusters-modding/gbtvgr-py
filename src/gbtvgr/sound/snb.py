"""snb.py - single tool for compiled sound banks (.snb) and XML sound projects (.snd)

  info <file.snb>                  summary
  verify <file.snb ...>            parse->rebuild, must be byte-identical
  roundtrip-test <dir>             verify every .snb under dir
  snd-info <file.snd>              project summary
  snd-roundtrip-test <dir>         snb->snd->snb for every .snb under dir
  to-snd <in.snb> <out.snd>        decompile to XML project (--verify, see --help)
  from-snd <in.snd> <out.snb>      compile XML project (see the subcommand's --help)

SNB layout (version 13), all integers/floats little-endian, strings are
NUL-terminated Latin-1:

  u32   version                         (13)
  16B   header id / hash                (opaque, kept as hex)
  u32   reserved                        (0)
  u32   nChannels
          nChannels x { str name, f32 mixVolume, f32 defaultUserVolume }
  f32   priorities.fadeTime
  u32   nPriorities
          nPriorities x { str name, f32 value }
  u32   nReverbPresets (51)
          each { str name, i32 room, i32 roomHF, f32 decayTime,
                 f32 decayHFRatio, i32 reflections, f32 reflectionsDelay,
                 i32 reverb, f32 reverbDelay, f32 diffusion, f32 density,
                 f32 referenceHF }
  u32   nMixSnapshots (50)
          each { str name, u32 n, n x { u32 cueIndex, f32 volume } }
  u32   nCues
          each cue:
            str name
            u32 limiterType(0=fail) u32 limit u32 channel u32 priority
            u32 spatial
            f32 volume randomVolume pitch randomPitch fadeIn fadeOut
            f32 reverbFactor
            u32 autokill
            f32 l r c lfe lr rr ls rs          (settings-2d)
            f32 minDist maxDist dopplerFactor  (settings-3d)
            u32 nSelects
              each select: u32 3, u32 crossfade, f32 crossfadeStart, u32 n
                each sample: u32 1, str filename, u32 cacheControl(1=stream),
                             u32 looping, u32 0, u32 0
"""
import argparse
import os
import struct
import sys
import xml.etree.ElementTree as ET

SND_VERSION = 18
PLATFORMS = ["W32", "MAC", "PS2", "XBX", "GCB", "XB2", "PSP", "EDT", "PS3", "WII", "UNX"]
LIMITER = {0: "fail"}
LIMITER_REV = {"fail": 0}


# ----------------------------------------------------------------- reader
class R:
    def __init__(self, data):
        self.d, self.p = data, 0

    def u32(self):
        v = struct.unpack_from("<I", self.d, self.p)[0]; self.p += 4; return v

    def i32(self):
        v = struct.unpack_from("<i", self.d, self.p)[0]; self.p += 4; return v

    def f32(self):
        v = struct.unpack_from("<f", self.d, self.p)[0]; self.p += 4; return v

    def cs(self):
        e = self.d.index(b"\0", self.p)
        s = self.d[self.p:e].decode("latin1"); self.p = e + 1; return s

    def raw(self, n):
        b = self.d[self.p:self.p + n]; self.p += n; return b


def parse(data):
    r = R(data)
    m = {}
    m["version"] = r.u32()
    m["hdr_id"] = r.raw(16)
    m["reserved"] = r.u32()
    m["channels"] = [(r.cs(), r.f32(), r.f32()) for _ in range(r.u32())]
    m["fadeTime"] = r.f32()
    m["priorities"] = [(r.cs(), r.f32()) for _ in range(r.u32())]
    pres = []
    for _ in range(r.u32()):
        pres.append((r.cs(), r.i32(), r.i32(), r.f32(), r.f32(), r.i32(),
                     r.f32(), r.i32(), r.f32(), r.f32(), r.f32(), r.f32()))
    m["presets"] = pres
    snaps = []
    for _ in range(r.u32()):
        name = r.cs()
        snaps.append((name, [(r.u32(), r.f32()) for _ in range(r.u32())]))
    m["snapshots"] = snaps
    cues = []
    for _ in range(r.u32()):
        c = {"name": r.cs()}
        c["limiter"], c["limit"], c["channel"], c["priority"], c["spatial"] = (r.u32() for _ in range(5))
        (c["volume"], c["randomVolume"], c["pitch"], c["randomPitch"],
         c["fadeIn"], c["fadeOut"], c["reverbFactor"]) = (r.f32() for _ in range(7))
        c["autokill"] = r.u32()
        c["s2d"] = [r.f32() for _ in range(8)]
        c["s3d"] = [r.f32() for _ in range(3)]
        sels = []
        for _ in range(r.u32()):
            k = r.u32()
            if k != 3:
                raise ValueError("unexpected node kind %d at 0x%x" % (k, r.p - 4))
            xf, xfs, n = r.u32(), r.f32(), r.u32()
            samples = []
            for _ in range(n):
                k = r.u32()
                if k != 1:
                    raise ValueError("unexpected node kind %d at 0x%x" % (k, r.p - 4))
                samples.append((r.cs(), r.u32(), r.u32(), r.u32(), r.u32()))
            sels.append((xf, xfs, samples))
        c["selects"] = sels
        cues.append(c)
    m["cues"] = cues
    if r.p != len(data):
        raise ValueError("%d trailing bytes" % (len(data) - r.p))
    return m


# ----------------------------------------------------------------- writer (verification)
def encode(m):
    o = bytearray()
    u = lambda v: o.extend(struct.pack("<I", v))
    i = lambda v: o.extend(struct.pack("<i", v))
    f = lambda v: o.extend(struct.pack("<f", v))
    s = lambda v: o.extend(v.encode("latin1") + b"\0")
    u(m["version"]); o.extend(m["hdr_id"]); u(m["reserved"])
    u(len(m["channels"]))
    for n, a, b in m["channels"]: s(n); f(a); f(b)
    f(m["fadeTime"])
    u(len(m["priorities"]))
    for n, v in m["priorities"]: s(n); f(v)
    u(len(m["presets"]))
    for p in m["presets"]:
        s(p[0]); i(p[1]); i(p[2]); f(p[3]); f(p[4]); i(p[5]); f(p[6]); i(p[7])
        f(p[8]); f(p[9]); f(p[10]); f(p[11])
    u(len(m["snapshots"]))
    for n, ents in m["snapshots"]:
        s(n); u(len(ents))
        for idx, v in ents: u(idx); f(v)
    u(len(m["cues"]))
    for c in m["cues"]:
        s(c["name"])
        for k in ("limiter", "limit", "channel", "priority", "spatial"): u(c[k])
        for k in ("volume", "randomVolume", "pitch", "randomPitch", "fadeIn", "fadeOut", "reverbFactor"): f(c[k])
        u(c["autokill"])
        for v in c["s2d"]: f(v)
        for v in c["s3d"]: f(v)
        u(len(c["selects"]))
        for xf, xfs, samples in c["selects"]:
            u(3); u(xf); f(xfs); u(len(samples))
            for fn, a, b, c2, d2 in samples:
                u(1); s(fn); u(a); u(b); u(c2); u(d2)
    return bytes(o)


# ----------------------------------------------------------------- SND writer
def F(v):
    return "%f" % v


def A(v):
    return (str(v).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


YN = lambda v: "yes" if v else "no"


def to_snd(m):
    L = []
    w = L.append
    w('<root version="%d" projectName="">' % SND_VERSION)
    w("\t<channels>")
    for code in PLATFORMS:
        w('\t\t<platform platformCode="%s">' % code)
        for n, a, b in m["channels"]:
            w('\t\t\t<channel name="%s" mixVolume="%s" defaultUserVolume="%s" />' % (A(n), F(a), F(b)))
        w("\t\t</platform>")
    w("\t</channels>")
    if m["priorities"]:
        w('\t<priorities fadeTime="%s">' % F(m["fadeTime"]))
        for n, v in m["priorities"]:
            w('\t\t<priority name="%s" value="%s" />' % (A(n), F(v)))
        w("\t</priorities>")
    else:
        w('\t<priorities fadeTime="%s" />' % F(m["fadeTime"]))
    w("\t<reverbPresets>")
    for (n, room, rhf, dt, dhf, refl, rdel, rev, revdel, dif, den, ref) in m["presets"]:
        w('\t\t<reverbPreset name="%s" room="%d" roomHF="%d" decayTime="%s" decayHFRatio="%s" '
          'reflections="%s" reflectionsDelay="%s" reverb="%d" reverbDelay="%s" diffusion="%s" '
          'density="%s" referenceHF="%s" />'
          % (A(n), room, rhf, F(dt), F(dhf), F(refl), F(rdel), rev, F(revdel), F(dif), F(den), F(ref)))
    w("\t</reverbPresets>")
    w("\t<mixSnapshots>")
    cues = m["cues"]
    for n, ents in m["snapshots"]:
        if not ents:
            w('\t\t<mixSnapshot name="%s" />' % A(n))
        else:
            w('\t\t<mixSnapshot name="%s">' % A(n))
            for idx, v in ents:
                w('\t\t\t<cue name="%s" volume="%s" />' % (A(cues[idx]["name"]), F(v)))
            w("\t\t</mixSnapshot>")
    w("\t</mixSnapshots>")
    groups = set()
    for c in cues:
        parts = c["name"].split("/")[:-1]
        for k in range(1, len(parts) + 1):
            groups.add("/".join(parts[:k]))
    w("\t<groups>")
    for g in sorted(groups):
        w('\t\t<group name="%s" />' % A(g))
    w("\t</groups>")
    w("\t<cues>")
    for c in cues:
        lim = LIMITER.get(c["limiter"], "unknown%d" % c["limiter"])
        w('\t\t<cue name="%s" limiterType="%s" limit="%d" channel="%d" priority="%d" spatial="%s" '
          'volume="%s" randomVolume="%s" pitch="%s" randomPitchLowRange="%s" randomPitchHighRange="%s" '
          'pitchParameter="none" pitchParameterMin="0.000000" pitchParameterMax="1.000000" '
          'pitchValueMin="1.000000" pitchValueMax="1.000000" fadeIn="%s" fadeOut="%s" '
          'reverbFactor="%s" autokill="%s">'
          % (A(c["name"]), lim, c["limit"], c["channel"], c["priority"], YN(c["spatial"]),
             F(c["volume"]), F(c["randomVolume"]), F(c["pitch"]), F(c["randomPitch"]), F(c["randomPitch"]),
             F(c["fadeIn"]), F(c["fadeOut"]), F(c["reverbFactor"]), YN(c["autokill"])))
        l, r_, ce, lfe, lr, rr, ls, rs = c["s2d"]
        w('\t\t\t<settings-2d l="%s" r="%s" c="%s" lfe="%s" lr="%s" rr="%s" ls="%s" rs="%s" />'
          % tuple(F(x) for x in (l, r_, ce, lfe, lr, rr, ls, rs)))
        w('\t\t\t<settings-3d minDist="%s" maxDist="%s" dopplerFactor="%s" />' % tuple(F(x) for x in c["s3d"]))
        base = c["name"].split("/")[-1]
        for k, (xf, xfs, samples) in enumerate(c["selects"]):
            sname = base if k == 0 else "%s_%d" % (base, k)
            w('\t\t\t<sound type="select" name="%s" selectType="random" crossfade="%s" crossfadeStart="%s">'
              % (A(sname), YN(xf), F(xfs)))
            for fn, stream, loop, _r1, _r2 in samples:
                stem = fn.rsplit(".", 1)[0] if "." in fn else fn
                w('\t\t\t\t<sound type="sample" name="%s" filename="%s" cacheControl="%s" looping="%s" />'
                  % (A(stem), A(fn), "stream" if stream else "static", YN(loop)))
            w("\t\t\t</sound>")
        w("\t\t</cue>")
    w("\t</cues>")
    w("</root>")
    return "\r\n".join(L) + "\r\n"



# ----------------------------------------------------------------- SND reader (compile)
def yn(v):
    return 1 if v == "yes" else 0


def from_snd(path, platform="W32", hdr_id=b"\0" * 16):
    root = ET.parse(path).getroot()
    f32 = lambda s: struct.unpack("<f", struct.pack("<f", float(s)))[0]
    m = {"version": 13, "hdr_id": hdr_id, "reserved": 0}

    plat = None
    for p in root.find("channels").findall("platform"):
        if p.get("platformCode") == platform:
            plat = p
    if plat is None:
        raise SystemExit("platform %s not found in .snd" % platform)
    m["channels"] = [(c.get("name"), f32(c.get("mixVolume")), f32(c.get("defaultUserVolume")))
                     for c in plat.findall("channel")]

    pr = root.find("priorities")
    m["fadeTime"] = f32(pr.get("fadeTime"))
    m["priorities"] = [(p.get("name"), f32(p.get("value"))) for p in pr.findall("priority")]

    m["presets"] = []
    for r in root.find("reverbPresets").findall("reverbPreset"):
        g = r.get
        m["presets"].append((
            g("name"), int(float(g("room"))), int(float(g("roomHF"))), f32(g("decayTime")),
            f32(g("decayHFRatio")), int(float(g("reflections"))), f32(g("reflectionsDelay")),
            int(float(g("reverb"))), f32(g("reverbDelay")), f32(g("diffusion")),
            f32(g("density")), f32(g("referenceHF"))))

    cues = []
    for c in root.find("cues").findall("cue"):
        g = c.get
        lim = g("limiterType")
        if lim not in LIMITER_REV:
            raise SystemExit("cue %s: unknown limiterType %r (only 'fail' is known)" % (g("name"), lim))
        s2 = c.find("settings-2d").attrib
        s3 = c.find("settings-3d").attrib
        # SNB stores one random-pitch value; SND has low/high ranges.
        lo, hi = float(g("randomPitchLowRange")), float(g("randomPitchHighRange"))
        if lo != hi:
            print("warning: cue %s has asymmetric random pitch (%s/%s); using the larger"
                  % (g("name"), lo, hi), file=sys.stderr)
        cue = {
            "name": g("name"), "limiter": LIMITER_REV[lim], "limit": int(g("limit")),
            "channel": int(g("channel")), "priority": int(g("priority")),
            "spatial": yn(g("spatial")), "volume": f32(g("volume")),
            "randomVolume": f32(g("randomVolume")), "pitch": f32(g("pitch")),
            "randomPitch": f32(max(lo, hi)), "fadeIn": f32(g("fadeIn")),
            "fadeOut": f32(g("fadeOut")), "reverbFactor": f32(g("reverbFactor")),
            "autokill": yn(g("autokill")),
            "s2d": [f32(s2[k]) for k in ("l", "r", "c", "lfe", "lr", "rr", "ls", "rs")],
            "s3d": [f32(s3[k]) for k in ("minDist", "maxDist", "dopplerFactor")],
            "selects": [],
        }
        for sel in c.findall("sound"):
            if sel.get("type") != "select":
                raise SystemExit("cue %s: top-level sound is not a select" % cue["name"])
            samples = []
            for s in sel.findall("sound"):
                if s.get("type") != "sample":
                    raise SystemExit("cue %s: nested selects are not supported" % cue["name"])
                samples.append((s.get("filename"),
                                1 if s.get("cacheControl") == "stream" else 0,
                                yn(s.get("looping")), 0, 0))
            cue["selects"].append((yn(sel.get("crossfade")), f32(sel.get("crossfadeStart")), samples))
        cues.append(cue)
    m["cues"] = cues

    index = {}
    for i, c in enumerate(cues):
        index.setdefault(c["name"], i)
    m["snapshots"] = []
    for s in root.find("mixSnapshots").findall("mixSnapshot"):
        ents = []
        for e in s.findall("cue"):
            if e.get("name") not in index:
                raise SystemExit("snapshot %s references unknown cue %s" % (s.get("name"), e.get("name")))
            ents.append((index[e.get("name")], f32(e.get("volume"))))
        m["snapshots"].append((s.get("name"), ents))
    return m



# ----------------------------------------------------------------- CLI
def find_files(root, ext):
    out = []
    for dp, _, fns in os.walk(root):
        out += [os.path.join(dp, f) for f in fns if f.lower().endswith(ext)]
    return sorted(out)


def n_samples(m):
    return sum(len(s[2]) for c in m["cues"] for s in c["selects"])


def cmd_info(a):
    m = parse(open(a.file, "rb").read())
    groups = {"/".join(c["name"].split("/")[:k]) for c in m["cues"]
              for k in range(1, len(c["name"].split("/")))}
    print("file         :", a.file)
    print("version      :", m["version"])
    print("header id    :", m["hdr_id"].hex())
    print("channels     : %d" % len(m["channels"]))
    print("priorities   : %d (fadeTime %g)" % (len(m["priorities"]), m["fadeTime"]))
    print("reverb preset: %d" % len(m["presets"]))
    print("mix snapshots: %d" % len(m["snapshots"]))
    print("groups       : %d" % len(groups))
    print("cues         : %d" % len(m["cues"]))
    print("samples      : %d (%d streamed)" % (n_samples(m), sum(
        1 for c in m["cues"] for s in c["selects"] for x in s[2] if x[1])))
    if a.verbose:
        print("\nchannels:")
        for i, (n, mv, uv) in enumerate(m["channels"]):
            print("  [%d] %-24s mix=%g user=%g" % (i, n, mv, uv))
        print("priorities:")
        for n, v in m["priorities"]:
            print("  %-24s %g" % (n, v))
        print("cues:")
        for i, c in enumerate(m["cues"]):
            print("  [%d] %-40s ch=%d prio=%d vol=%g selects=%d samples=%d" % (
                i, c["name"], c["channel"], c["priority"], c["volume"],
                len(c["selects"]), sum(len(s[2]) for s in c["selects"])))
    return 0


def verify_one(path):
    data = open(path, "rb").read()
    try:
        return encode(parse(data)) == data, ""
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
    files = find_files(a.dir, ".snb")
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


def cmd_snd_info(a):
    root = ET.parse(a.file).getroot()
    cues = root.find("cues").findall("cue")
    plats = [p.get("platformCode") for p in root.find("channels").findall("platform")]
    print("file         :", a.file)
    print("version      :", root.get("version"))
    print("project      :", root.get("projectName"))
    print("platforms    :", " ".join(plats))
    print("groups       :", len(root.find("groups").findall("group")))
    print("reverb preset:", len(root.find("reverbPresets").findall("reverbPreset")))
    print("mix snapshots:", len(root.find("mixSnapshots").findall("mixSnapshot")))
    print("cues         :", len(cues))
    print("samples      :", sum(len(s.findall("sound")) for c in cues for s in c.findall("sound")))
    return 0


def cmd_snd_roundtrip(a):
    files = find_files(a.dir, ".snb")
    bad = 0
    for p in files:
        try:
            data = open(p, "rb").read()
            m = parse(data)
            tmp = p + ".__rt.snd"
            try:
                open(tmp, "w", encoding="latin1", newline="").write(to_snd(m))
                out = encode(from_snd(tmp, "W32", m["hdr_id"]))
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            ok = out == data
            err = "" if ok else "differs after snb->snd->snb"
        except BaseException as e:
            ok, err = False, "%s: %s" % (type(e).__name__, e)
        if not ok:
            bad += 1
            print("FAIL", p, err)
        elif a.verbose:
            print("OK  ", p)
    print("%d/%d byte-identical via .snd" % (len(files) - bad, len(files)))
    return 2 if bad else 0


def cmd_to_snd(a):
    data = open(a.input, "rb").read()
    m = parse(data)
    if a.verify:
        ok = encode(m) == data
        print("re-encode byte-identical:", ok)
        if not ok:
            return 2
    open(a.output, "w", encoding="latin1", newline="").write(to_snd(m))
    print("wrote %s: %d cues, %d samples" % (a.output, len(m["cues"]), n_samples(m)))
    return 0


def cmd_from_snd(a):
    hdr = b"\0" * 16
    if a.template:
        hdr = parse(open(a.template, "rb").read())["hdr_id"]
    m = from_snd(a.input, a.platform, hdr)
    out = encode(m)
    open(a.output, "wb").write(out)
    print("wrote %s: %d bytes, %d cues, %d samples" % (a.output, len(out), len(m["cues"]), n_samples(m)))
    if a.compare:
        ref = open(a.compare, "rb").read()
        same = out == ref
        print("byte-identical to %s: %s" % (a.compare, same))
        return 0 if same else 2
    return 0


def build_parser():
    ap = argparse.ArgumentParser(
        description="Inspect, verify and convert compiled sound banks (.snb) and XML sound projects (.snd).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("SNB layout")[0].split("snb.py - ")[1].split("\n", 1)[1])
    sub = ap.add_subparsers(dest="cmd", metavar="<command>")
    sub.required = True

    p = sub.add_parser("info", help="summary of an .snb")
    p.add_argument("file"); p.add_argument("-v", "--verbose", action="store_true", help="list channels, priorities, cues")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("verify", help="parse->rebuild, must be byte-identical")
    p.add_argument("files", nargs="+"); p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("roundtrip-test", help="verify every .snb under dir")
    p.add_argument("dir"); p.add_argument("-v", "--verbose", action="store_true", help="also print passing files")
    p.set_defaults(fn=cmd_roundtrip)

    p = sub.add_parser("snd-info", help="summary of an .snd project")
    p.add_argument("file"); p.set_defaults(fn=cmd_snd_info)

    p = sub.add_parser("snd-roundtrip-test", help="snb->snd->snb for every .snb under dir")
    p.add_argument("dir"); p.add_argument("-v", "--verbose", action="store_true", help="also print passing files")
    p.set_defaults(fn=cmd_snd_roundtrip)

    p = sub.add_parser("to-snd", help="decompile .snb to the XML .snd project")
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--verify", action="store_true", help="first check parse->re-encode is byte-identical to the input")
    p.set_defaults(fn=cmd_to_snd)

    p = sub.add_parser("from-snd", help="compile an .snd project into an .snb")
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--template", metavar="OLD.snb",
                   help="copy the 16-byte header id from an existing .snb (the .snd does not store it; default: zeros)")
    p.add_argument("--platform", default="W32", help="which <platform> channel table to use (default W32)")
    p.add_argument("--compare", metavar="REF.snb", help="report whether output is byte-identical to REF.snb")
    p.set_defaults(fn=cmd_from_snd)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

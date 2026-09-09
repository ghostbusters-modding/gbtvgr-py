#!/usr/bin/env python3
""".tex textures: DXT/BC, cubemaps, mipmaps.

Commands:
  info <file.tex ...>              header summary
  to-dds <file.tex> <out.dds>      decode to DDS
  from-dds <in.dds> <file.tex>     encode from DDS
  to-png <file.tex> <out.png>      decode to PNG (needs Pillow)
  roundtrip-test <dir>             verify every .tex under dir
"""
import argparse, os, struct, sys, zlib

HEADER_LEN = 0x34  # 52

# fmt id -> (name, kind, bpp-or-blockbytes, cube)
FORMATS = {
    3:  dict(name='RAW_BGRA8',      kind='raw', unit=4,  cube=False),
    24: dict(name='RAW_BGRA8_CUBE', kind='raw', unit=4,  cube=True),
    43: dict(name='BC1',            kind='bc',  unit=8,  cube=False),
    47: dict(name='RAW_RG8',        kind='raw', unit=2,  cube=False),
    50: dict(name='BC3',            kind='bc',  unit=16, cube=False),
}

DEFAULT_D = 0x0DA3853C


class TexError(Exception):
    pass


def _read(path):
    with open(path, 'rb') as fh:
        return fh.read()


def mip_dims(w, h):
    """[(w0,h0), (w1,h1), ...] down to (1,1), standard box-mip halving."""
    levels = []
    cw, ch = w, h
    while True:
        levels.append((cw, ch))
        if cw == 1 and ch == 1:
            break
        cw = max(1, cw // 2)
        ch = max(1, ch // 2)
    return levels


def level_size(fmt_info, w, h):
    if fmt_info['kind'] == 'bc':
        bx = max(1, (w + 3) // 4)
        by = max(1, (h + 3) // 4)
        return bx * by * fmt_info['unit']
    return w * h * fmt_info['unit']


def faces(fmt_info):
    return 6 if fmt_info['cube'] else 1


def total_size(fmt_info, w, h, mipcount):
    levels = mip_dims(w, h)[:mipcount]
    return sum(level_size(fmt_info, lw, lh) for lw, lh in levels) * faces(fmt_info)


def guess_mipcount(fmt_info, w, h, payload_len):
    """Find the mip count whose cumulative size exactly equals payload_len."""
    levels = mip_dims(w, h)
    total = 0
    nf = faces(fmt_info)
    for i, (lw, lh) in enumerate(levels):
        total += level_size(fmt_info, lw, lh) * nf
        if total == payload_len:
            return i + 1
    return None


def parse_header(data):
    if len(data) < HEADER_LEN:
        raise TexError("file too short for header: %d bytes" % len(data))
    ver, = struct.unpack_from('<I', data, 0)
    hashb = data[0x04:0x14]
    zero1, = struct.unpack_from('<I', data, 0x14)
    fmt, w, h = struct.unpack_from('<3I', data, 0x18)
    A, B, C, D = struct.unpack_from('<4I', data, 0x24)
    payload = data[HEADER_LEN:]
    if fmt not in FORMATS:
        raise TexError("unknown fmt id %d" % fmt)
    fmt_info = FORMATS[fmt]
    mc = guess_mipcount(fmt_info, w, h, len(payload))
    return dict(ver=ver, hash=hashb, zero1=zero1, fmt=fmt, w=w, h=h,
                A=A, B=B, mipcount=mc, C=C, D=D, payload=payload,
                raw_header=data[:HEADER_LEN], fmt_info=fmt_info)


def build_header(ver, hashb, zero1, fmt, w, h, A, mipcount, C, D):
    if len(hashb) != 16:
        raise TexError("hash must be 16 bytes")
    return struct.pack('<I16sIIIIIIII', ver, hashb, zero1, fmt, w, h,
                        A, mipcount - 1, C, D)


# --------------------------------------------------------------------------
# DDS
# --------------------------------------------------------------------------
DDS_MAGIC = b'DDS '
DDPF_ALPHAPIXELS = 0x1
DDPF_FOURCC = 0x4
DDPF_RGB = 0x40
DDSD_CAPS = 0x1
DDSD_HEIGHT = 0x2
DDSD_WIDTH = 0x4
DDSD_PITCH = 0x8
DDSD_PIXELFORMAT = 0x1000
DDSD_MIPMAPCOUNT = 0x20000
DDSD_LINEARSIZE = 0x80000
DDSCAPS_COMPLEX = 0x8
DDSCAPS_MIPMAP = 0x400000
DDSCAPS_TEXTURE = 0x1000
DDSCAPS2_CUBEMAP = 0x200
DDSCAPS2_CUBEMAP_ALLFACES = 0xFC00
DXGI_FORMAT_R8G8_UNORM = 49

# fmt id -> dds fourcc ('DXT1'/'DXT5') or None (raw -> uses masks / DX10)
FOURCC = {43: b'DXT1', 50: b'DXT5'}


def to_dds(tex_bytes):
    """Wrap a .tex file's pixel payload in a standard DDS container."""
    h = parse_header(tex_bytes)
    fmt, w, ht, mc = h['fmt'], h['w'], h['h'], h['mipcount']
    if mc is None:
        raise TexError("payload size does not match any mip count for fmt %d %dx%d "
                        "(%d bytes)" % (fmt, w, ht, len(h['payload'])))
    fmt_info = h['fmt_info']
    cube = fmt_info['cube']

    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
    caps = DDSCAPS_TEXTURE
    if mc > 1:
        flags |= DDSD_MIPMAPCOUNT
        caps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
    caps2 = 0
    if cube:
        caps2 = DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_ALLFACES
        caps |= DDSCAPS_COMPLEX

    dx10_format = None
    if fmt in FOURCC:
        fourcc = FOURCC[fmt]
        flags |= DDSD_LINEARSIZE
        pitch_or_linsize = level_size(fmt_info, w, ht)
        pf = struct.pack('<2I4s5I', 32, DDPF_FOURCC, fourcc, 0, 0, 0, 0, 0)
    elif fmt in (3, 24):
        # raw BGRA8, D3D A8R8G8B8 mask convention (byte order B,G,R,A)
        flags |= DDSD_PITCH
        pitch_or_linsize = w * 4
        pf = struct.pack('<2I4s5I', 32, DDPF_RGB | DDPF_ALPHAPIXELS, b'\x00\x00\x00\x00',
                          32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    elif fmt == 47:
        # raw 2-channel: no classic DX9 FourCC, use a DX10 header
        flags |= DDSD_PITCH
        pitch_or_linsize = w * 2
        pf = struct.pack('<2I4s5I', 32, DDPF_FOURCC, b'DX10', 0, 0, 0, 0, 0)
        dx10_format = DXGI_FORMAT_R8G8_UNORM
    else:
        raise TexError("no DDS mapping for fmt %d" % fmt)

    hdr = struct.pack('<7I44s', 124, flags, ht, w, pitch_or_linsize, 0, mc, b'\x00' * 44)
    hdr += pf
    hdr += struct.pack('<4I', caps, caps2, 0, 0)
    hdr += b'\x00' * 4
    out = DDS_MAGIC + hdr
    if dx10_format is not None:
        rt = 4 if cube else 3  # D3D10_RESOURCE_DIMENSION_TEXTURE2D(3) / cube uses same but misc flag
        misc = 4 if cube else 0
        out += struct.pack('<5I', dx10_format, 3, misc, 6 if cube else 1, 0)
    out += h['payload']
    return out


def parse_dds(dds_bytes):
    if dds_bytes[:4] != DDS_MAGIC:
        raise TexError("not a DDS file")
    (size, flags, height, width, pitch, depth, mipcount) = struct.unpack_from('<7I', dds_bytes, 4)
    off = 4 + 4 * 7 + 44  # magic + 7 header ints + reserved1[11]
    pf_size, pf_flags, fourcc, rgbbits, rmask, gmask, bmask, amask = struct.unpack_from('<2I4s5I', dds_bytes, off)
    off += 32
    caps, caps2, caps3, caps4 = struct.unpack_from('<4I', dds_bytes, off)
    off += 16
    off += 4  # reserved2
    mipcount = mipcount or 1
    cube = bool(caps2 & DDSCAPS2_CUBEMAP)

    dx10_format = None
    if pf_flags & DDPF_FOURCC and fourcc == b'DX10':
        dx10_format, resdim, misc, arraysize, misc2 = struct.unpack_from('<5I', dds_bytes, off)
        off += 20

    fmt = None
    if pf_flags & DDPF_FOURCC:
        if fourcc == b'DXT1':
            fmt = 43
        elif fourcc == b'DXT5':
            fmt = 50
        elif fourcc == b'DX10' and dx10_format == DXGI_FORMAT_R8G8_UNORM:
            fmt = 47
    elif pf_flags & DDPF_RGB:
        if rgbbits == 32 and rmask == 0x00FF0000 and gmask == 0x0000FF00 and bmask == 0x000000FF:
            fmt = 24 if cube else 3

    if fmt is None:
        raise TexError("unrecognized DDS pixel format (fourcc=%r rgbbits=%s)" %
                        (fourcc, rgbbits if pf_flags & DDPF_RGB else None))

    payload = dds_bytes[off:]
    return dict(fmt=fmt, w=width, h=height, mipcount=mipcount, cube=cube, payload=payload)


def from_dds(dds_bytes, template=None):
    """`template` is an existing .tex supplying the header fields a DDS cannot.
    The hash algorithm is unknown, so without one the hash will not match."""
    d = parse_dds(dds_bytes)
    fmt_info = FORMATS[d['fmt']]
    expect = total_size(fmt_info, d['w'], d['h'], d['mipcount'])
    if len(d['payload']) != expect:
        raise TexError("DDS payload is %d bytes, expected %d for fmt %d %dx%d mips=%d" %
                        (len(d['payload']), expect, d['fmt'], d['w'], d['h'], d['mipcount']))

    if template is not None:
        th = parse_header(template)
        ver, hashb, zero1, A, C, D = th['ver'], th['hash'], th['zero1'], th['A'], th['C'], th['D']
    else:
        ver, hashb, zero1, A, C, D = 7, b'\x00' * 16, 0, 0, 0, DEFAULT_D

    header = build_header(ver, hashb, zero1, d['fmt'], d['w'], d['h'], A, d['mipcount'], C, D)
    return header + d['payload']


# --------------------------------------------------------------------------
# PNG preview. PIL does the BC1/BC3 decode; raw formats are unpacked by hand,
# since PIL has no opener for a raw D3D BGRA buffer that beats doing it here.
# --------------------------------------------------------------------------
def to_png(tex_bytes, out_path, face=0):
    h = parse_header(tex_bytes)
    fmt, w, ht = h['fmt'], h['w'], h['h']
    fmt_info = h['fmt_info']
    if h['mipcount'] is None:
        raise TexError("cannot determine mip count; payload size doesn't match fmt %d" % fmt)

    from PIL import Image
    import io

    if fmt_info['kind'] == 'bc':
        # top mip only, single "1-mip" DDS so PIL decodes just the base level
        top = h['payload'][:level_size(fmt_info, w, ht)]
        mini_tex = build_header(h['ver'], h['hash'], h['zero1'], fmt, w, ht, h['A'], 1, h['C'], h['D']) + top
        dds = to_dds(mini_tex)
        im = Image.open(io.BytesIO(dds)).convert('RGBA')
        im.save(out_path)
        return

    if fmt in (3, 24):
        # raw BGRA8; for cubemaps, preview one face (default +X, face=0)
        face_bytes = level_size(fmt_info, w, ht)
        start = face * face_bytes if fmt == 24 else 0
        buf = h['payload'][start:start + w * ht * 4]
        im = Image.frombytes('RGBA', (w, ht), buf, 'raw', 'BGRA')
        im.save(out_path)
        return

    if fmt == 47:
        # 2-channel tangent normal XY -> reconstruct Z for a nicer preview
        buf = h['payload'][:w * ht * 2]
        try:
            import numpy as np
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(ht, w, 2).astype(np.float32)
            nx = arr[:, :, 0] / 127.5 - 1.0
            ny = arr[:, :, 1] / 127.5 - 1.0
            nz = (1.0 - nx * nx - ny * ny).clip(0, 1) ** 0.5
            rgb = np.zeros((ht, w, 3), dtype=np.uint8)
            rgb[:, :, 0] = arr[:, :, 0]
            rgb[:, :, 1] = arr[:, :, 1]
            rgb[:, :, 2] = ((nz + 1.0) * 0.5 * 255).astype(np.uint8)
            Image.fromarray(rgb, 'RGB').save(out_path)
        except ImportError:
            im = Image.frombytes('L', (w, ht), bytes(buf[0::2]))
            im.save(out_path)
        return

    raise TexError("no PNG preview path for fmt %d" % fmt)


# --------------------------------------------------------------------------
# .tex -> RGBA, pure python: the Blender add-on has no Pillow to lean on
# --------------------------------------------------------------------------
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
    h = parse_header(data)
    fi = h['fmt_info']
    if fi['cube']:
        raise ValueError('cubemap (fmt %d) has no flat preview' % h['fmt'])
    levels = mip_dims(h['w'], h['h'])[:max(1, h['mipcount'] or 1)]
    off = 0
    pick = None
    for lw, lh in levels:
        size = level_size(fi, lw, lh)
        if pick is None and max(lw, lh) <= max_size:
            pick = (lw, lh, off, size)
        off += size
    if pick is None:                                # every mip is too big
        lw, lh = levels[-1]
        off -= level_size(fi, lw, lh)
        pick = (lw, lh, off, level_size(fi, lw, lh))
    lw, lh, off, size = pick
    blob = h['payload'][off:off + size]
    if len(blob) < size:
        raise ValueError('truncated mip (%d of %d bytes)' % (len(blob), size))
    if fi['kind'] == 'bc':
        return lw, lh, decode_bc(blob, lw, lh, h['fmt'] == 50)
    return lw, lh, decode_raw(blob, lw, lh, fi['unit'])


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def cmd_info(a):
    data = _read(a.infile)
    h = parse_header(data)
    fi = h['fmt_info']
    print("file       %s (%d bytes)" % (a.infile, len(data)))
    print("ver        %d" % h['ver'])
    print("hash       %s" % h['hash'].hex())
    print("zero1      %d" % h['zero1'])
    print("fmt        %d (%s)" % (h['fmt'], fi['name']))
    print("size       %dx%d%s" % (h['w'], h['h'], "  (cubemap x6)" if fi['cube'] else ""))
    print("A          %d" % h['A'])
    print("B (mip-1)  %d" % h['B'])
    print("mipcount   %s" % (h['mipcount'] if h['mipcount'] is not None else "UNRESOLVED (size mismatch)"))
    print("C          %d" % h['C'])
    print("D          0x%08X%s" % (h['D'], "  (default)" if h['D'] == DEFAULT_D else "  (non-default!)"))
    print("payload    %d bytes" % len(h['payload']))


def cmd_to_dds(a):
    data = _read(a.infile)
    out = to_dds(data)
    with open(a.outfile, 'wb') as fh:
        fh.write(out)
    print("wrote %s (%d bytes)" % (a.outfile, len(out)))


def cmd_from_dds(a):
    data = _read(a.infile)
    template = _read(a.like) if a.like else None
    out = from_dds(data, template)
    with open(a.outfile, 'wb') as fh:
        fh.write(out)
    print("wrote %s (%d bytes)%s" % (a.outfile, len(out), "" if a.like else "  (no --like: header defaults used, hash NOT authentic)"))


def cmd_to_png(a):
    data = _read(a.infile)
    to_png(data, a.outfile, face=a.face)
    print("wrote %s" % a.outfile)


def cmd_roundtrip_test(a):
    from .archive.pod import Pod
    import collections

    by_fmt = collections.Counter()
    ok_fmt = collections.Counter()
    fails = []
    total = 0
    for podpath in a.pods:
        with Pod(podpath) as p:
            for e in p.entries:
                if not e['name'].lower().endswith('.tex'):
                    continue
                total += 1
                try:
                    data = p.read(e)
                    h = parse_header(data)
                    dds = to_dds(data)
                    rebuilt = from_dds(dds, template=data)
                except (OSError, ValueError, TexError, zlib.error, struct.error) as ex:
                    fails.append((podpath, e['name'], str(ex)))
                    continue
                fmt = h['fmt']
                by_fmt[fmt] += 1
                if rebuilt == data:
                    ok_fmt[fmt] += 1
                else:
                    fails.append((podpath, e['name'], "byte mismatch: %d vs %d bytes" %
                                  (len(rebuilt), len(data))))
    print("scanned %d .tex files across %d pod(s)" % (total, len(a.pods)))
    for fmt in sorted(by_fmt):
        n, ok = by_fmt[fmt], ok_fmt[fmt]
        name = FORMATS.get(fmt, {}).get('name', '?')
        print("  fmt %-3d %-16s %d/%d identical" % (fmt, name, ok, n))
    print("TOTAL %d/%d identical" % (sum(ok_fmt.values()), total))
    if fails:
        print("%d failures:" % len(fails))
        for f in fails[:30]:
            print("  ", f)
    return 0 if not fails and sum(ok_fmt.values()) == total else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('info'); p.add_argument('infile'); p.set_defaults(fn=cmd_info)

    p = sub.add_parser('to-dds'); p.add_argument('infile'); p.add_argument('outfile')
    p.set_defaults(fn=cmd_to_dds)

    p = sub.add_parser('from-dds'); p.add_argument('infile'); p.add_argument('outfile')
    p.add_argument('--like', help='template .tex to copy header fields (ver/hash/A/C/D) from')
    p.set_defaults(fn=cmd_from_dds)

    p = sub.add_parser('to-png'); p.add_argument('infile'); p.add_argument('outfile')
    p.add_argument('--face', type=int, default=0, help='cubemap face index 0-5 (fmt 24 only)')
    p.set_defaults(fn=cmd_to_png)

    p = sub.add_parser('roundtrip-test'); p.add_argument('pods', nargs='+')
    p.set_defaults(fn=cmd_roundtrip_test)

    a = ap.parse_args()
    rc = a.fn(a)
    sys.exit(rc or 0)


if __name__ == '__main__':
    main()

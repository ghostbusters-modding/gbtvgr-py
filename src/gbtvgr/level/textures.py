"""`.tex` <-> numpy RGBA. Decoding is vectorised; `gbtvgr.tex` stays the oracle
the tests compare against pixel for pixel."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import struct

import numpy as np

from gbtvgr import tex as gtex

HEADER_LEN = gtex.HEADER_LEN
DEFAULT_D = gtex.DEFAULT_D
FMT_BGRA8, FMT_CUBE, FMT_BC1, FMT_RG8, FMT_BC3 = 3, 24, 43, 47, 50


def _rgb565(v):
    r = (v >> 11) & 31
    g = (v >> 5) & 63
    b = v & 31
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)],
                    axis=-1).astype(np.int64)


def _bc_color_blocks(cblk, punch):
    """cblk: (n, 8) uint8 colour blocks -> (n, 16, 4) RGBA."""
    n = len(cblk)
    c0 = cblk[:, 0].astype(np.int64) | (cblk[:, 1].astype(np.int64) << 8)
    c1 = cblk[:, 2].astype(np.int64) | (cblk[:, 3].astype(np.int64) << 8)
    bits = (cblk[:, 4].astype(np.int64) | (cblk[:, 5].astype(np.int64) << 8)
            | (cblk[:, 6].astype(np.int64) << 16) | (cblk[:, 7].astype(np.int64) << 24))
    a = _rgb565(c0)
    b = _rgb565(c1)
    four = (c0 > c1) | (not punch)
    pal = np.zeros((n, 4, 4), dtype=np.int64)
    pal[:, 0, :3] = a
    pal[:, 1, :3] = b
    pal[:, 2, :3] = np.where(four[:, None], (2 * a + b) // 3, (a + b) // 2)
    pal[:, 3, :3] = np.where(four[:, None], (a + 2 * b) // 3, 0)
    pal[:, :, 3] = 255
    pal[:, 3, 3] = np.where(four, 255, 0)
    shifts = np.arange(16, dtype=np.int64) * 2
    idx = (bits[:, None] >> shifts[None, :]) & 3
    return np.take_along_axis(pal, idx[:, :, None].repeat(4, axis=2), axis=1)


def _bc_alpha_blocks(ablk):
    """ablk: (n, 8) uint8 alpha blocks -> (n, 16) alpha."""
    n = len(ablk)
    a0 = ablk[:, 0].astype(np.int64)
    a1 = ablk[:, 1].astype(np.int64)
    abits = np.zeros(n, dtype=np.int64)
    for k in range(6):
        abits |= ablk[:, 2 + k].astype(np.int64) << (8 * k)
    pal = np.zeros((n, 8), dtype=np.int64)
    pal[:, 0], pal[:, 1] = a0, a1
    gt = a0 > a1
    for i in range(1, 7):
        pal[:, i + 1] = np.where(gt, ((7 - i) * a0 + i * a1) // 7, 0)
    for i in range(1, 5):
        pal[:, i + 1] = np.where(gt, pal[:, i + 1], ((5 - i) * a0 + i * a1) // 5)
    pal[:, 6] = np.where(gt, pal[:, 6], 0)
    pal[:, 7] = np.where(gt, pal[:, 7], 255)
    shifts = np.arange(16, dtype=np.int64) * 3
    idx = (abits[:, None] >> shifts[None, :]) & 7
    return np.take_along_axis(pal, idx, axis=1)


def decode_bc(data, w, h, bc3):
    bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
    stride = 16 if bc3 else 8
    blk = np.frombuffer(data, dtype=np.uint8, count=bw * bh * stride).reshape(bw * bh, stride)
    if bc3:
        px = _bc_color_blocks(blk[:, 8:], False)
        px[:, :, 3] = _bc_alpha_blocks(blk[:, :8])
    else:
        px = _bc_color_blocks(blk, True)
    img = px.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return img[:h, :w].astype(np.uint8)


def decode_raw(data, w, h, unit):
    if unit == 4:
        img = np.frombuffer(data, dtype=np.uint8, count=w * h * 4).reshape(h, w, 4)
        return img[:, :, [2, 1, 0, 3]].copy()
    rg = np.frombuffer(data, dtype=np.uint8, count=w * h * 2).reshape(h, w, 2)
    nx = rg[:, :, 0].astype(np.float64) / 127.5 - 1.0
    ny = rg[:, :, 1].astype(np.float64) / 127.5 - 1.0
    nz = np.sqrt(np.maximum(0.0, 1.0 - nx * nx - ny * ny))
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[:, :, 0] = rg[:, :, 0]
    out[:, :, 1] = rg[:, :, 1]
    out[:, :, 2] = ((nz + 1.0) * 127.5).astype(np.int64).clip(0, 255)
    out[:, :, 3] = 255
    return out


def _pick_mip(hdr, max_size):
    fi = hdr['fmt_info']
    levels = gtex.mip_dims(hdr['w'], hdr['h'])[:max(1, hdr['mipcount'] or 1)]
    off, pick = 0, None
    for lw, lh in levels:
        size = gtex.level_size(fi, lw, lh)
        if pick is None and (max_size is None or max(lw, lh) <= max_size):
            pick = (lw, lh, off, size)
        off += size
    if pick is None:
        lw, lh = levels[-1]
        size = gtex.level_size(fi, lw, lh)
        pick = (lw, lh, off - size, size)
    return pick


def decode(blob, max_size=None):
    """.tex bytes -> (w, h, (h, w, 4) uint8 RGBA), the largest mip within max_size."""
    hdr = gtex.parse_header(blob)
    fi = hdr['fmt_info']
    if fi['cube']:
        raise ValueError('cubemap has no flat image; use decode_cube')
    lw, lh, off, size = _pick_mip(hdr, max_size)
    chunk = hdr['payload'][off:off + size]
    if len(chunk) < size:
        raise ValueError('truncated mip level')
    if fi['kind'] == 'bc':
        return lw, lh, decode_bc(chunk, lw, lh, hdr['fmt'] == FMT_BC3)
    return lw, lh, decode_raw(chunk, lw, lh, fi['unit'])


def decode_cube(blob):
    """fmt 24 -> six (h, w, 4) RGBA faces, top mip only."""
    hdr = gtex.parse_header(blob)
    if not hdr['fmt_info']['cube']:
        raise ValueError('not a cubemap')
    w, h = hdr['w'], hdr['h']
    per_face = gtex.total_size(dict(hdr['fmt_info'], cube=False), w, h,
                               max(1, hdr['mipcount'] or 1))
    faces = []
    for i in range(6):
        chunk = hdr['payload'][i * per_face:i * per_face + w * h * 4]
        faces.append(decode_raw(chunk, w, h, 4))
    return faces


def encode_raw(rgba, template=None, fmt=FMT_BGRA8):
    """(h, w, 4) RGBA, or six of them for a cube -> single-mip .tex bytes.
    The 16-byte hash is unresolved, so it is copied from a template when given."""
    if fmt == FMT_CUBE:
        faces = list(rgba)
        h, w = faces[0].shape[:2]
    else:
        faces = [rgba]
        h, w = rgba.shape[:2]
    payload = b''.join(np.ascontiguousarray(f[:, :, [2, 1, 0, 3]], dtype=np.uint8).tobytes()
                       for f in faces)
    if template is not None:
        th = gtex.parse_header(template)
        ver, hashb, zero1, A, C, D = th['ver'], th['hash'], th['zero1'], th['A'], th['C'], th['D']
    else:
        ver, hashb, zero1, A, C, D = 7, b'\0' * 16, 0, 0, 0, DEFAULT_D
    return gtex.build_header(ver, hashb, zero1, fmt, w, h, A, 1, C, D) + payload


def header_desc(blob):
    """(fmt, w, h) as an STexSlot wants them."""
    hdr = gtex.parse_header(blob)
    return hdr['fmt'], hdr['w'], hdr['h']


def to_gl(img):
    """RGBA rows top-down -> bottom-up bytes for glTexImage."""
    return np.ascontiguousarray(img[::-1]).tobytes()


def checker(w=64, h=64):
    y, x = np.mgrid[0:h, 0:w]
    c = (((x // 8) + (y // 8)) & 1).astype(np.uint8)
    img = np.empty((h, w, 4), dtype=np.uint8)
    img[:, :, :3] = (160 + 60 * c)[:, :, None]
    img[:, :, 3] = 255
    return img


def pack_header_fields(blob):
    """The header words the STexDesc needs, straight off a .tex blob."""
    fmt, w, h = struct.unpack_from('<3I', blob, 0x18)
    return fmt, w, h

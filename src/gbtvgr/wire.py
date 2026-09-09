"""Byte reader/writer and the primitives every engine record is built from."""
import io
import struct


class R:
    def __init__(self, d):
        self.d = d; self.o = 0
    def take(self, n):
        b = self.d[self.o:self.o+n]
        if len(b) != n: raise ValueError('EOF @%#x (+%d)' % (self.o, n))
        self.o += n; return b
    def u32(self): return struct.unpack('<I', self.take(4))[0]
    def i32(self): return struct.unpack('<i', self.take(4))[0]
    def u16(self): return struct.unpack('<H', self.take(2))[0]
    def f32(self): return struct.unpack('<f', self.take(4))[0]
    def cstr(self):
        e = self.d.index(b'\0', self.o)
        s = self.d[self.o:e]; self.o = e + 1; return s
    def align(self, n):
        pad = (-self.o) % n
        return self.take(pad)

class W:
    def __init__(self):
        self.b = io.BytesIO()
    def w(self, b): self.b.write(b)
    def u32(self, v): self.w(struct.pack('<I', v))
    def u16(self, v): self.w(struct.pack('<H', v))
    def f32(self, v): self.w(struct.pack('<f', v))
    def cstr(self, s): self.w(s + b'\0')
    def align(self, n, fill=b'\0'):
        pad = (-self.b.tell()) % n
        self.w(fill * pad)
    def data(self): return self.b.getvalue()

def want(got, expect, what):
    """Wire-format version gate. A bare assert would vanish under `python -O`."""
    if got != expect:
        raise ValueError('%s: expected %#x, got %#x' % (what, expect, got))
    return got

def read_deps(r):
    """Repeat { path + 16-byte hash } until an empty path, then align 4.
    The enclosing record's own hash was already read by the caller."""
    deps = []
    while r.d[r.o] != 0:
        name = r.cstr()
        deps.append((name, r.take(16)))
    r.o += 1
    pad = r.align(4)
    return deps, pad

def write_deps(w, deps):
    for name, h in deps:
        w.cstr(name); w.w(h)
    w.w(b'\0'); w.align(4)


def read_str(r):
    """NUL-terminated, then padded to a multiple of 4 -- counted from the string's
    own start, not from the file offset."""
    s = r.cstr()
    pad = r.take((-(len(s) + 1)) % 4)
    return s, pad

def write_str(w, sp):
    s, pad = sp
    w.cstr(s); w.w(pad)


def half_to_f32(h):
    s, e, f = (h >> 15) & 1, (h >> 10) & 0x1F, h & 0x3FF
    if e == 0:
        v = f * 2.0 ** -24
    elif e == 31:
        v = float('inf') if f == 0 else float('nan')
    else:
        v = (1 + f / 1024.0) * 2.0 ** (e - 15)
    return -v if s else v

def f32_to_half(x):
    """Truncating f32->f16 (matches the game's own data; sign-preserving)."""
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    s, e, f = (bits >> 31) << 15, (bits >> 23) & 0xFF, bits & 0x7FFFFF
    if e == 0: return s
    if e == 255: return s | 0x7C00 | (f >> 13)
    he = e - 127 + 15
    if he >= 31: return s | 0x7C00
    if he <= 0:
        if he < -10: return s
        return s | ((f | 0x800000) >> (1 - he) >> 13)
    return s | (he << 10) | (f >> 13)

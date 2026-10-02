"""Game audio: .smp sound files and .snb sound banks.

The synthetic checks build minimal files from nothing, so they run with no
game installation; the corpus checks verify the shipped files byte-identically
and skip when no corpus is given (see tests/conftest.py).
"""
import os
import struct

from _run import gbtvgr, ok

from gbtvgr.sound import smp, snb


def _ogg_page(granule, serial, seq, body, bos=False):
    flags = 0x02 if bos else 0x00
    nseg = (len(body) + 254) // 255 if body else 0
    segs = bytes([255] * (nseg - 1) + ([len(body) % 255 or 255] if body else []))
    head = struct.pack("<4sBBqIIIB", b"OggS", 0, flags, granule,
                       serial, seq, 0, nseg) + segs
    return head + body


def _fake_ogg(rate=44100, samples=44100):
    """A structurally valid Ogg Vorbis stream: one BOS page carrying the
    identification header, one audio page setting the granule position."""
    ident = (b"\x01vorbis" + struct.pack("<I", 0) + bytes([1])
             + struct.pack("<I", rate) + struct.pack("<III", 0, 0, 0)
             + bytes([0x11, 0x01]))
    return (_ogg_page(0, 1, 0, ident, bos=True)
            + _ogg_page(samples, 1, 1, b"\x00" * 10))


def _fake_smp():
    ogg = _fake_ogg()
    return smp.build_header(ogg) + ogg


def _fake_snb_model():
    cue = {
        "name": "sfx/hit", "limiter": 0, "limit": 1, "channel": 0,
        "priority": 0, "spatial": 0, "volume": 1.0, "randomVolume": 0.0,
        "pitch": 1.0, "randomPitch": 0.0, "fadeIn": 0.0, "fadeOut": 0.0,
        "reverbFactor": 0.0, "autokill": 0,
        "s2d": [1.0, 1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
        "s3d": [1.0, 100.0, 0.0],
        "selects": [(0, 0.0, [("hit.wav", 1, 0, 0, 0)])],
    }
    return {
        "version": 13, "hdr_id": b"\0" * 16, "reserved": 0,
        "channels": [("master", 1.0, 1.0)], "fadeTime": 0.5,
        "priorities": [("high", 1.0)],
        "presets": [("hall", -1000, -100, 1.49, 0.83, -2602, 0.007,
                     -200, 0.011, 1.0, 1.0, 5000.0)],
        "snapshots": [("default", [(0, 0.8)])],
        "cues": [cue],
    }


# ---------------------------------------------------------------- self-contained

def test_smp_header_roundtrip(tmp_path):
    data = _fake_smp()
    hdr, ogg = smp.split_smp(data)
    assert len(hdr) == smp.HEADER_SIZE
    assert ogg[:4] == b"OggS"
    ch, rate, total, ms = smp.ogg_summary(ogg)
    assert (ch, rate, total, ms) == (1, 44100, 44100, 1000)
    assert smp.build_header(ogg) == hdr  # verify_one's comparison, in-process

    p = tmp_path / "hit.smp"
    p.write_bytes(data)
    out = ok(gbtvgr("smp", "verify", str(p)))
    assert "OK" in out, out
    out = ok(gbtvgr("smp", "info", str(p)))
    assert "ogg duration  : 1000 ms" in out, out


def test_smp_ogg_both_ways(tmp_path):
    src = tmp_path / "hit.smp"
    src.write_bytes(_fake_smp())
    ogg = tmp_path / "hit.ogg"
    ok(gbtvgr("smp", "to-ogg", str(src), str(ogg)))
    assert ogg.read_bytes() == _fake_ogg()
    back = tmp_path / "hit2.smp"
    ok(gbtvgr("smp", "from-ogg", str(ogg), str(back)))
    assert back.read_bytes() == src.read_bytes()


def test_snb_encode_roundtrip(tmp_path):
    data = snb.encode(_fake_snb_model())
    assert snb.encode(snb.parse(data)) == data  # parse -> rebuild, in-process

    p = tmp_path / "sfx.snb"
    p.write_bytes(data)
    out = ok(gbtvgr("snb", "verify", str(p)))
    assert "OK" in out, out
    out = ok(gbtvgr("snb", "info", str(p)))
    assert "cues         : 1" in out, out


def test_snb_snd_both_ways(tmp_path):
    data = snb.encode(_fake_snb_model())
    m = snb.parse(data)
    # to_snd -> from_snd round-trips through a temp .snd on disk, as the CLI does
    snd = tmp_path / "sfx.snd"
    snd.write_text(snb.to_snd(m), encoding="latin1", newline="")
    assert snb.encode(snb.from_snd(str(snd), "W32", m["hdr_id"])) == data

    p = tmp_path / "sfx.snb"
    p.write_bytes(data)
    out_snd = tmp_path / "sfx2.snd"
    ok(gbtvgr("snb", "to-snd", str(p), str(out_snd)))
    assert os.path.isfile(out_snd)
    out_snb = tmp_path / "sfx2.snb"
    ok(gbtvgr("snb", "from-snd", str(out_snd), str(out_snb),
              "--template", str(p), "--compare", str(p)))
    assert out_snb.read_bytes() == data


def test_formats_listed():
    out = ok(gbtvgr())
    assert "smp" in out and "snb" in out, out
    ok(gbtvgr("smp", "--help"))
    ok(gbtvgr("snb", "--help"))


# ---------------------------------------------------------------- corpus

def test_smp_roundtrip_corpus(smp_corpus):
    """rebuild header from the Ogg payload: byte-identical, whole corpus."""
    out = ok(gbtvgr("smp", "roundtrip-test", smp_corpus))
    assert "FAIL" not in out, out


def test_snb_roundtrip_corpus(snb_corpus):
    """parse -> rebuild is byte-identical across the whole corpus."""
    out = ok(gbtvgr("snb", "roundtrip-test", snb_corpus))
    assert "FAIL" not in out, out


def test_snb_snd_roundtrip_corpus(snb_corpus):
    """snb -> snd -> snb is byte-identical across the whole corpus."""
    out = ok(gbtvgr("snb", "snd-roundtrip-test", snb_corpus))
    assert "FAIL" not in out, out

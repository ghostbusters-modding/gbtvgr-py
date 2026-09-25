"""Skinned character meshes and skeleton bone lists."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import glob
import os
import sys

from _run import gbtvgr, ok

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from gbtvgr.mesh import bfm  # noqa: E402

GB_CAST = ("gb_player", "venkman", "stantz", "spengler", "zeddemore")


def test_roundtrip_corpus(bfm_corpus):
    """parse -> rebuild is byte-identical across the whole shipped corpus."""
    out = ok(gbtvgr("bfm", "roundtrip-test", bfm_corpus))
    assert "FAIL" not in out, out


def test_vertex_codec_corpus(bfm_corpus):
    out = ok(gbtvgr("bfm", "vertex-test", bfm_corpus))
    assert "FAIL" not in out, out


def test_packet_invariants(bfm_corpus, corpus_files):
    """What the spec says holds on every shipped packet, so a writer must keep it."""
    files = corpus_files(bfm_corpus, ".bfm", 30)
    assert files, "no .bfm under %s" % bfm_corpus
    for fn in files:
        m = bfm.parse(open(fn, "rb").read())
        for p in m["packets"][0]:
            pal = m["lists"][p["list"]]
            assert bfm.STRIDE.get(p["decl"]) == bfm.stride(p), fn
            assert p["datasize"] == p["nverts"] * bfm.stride(p) + p["ntris"] * 6, fn
            assert p["palette"] == len(pal) <= bfm.MAX_PALETTE, fn
            assert len(p["parts"]) == 1 and p["parts"][0] < len(m["parts"]) and not p["extra"], fn
            v = bfm.decode_vertices(p)
            for bi, w in zip(v["bones"], v["weights"]):
                assert abs(sum(w) - 1) < 0.01, fn
                assert all(b < len(pal) for b, x in zip(bi, w) if x > 0), fn
            assert all(i < p["nverts"] for t in bfm.decode_tris(p) for i in t), fn


def test_set_geometry_identity(bfm_corpus, corpus_files):
    """Re-setting every packet from its own decoded geometry changes nothing."""
    for fn in corpus_files(bfm_corpus, ".bfm", 30):
        d = open(fn, "rb").read()
        m = bfm.parse(d)
        for p in m["packets"][0]:
            bfm.set_geometry(m, p, bfm.decode_vertices(p), bfm.decode_tris(p), p["decl"])
        assert bfm.build(m) == d, fn


def test_set_geometry_edit(bfm_corpus):
    """A real edit (half the triangles of every packet) rebuilds to a file that re-parses."""
    fn = os.path.join(bfm_corpus, "skeletal", "ghostbuster", "gb_player.bfm")
    m = bfm.parse(open(fn, "rb").read())
    for p in m["packets"][0]:
        tris = bfm.decode_tris(p)[:max(1, p["ntris"] // 2)]
        bfm.set_geometry(m, p, bfm.decode_vertices(p), tris, p["decl"])
    out = bfm.build(m)
    assert bfm.build(bfm.parse(out)) == out


def test_skb_corpus(skb_corpus):
    files = sorted(glob.glob(os.path.join(skb_corpus, "**", "*.skb"), recursive=True))
    assert files, "no .skb under %s" % skb_corpus
    for fn in files:
        bones = bfm.read_skb_bones(open(fn, "rb").read())
        assert all(-1 <= b["parent"] < len(bones) and -1 <= b["mirror"] < len(bones) for b in bones), fn


def test_ghostbuster_rigs_match(skb_corpus):
    """The ten ghostbuster rigs share one bone list, which is what makes a player swap work."""
    names = set()
    for c in GB_CAST:
        for s in ("", "_nopack"):
            fn = os.path.join(skb_corpus, "skeletal", "ghostbuster", c + s + ".skb")
            names.add(tuple((b["name"], b["parent"]) for b in bfm.read_skb_bones(open(fn, "rb").read())))
    assert len(names) == 1 and len(next(iter(names))) == 148

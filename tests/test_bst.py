"""Sets: the wire codec, and the editable geometry layer on top of it."""
from _run import gbtvgr, ok


def test_wire_roundtrip(bst_corpus):
    """parse -> rebuild is byte-identical across every shipped set."""
    out = ok(gbtvgr("bst", "roundtrip-test", bst_corpus))
    assert "FAIL" not in out, out


def test_geometry_and_relayout(bst_corpus):
    """Every mesh decodes and re-encodes exactly, and a full relayout of the
    file still reproduces it byte-for-byte."""
    out = ok(gbtvgr("bst-geom", "roundtrip-test", bst_corpus))
    assert "FAIL" not in out, out

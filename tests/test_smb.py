"""Static meshes: the codec, and OBJ in both directions."""
import os
import tempfile

from _run import gbtvgr, ok, script

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def test_roundtrip_corpus(smb_corpus):
    """parse -> rebuild is byte-identical across the whole shipped corpus."""
    out = ok(gbtvgr("smb", "roundtrip-test", smb_corpus))
    assert "FAIL" not in out, out


def test_obj_roundtrip_sample(smb_corpus, corpus_files):
    """smb -> obj -> smb is byte-identical, with the sidecar carrying what OBJ
    cannot express."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src"))
    import contextlib
    import io

    from gbtvgr.mesh import smb

    files = corpus_files(smb_corpus, ".smb", 40)
    assert files, "no .smb under %s" % smb_corpus

    tmp = tempfile.mkdtemp(prefix="smb_obj_")
    bad = []
    for fn in files:
        a = type("A", (), {})()
        a.infile = fn
        a.outfile = os.path.join(tmp, "x.obj")
        a.sidecar = os.path.join(tmp, "x.smb.json")
        with contextlib.redirect_stdout(io.StringIO()):
            smb.cmd_to_obj(a)
        a.infile = a.outfile
        if smb.build(smb.rebuild_exact(a)) != open(fn, "rb").read():
            bad.append(fn)
    assert not bad, "%d/%d differed: %s" % (len(bad), len(files), bad[:5])


def test_mtb_roundtrip(mtb_corpus):
    """.mtb material tables rebuild byte-identically."""
    out = ok(gbtvgr("smb", "mtb-roundtrip-test", mtb_corpus))
    assert "FAIL" not in out, out


def test_from_obj_new_mesh(tmp_path):
    """A mesh authored from scratch, not derived from the corpus: the obelisk
    example -> OBJ -> from-obj -> verify.  Needs no game files."""
    obj = tmp_path / "obelisk.obj"
    out = tmp_path / "obelisk.smb"
    r = script(os.path.join(EXAMPLES, "make_obelisk.py"), obj)
    assert r.returncode == 0, r.stdout + r.stderr
    ok(gbtvgr("smb", "from-obj", obj, out))
    assert out.stat().st_size > 0
    text = ok(gbtvgr("smb", "verify", out))
    assert "FAIL" not in text, text

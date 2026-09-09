"""POD6 archives and the mount chain.

The substance is in `pod_suite.py`, a self-contained suite that builds its own
archives in a temp directory; this runs it and surfaces its output on failure.
"""
import os

from _run import script

HERE = os.path.dirname(os.path.abspath(__file__))


def test_pod_suite(request):
    """Header layout, chain field, legacy rewrite, content/install/uninstall."""
    game = request.config.getoption("--game") or os.environ.get("GAME_DIR") or "/nonexistent"
    r = script(os.path.join(HERE, "pod_suite.py"), "--game", game)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "all passed" in r.stdout


def test_shipped_archives_readable(game_dir):
    """Every shipped .POD parses, and its chain field is legible."""
    import glob
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
    from gbtvgr.archive import pod

    pods = sorted(glob.glob(os.path.join(game_dir, "*.POD")))
    assert pods, "no .POD archives in %s" % game_dir
    for path in pods:
        p = pod.Pod(path)
        assert p.entries, "%s parsed to zero entries" % os.path.basename(path)

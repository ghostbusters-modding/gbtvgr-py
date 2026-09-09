"""Shelling out to the real `gbtvgr` entry point.

The codec round-trips are driven through the CLI rather than the module
functions on purpose: a break in argument parsing is as much a regression as a
break in a decoder, and this is the surface every mod build script uses.
"""
import os
import subprocess
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def gbtvgr(*args, **kw):
    """Run `gbtvgr <args>` in-process-adjacent (own interpreter, own env)."""
    env = dict(os.environ, PYTHONPATH=SRC)
    return subprocess.run([sys.executable, "-m", "gbtvgr"] + [str(a) for a in args],
                          capture_output=True, text=True, env=env, **kw)


def ok(r):
    """Assert a run succeeded, showing its own output when it did not."""
    assert r.returncode == 0, "gbtvgr exited %d\n%s%s" % (r.returncode, r.stdout, r.stderr)
    return r.stdout


def script(path, *args):
    env = dict(os.environ, PYTHONPATH=SRC)
    return subprocess.run([sys.executable, path] + [str(a) for a in args],
                          capture_output=True, text=True, env=env)

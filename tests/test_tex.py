"""Textures: the pure-python decoders against the Pillow-backed path.

`bst_tex` decodes BC1/BC3/raw itself because Blender ships no Pillow; these
checks hold it to what `tex` produces through DDS.
"""
import os

from _run import script

HERE = os.path.dirname(os.path.abspath(__file__))


def test_decoders_match(game_dir):
    r = script(os.path.join(HERE, "tex_suite.py"), "--game", game_dir)
    assert r.returncode == 0, r.stdout + r.stderr

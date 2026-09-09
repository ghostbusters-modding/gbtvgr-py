"""
Shared fixtures for the gbtvgr suite.

Nothing here ships game content, so almost every check needs an installation to
read.  Point the suite at yours -- the archive-level checks want the game
directory, the codec round-trips want a corpus extracted out of it:

    gbtvgr pod extract "<game>/W64MODEL.POD" -o out/smb  -f .smb
    gbtvgr pod extract "<game>/W64SET.POD"   -o out/sets -f .bst
    gbtvgr pod extract "<game>/W64ART02.POD" -o out/mtb  -f .mtb

    GAME_DIR="<game>" SMB_CORPUS=out/smb BST_CORPUS=out/sets pytest
    pytest --game="<game>" --smb-corpus=out/smb --bst-corpus=out/sets

Pass these as `--opt=VALUE`: a bare `--game <dir>` with no test path makes
pytest read the directory as the test path and miss this repo's config.

Whatever you leave out, those tests skip and the rest still run.
"""
import os

import pytest

_OPTS = [
    ("--game",       "GAME_DIR",    "the game installation directory"),
    ("--smb-corpus", "SMB_CORPUS",  "directory of shipped .smb meshes (from W64MODEL.POD)"),
    ("--bst-corpus", "BST_CORPUS",  "directory of shipped .bst sets (from W64SET.POD)"),
    ("--mtb-corpus", "MTB_CORPUS",  "directory of shipped .mtb materials (from W64ART02.POD)"),
]


def pytest_addoption(parser):
    for flag, env, help_ in _OPTS:
        parser.addoption(flag, action="store", default=None,
                         help="%s; pass it as %s=DIR (default: $%s)" % (help_, flag, env))


def _dir(config, flag, env):
    path = config.getoption(flag) or os.environ.get(env)
    return path if path and os.path.isdir(path) else None


def _fixture(flag, env, label):
    @pytest.fixture(scope="session")
    def f(request):
        path = _dir(request.config, flag, env)
        if path is None:
            pytest.skip("needs %s (%s / $%s)" % (label, flag, env))
        return path
    return f


game_dir    = _fixture("--game",       "GAME_DIR",   "the game directory")
smb_corpus  = _fixture("--smb-corpus", "SMB_CORPUS", "a .smb corpus")
bst_corpus  = _fixture("--bst-corpus", "BST_CORPUS", "a .bst corpus")
mtb_corpus  = _fixture("--mtb-corpus", "MTB_CORPUS", "a .mtb corpus")


@pytest.fixture(scope="session")
def corpus_files():
    """A stable, spread-out sample of a corpus -- the round-trips run over the
    whole thing, the slower per-mesh checks over a sample of it."""
    import glob

    def pick(root, ext, n=None):
        files = sorted(glob.glob(os.path.join(root, "**", "*" + ext), recursive=True))
        if n is None or len(files) <= n:
            return files
        return files[::max(1, len(files) // n)][:n]
    return pick

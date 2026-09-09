#!/usr/bin/env python3
"""POD6 archive + chain-field regression checks.

  python3 tests/pod_suite.py [--game "<gamedir>"]

The shipped-archive checks are skipped when --game is absent or unreadable; every
other test is self-contained (it builds its own archives in a temp directory).
"""
import argparse, os, shutil, struct, subprocess, sys, tempfile, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
from gbtvgr.archive import pod  # noqa: E402

# The CLI checks drive the real entry point rather than the module functions,
# so an argument-parsing regression fails here too.
POD_CLI = ['-m', 'gbtvgr', 'pod']
PATCHPOD_CLI = ['-m', 'gbtvgr', 'patchpod']

FAILURES = []


def check(cond, label):
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILURES.append(label)
    return cond


def run(*args, expect_ok=True):
    argv = list(args)
    cmd = [sys.executable] + (argv[0] if isinstance(argv[0], list) else [argv[0]]) + \
        [str(x) for x in argv[1:]]
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(HERE), 'src'))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if expect_ok and r.returncode != 0:
        print(r.stdout + r.stderr)
    return r


def make_tree(root, files):
    for rel, data in files.items():
        p = os.path.join(root, rel.replace('\\', os.sep))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'wb') as fh:
            fh.write(data)


SAMPLE = {
    'world\\immortal1.dante': b'ASSUMPTIONS\nSTRINGS\n' + b'x' * 5000,
    'art\\ui\\thing.tex': bytes(range(256)) * 40,
    'data\\ui\\thing.fnt': b'font data\n',
}


# --- header + chain field ----------------------------------------------------

def test_header_layout(tmp):
    print("header layout")
    src = os.path.join(tmp, 'src'); make_tree(src, SAMPLE)
    out = os.path.join(tmp, 'a.POD')
    run(POD_CLI, 'create', out, '-d', src)
    raw = open(out, 'rb').read(pod.HDR)
    check(len(raw) == 0x80 and raw[:4] == b'POD6', "128-byte header, POD6 magic")
    p = pod.Pod(out)
    check(p.body_start >= pod.HDR,
          "body starts at 0x%X, at or after the 0x80 header" % p.body_start)
    check(p.next_pod == '', "chain field empty by default")
    check(raw[pod.CHAIN_OFF:pod.HDR] == b'\0' * pod.CHAIN_ROOM, "chain field is all NUL")
    p.f.close()


def test_create_with_next(tmp):
    print("create --next")
    src = os.path.join(tmp, 'src')
    out = os.path.join(tmp, 'b.POD')
    run(POD_CLI, 'create', out, '-d', src, '--next', 'IMMORTAL.POD')
    p = pod.Pod(out)
    check(p.next_pod == 'IMMORTAL.POD', "chain field round-trips through create")
    check(len(p.entries) == len(SAMPLE), "all %d entries present" % len(SAMPLE))
    p.f.close()


def test_stamp_preserves_content(tmp):
    print("chain stamp preserves content byte-for-byte")
    src = os.path.join(tmp, 'src')
    out = os.path.join(tmp, 'c.POD')
    run(POD_CLI, 'create', out, '-d', src)
    before = open(out, 'rb').read()
    r = run(POD_CLI, 'chain', out, 'IMMORTAL.POD')
    check(r.returncode == 0 and 'stamped' in r.stdout, "stamped in place (no rebuild)")
    after = open(out, 'rb').read()
    check(len(before) == len(after), "file size unchanged")
    check(before[:pod.CHAIN_OFF] == after[:pod.CHAIN_OFF] and before[pod.HDR:] == after[pod.HDR:],
          "only the 108 chain bytes changed")
    p = pod.Pod(out)
    check(p.next_pod == 'IMMORTAL.POD', "chain reads back")
    check(all(p.read(e) == SAMPLE[e['name']] for e in p.entries), "every stream still decompresses")
    p.f.close()
    run(POD_CLI, 'chain', out, '')
    check(open(out, 'rb').read() == before, "clearing the chain restores the original bytes")


def test_chain_print(tmp):
    print("chain printing")
    src = os.path.join(tmp, 'src')
    a, b, c = (os.path.join(tmp, n) for n in ('A.POD', 'B.POD', 'C.POD'))
    for path in (a, b, c):
        run(POD_CLI, 'create', path, '-d', src)
    run(POD_CLI, 'chain', a, 'B.POD')
    run(POD_CLI, 'chain', b, 'C.POD')
    r = run(POD_CLI, 'chain', a)
    check('A.POD -> B.POD -> C.POD -> (end)' in r.stdout, "walks the whole chain: %s" % r.stdout.strip())
    names, note = pod.walk_chain(a)
    check(names == ['A.POD', 'B.POD', 'C.POD'] and note is None, "walk_chain() agrees")
    r = run(POD_CLI, 'chain', c, 'MISSING.POD')
    r = run(POD_CLI, 'chain', a)
    check('not present' in r.stdout, "flags a chain target that isn't on disk")
    run(POD_CLI, 'chain', c, '')


def test_chain_refusals(tmp):
    print("chain refusals")
    src = os.path.join(tmp, 'src')
    a, b = os.path.join(tmp, 'R1.POD'), os.path.join(tmp, 'R2.POD')
    for path in (a, b):
        run(POD_CLI, 'create', path, '-d', src)
    r = run(POD_CLI, 'chain', a, 'R1.POD', expect_ok=False)
    check(r.returncode != 0 and 'itself' in r.stderr, "refuses a self-chain")
    run(POD_CLI, 'chain', b, 'R1.POD')
    r = run(POD_CLI, 'chain', a, 'R2.POD', expect_ok=False)
    check(r.returncode != 0 and 'cycle' in r.stderr, "refuses to close a cycle")
    r = run(POD_CLI, 'chain', a, 'X' * 80, expect_ok=False)
    check(r.returncode != 0 and 'buffer' in r.stderr, "refuses a name over %d bytes" % pod.CHAIN_MAX)
    r = run(POD_CLI, 'chain', a, 'X' * 79)
    check(r.returncode == 0, "accepts a name of exactly %d bytes" % pod.CHAIN_MAX)
    run(POD_CLI, 'chain', a, '')
    run(POD_CLI, 'chain', b, '')

    # a POD that sits next to ghost.exe is off-limits without the explicit flag
    gd = os.path.join(tmp, 'fakegame'); os.makedirs(gd, exist_ok=True)
    open(os.path.join(gd, 'ghost.exe'), 'wb').close()
    g = os.path.join(gd, 'PATCH.POD')
    run(POD_CLI, 'create', g, '-d', src)
    r = run(POD_CLI, 'chain', g, 'IMMORTAL.POD', expect_ok=False)
    check(r.returncode != 0 and 'ghost.exe' in r.stderr, "refuses a game-dir POD by default")
    r = run(POD_CLI, 'chain', g, 'IMMORTAL.POD', '--allow-game-dir')
    check(r.returncode == 0 and pod.read_chain(g) == 'IMMORTAL.POD', "--allow-game-dir permits it")


def _write_legacy_pod(path, files):
    """A pre-chain POD6: 20-byte header, body starting at 0x20 (what pod.py used to emit)."""
    out = open(path, 'wb')
    out.write(b'\0' * 20)
    index = []
    for rel in sorted(files):
        blob = zlib.compress(files[rel], 9)
        while out.tell() % 16:
            out.write(b'\0')
        off = out.tell()
        out.write(blob)
        index.append((rel, len(blob), off, len(files[rel])))
    while out.tell() % 16:
        out.write(b'\0')
    index_off = out.tell()
    names, name_off = bytearray(), {}
    for rel, _, _, _ in index:
        name_off[rel] = len(names)
        names += rel.encode('ascii') + b'\0'
    while len(names) % 4:
        names.append(0)
    for rel, csize, off, usize in index:
        out.write(struct.pack('<IIIIII', name_off[rel], csize, off, usize, 8, 0))
    out.write(bytes(names))
    out.seek(0)
    out.write(b'POD6' + struct.pack('<IIII', len(index), 992, index_off, len(names)))
    out.close()


def test_legacy_rewrite(tmp):
    print("legacy short-header POD")
    legacy = os.path.join(tmp, 'LEGACY.POD')
    _write_legacy_pod(legacy, SAMPLE)
    p = pod.Pod(legacy)
    check(p.body_start < pod.HDR, "body starts at 0x%X, inside the header" % p.body_start)
    check(not p.chain_writable, "chain field reported as not writable in place")
    p.f.close()
    r = run(POD_CLI, 'chain', legacy, 'IMMORTAL.POD', '--no-rewrite', expect_ok=False)
    check(r.returncode != 0 and 'rebuilt' in r.stderr, "--no-rewrite refuses instead of clobbering")
    r = run(POD_CLI, 'chain', legacy, 'IMMORTAL.POD')
    check(r.returncode == 0 and 'rebuilt' in r.stdout, "rebuilds it by default")
    p = pod.Pod(legacy)
    check(p.next_pod == 'IMMORTAL.POD' and p.body_start >= pod.HDR, "rebuilt with a full header")
    check(p.revision == 992, "revision preserved")
    check({e['name']: p.read(e) for e in p.entries} == SAMPLE, "every file survived the rebuild")
    p.f.close()


# --- patchpod content / install / uninstall ----------------------------------

def test_content_and_install(tmp):
    print("patchpod content + install + uninstall")
    src = os.path.join(tmp, 'src')
    game = os.path.join(tmp, 'game'); os.makedirs(game, exist_ok=True)
    open(os.path.join(game, 'ghost.exe'), 'wb').close()
    host = os.path.join(game, 'PATCH.POD')
    fonts = os.path.join(tmp, 'fonts')
    make_tree(fonts, {'data\\ui\\font_calibri_ja.fnt': b'shipped font'})
    run(POD_CLI, 'create', host, '-d', fonts)
    host_before = open(host, 'rb').read()

    content = os.path.join(tmp, 'IMMORTAL.POD')
    r = run(PATCHPOD_CLI, 'content', '--mods', src, '-o', content, '--game', game)
    check(r.returncode == 0, "content builds")
    check('overrides' not in r.stdout, "no shipped file is shadowed by this mod")

    r = run(PATCHPOD_CLI, 'content', '--mods', fonts, '-o', os.path.join(tmp, 'shadow.POD'),
            '--game', game)
    check('overrides PATCH.POD' in r.stdout, "reports what a mod file shadows")

    r = run(PATCHPOD_CLI, 'content', '--mods', src, '-o', os.path.join(game, 'nope.POD'),
            expect_ok=False)
    check(r.returncode != 0, "content refuses to write next to ghost.exe")

    r = run(PATCHPOD_CLI, 'install', '--game', game, '--pod', content, '--force')
    check(r.returncode == 0, "install succeeds")
    installed = os.path.join(game, 'IMMORTAL.POD')
    check(os.path.isfile(installed), "IMMORTAL.POD is in the game dir")
    check(open(installed, 'rb').read() == open(content, 'rb').read(), "installed copy is identical")
    check(pod.read_chain(host) == 'IMMORTAL.POD', "PATCH.POD now chains to it")
    after = open(host, 'rb').read()
    check(after[:pod.CHAIN_OFF] == host_before[:pod.CHAIN_OFF]
          and after[pod.HDR:] == host_before[pod.HDR:],
          "PATCH.POD content untouched -- only its chain field moved")
    check('PATCH.POD -> IMMORTAL.POD -> (end)' in r.stdout, "install prints the mount chain")

    r2 = run(PATCHPOD_CLI, 'install', '--game', game, '--pod', content, '--force')
    check(r2.returncode == 0 and 'unchanged' in r2.stdout, "re-installing is idempotent")
    check(open(host, 'rb').read() == after, "a no-op re-install doesn't rewrite PATCH.POD")

    r = run(PATCHPOD_CLI, 'uninstall', '--game', game, '--force')
    check(r.returncode == 0, "uninstall succeeds")
    check(not os.path.exists(installed), "IMMORTAL.POD removed")
    check(open(host, 'rb').read() == host_before, "PATCH.POD restored byte-for-byte")


def test_keep_chain(tmp):
    print("patchpod install --keep-chain inserts into the chain instead of truncating it")
    src = os.path.join(tmp, 'src')
    game = os.path.join(tmp, 'game3'); os.makedirs(game, exist_ok=True)
    open(os.path.join(game, 'ghost.exe'), 'wb').close()
    fonts = os.path.join(tmp, 'fonts')
    host = os.path.join(game, 'PATCH.POD')
    mods = os.path.join(game, 'MODS.POD')          # what a mod manager would leave behind
    content = os.path.join(tmp, 'IMMORTAL.POD')
    run(PATCHPOD_CLI, 'content', '--mods', src, '-o', content)

    def fresh_host():
        run(POD_CLI, 'create', host, '-d', fonts, '--next', 'MODS.POD')
        run(POD_CLI, 'create', mods, '-d', fonts)
        if os.path.exists(os.path.join(game, 'IMMORTAL.POD')):
            os.remove(os.path.join(game, 'IMMORTAL.POD'))

    fresh_host()
    r = run(PATCHPOD_CLI, 'install', '--game', game, '--pod', content, '--force')
    check(r.returncode == 0 and 'PATCH.POD -> IMMORTAL.POD -> (end)' in r.stdout,
          "without the flag, install truncates the chain (MODS.POD dropped)")

    fresh_host()
    r = run(PATCHPOD_CLI, 'install', '--game', game, '--pod', content, '--force', '--keep-chain')
    check(r.returncode == 0 and 'PATCH.POD -> IMMORTAL.POD -> MODS.POD -> (end)' in r.stdout,
          "--keep-chain on a fresh install chains onward to what the host used to mount")
    check('kept from the previous chain' in r.stdout, "and says where the link came from")
    check(pod.read_chain(os.path.join(game, 'IMMORTAL.POD')) == 'MODS.POD',
          "the installed archive's own chain field carries the link")

    r = run(PATCHPOD_CLI, 'install', '--game', game, '--pod', content, '--force', '--keep-chain')
    check(r.returncode == 0 and 'PATCH.POD -> IMMORTAL.POD -> MODS.POD -> (end)' in r.stdout,
          "--keep-chain over an archive that already links onward keeps the link")

    r = run(PATCHPOD_CLI, 'install', '--game', game, '--pod', content, '--force')
    check(r.returncode == 0 and 'PATCH.POD -> IMMORTAL.POD -> (end)' in r.stdout,
          "re-installing without the flag truncates again (the flag is opt-in)")

    r = run(PATCHPOD_CLI, 'uninstall', '--game', game, '--force')
    check(r.returncode == 0 and not os.path.exists(os.path.join(game, 'IMMORTAL.POD')),
          "uninstall still removes it")


def test_build_preserves_chain(tmp):
    print("patchpod build keeps the chain field")
    src = os.path.join(tmp, 'src')
    game = os.path.join(tmp, 'game2'); os.makedirs(game, exist_ok=True)
    fonts = os.path.join(tmp, 'fonts')
    host = os.path.join(game, 'PATCH.POD')
    run(POD_CLI, 'create', host, '-d', fonts, '--next', 'IMMORTAL.POD')
    out = os.path.join(tmp, 'rebuilt.POD')
    r = run(PATCHPOD_CLI, 'build', '--game', game, '--mods', src, '-o', out)
    check(r.returncode == 0, "build succeeds")
    p = pod.Pod(out)
    check(p.next_pod == 'IMMORTAL.POD', "rebuilt PATCH.POD still chains to IMMORTAL.POD")
    check(len(p.entries) == len(SAMPLE) + 1, "original entry kept, mod entries added")
    p.f.close()
    r = run(PATCHPOD_CLI, 'diff', host, out)
    check(r.returncode == 0 and '+ world\\immortal1.dante' in r.stdout, "diff reports the additions")


# --- shipped archives --------------------------------------------------------

def test_shipped(game):
    print("shipped archives in %s" % game)
    expect = {'W64ART.POD': 'W64ART02.POD'}
    seen = {}
    for name in sorted(os.listdir(game)):
        if not name.upper().endswith('.POD'):
            continue
        path = os.path.join(game, name)
        try:
            p = pod.Pod(path)
        except (OSError, ValueError) as exc:
            check(False, "%s: %s" % (name, exc))
            continue
        seen[name] = p.next_pod
        p.f.close()
    check(seen.get('W64ART.POD') == expect['W64ART.POD'],
          "W64ART.POD chains to W64ART02.POD (found %r)" % seen.get('W64ART.POD'))
    others = {k: v for k, v in seen.items() if v and k != 'W64ART.POD'}
    print("     chains found: %s" % (seen if others else
                                     {k: v for k, v in seen.items() if v}))
    for name in ('COMMON.POD', 'W64SET.POD', 'PATCH.POD'):
        if name in seen:
            check(True, "%s parses with the 128-byte header" % name)
    orig = os.path.join(game, 'PATCH.POD.orig')
    if os.path.isfile(orig):
        p = pod.Pod(orig)
        check(p.body_start >= pod.HDR,
              "shipped PATCH.POD body starts at 0x%X (>= 0x80)" % p.body_start)
        p.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--game', default=os.environ.get('GAME_DIR', ''),
                    help='a game installation; the shipped-archive checks skip '
                         'without one (default: $GAME_DIR)')
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix='podtest_')
    try:
        make_tree(os.path.join(tmp, 'src'), SAMPLE)
        test_header_layout(tmp)
        test_create_with_next(tmp)
        test_stamp_preserves_content(tmp)
        test_chain_print(tmp)
        test_chain_refusals(tmp)
        test_legacy_rewrite(tmp)
        test_content_and_install(tmp)
        test_keep_chain(tmp)
        test_build_preserves_chain(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if a.game and os.path.isdir(a.game):
        test_shipped(a.game)
    else:
        print("shipped archives: SKIPPED (no game dir at %s)" % a.game)

    print()
    if FAILURES:
        print("%d FAILED:" % len(FAILURES))
        for f in FAILURES:
            print("  - %s" % f)
        return 1
    print("pod tests: all passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())

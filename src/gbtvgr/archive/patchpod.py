#!/usr/bin/env python3
"""
Mod archives: build a chained IMMORTAL.POD, install it, roll it back.

`install` writes exactly two things into the game directory: the mod archive
itself, and PATCH.POD's 108-byte chain field. PATCH.POD's content is never
touched, so a Mod Manager PATCH.POD keeps working and `uninstall` restores it.
Every other subcommand opens the game directory read-only.

Commands:
  content   --mods MODS_FILES_DIR -o OUT.POD [--game GAMEDIR] [--next NAME] [--level N]
  install   --game GAMEDIR --pod OUT.POD [--as NAME.POD] [--host PATCH.POD] [--force] [--keep-chain]
  uninstall --game GAMEDIR [--as NAME.POD] [--host PATCH.POD] [--force]
  build     --game GAMEDIR --mods MODS_FILES_DIR -o OUT.POD [--revision N] [--level N] [--next NAME]
  verify    POD
  diff      A.POD B.POD

`build` is the older workflow: a full replacement PATCH.POD starting from every
entry of the shipped one, so its CJK font entries survive.
"""
import argparse, hashlib, os, shutil, subprocess, sys

from . import pod

PATCH_NAME = "PATCH.POD"
CONTENT_NAME = "IMMORTAL.POD"
# Mount order in CPod::mountDefaultPods; first match wins.
BULK_PODS = ["W64ENSND.POD", "W64MUSND.POD", "W64ART.POD", "W64SOUND.POD",
             "W64SET.POD", "W64MODEL.POD", "LANGUAGE.POD", "COMMON.POD"]


def _abspath(p):
    return os.path.realpath(os.path.abspath(p))


def _under(path, root):
    path = _abspath(path)
    root = _abspath(root)
    return path == root or path.startswith(root + os.sep)


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _walk_mod_files(moddir):
    """Yield (pod_name, abs_path) for every file under moddir, pod_name using backslashes."""
    out = []
    for root, _, names in os.walk(moddir):
        for nm in names:
            full = os.path.join(root, nm)
            rel = os.path.relpath(full, moddir).replace(os.sep, '\\')
            out.append((rel, full))
    out.sort()
    return out


def _game_is_running():
    """True if ghost.exe shows up in tasklist.exe; None if we can't tell."""
    try:
        out = subprocess.run(['tasklist.exe', '/FI', 'IMAGENAME eq ghost.exe'],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return 'ghost.exe' in out.stdout


def _shadow_report(mods, game):
    """Report which shipped archive each mod file overrides. Returns lines."""
    lines, seen, done = [], {}, set()
    roots = []
    for name in [PATCH_NAME] + BULK_PODS:
        path = os.path.join(game, name)
        if os.path.isfile(path):
            roots.extend(pod.walk_chain(path)[0])   # a chained archive mounts too
    for name in roots:
        if name.lower() in done:
            continue
        done.add(name.lower())
        path = os.path.join(game, name)
        if not os.path.isfile(path):
            continue
        try:
            p = pod.Pod(path)
        except (OSError, ValueError):
            continue
        for e in p.entries:
            seen.setdefault(e['name'].lower(), name)
        p.close()
    new = 0
    for rel, _ in mods:
        host = seen.get(rel.lower())
        if host is None:
            new += 1
        else:
            lines.append("  %-52s overrides %s" % (rel, host))
    if new:
        lines.append("  %d file(s) are new -- they exist in no shipped archive" % new)
    return lines


def cmd_content(a):
    if not os.path.isdir(a.mods):
        sys.exit("no such mods directory: %s" % a.mods)
    out_abs = _abspath(a.out)
    if a.game and _under(out_abs, a.game):
        sys.exit("refusing to write inside the game directory: %s" % a.out)
    if os.path.exists(os.path.join(os.path.dirname(out_abs), 'ghost.exe')):
        sys.exit("refusing to write next to ghost.exe: %s" % a.out)

    mod_files = _walk_mod_files(a.mods)
    if not mod_files:
        sys.exit("no files found under %s" % a.mods)

    index = pod.write_pod(a.out, [dict(name=rel, path=full) for rel, full in mod_files],
                          revision=a.revision, next_pod=a.next or '', level=a.level)
    print("wrote %s: %d entries, %d bytes%s"
          % (a.out, len(index), os.path.getsize(a.out),
             ("  chain -> %s" % a.next) if a.next else ""))
    if a.game and os.path.isdir(a.game):
        for line in _shadow_report(mod_files, a.game):
            print(line)


def _resolve_install(a):
    game = a.game
    if not os.path.isdir(game):
        sys.exit("no such game directory: %s" % game)
    host = os.path.join(game, a.host)
    if not os.path.isfile(host):
        sys.exit("no %s under %s" % (a.host, game))
    if not a.force:
        running = _game_is_running()
        if running:
            sys.exit("ghost.exe is running -- close the game first (or pass --force)")
        if running is None:
            print("warning: could not run tasklist.exe to check whether ghost.exe is running",
                  file=sys.stderr)
    return game, host


def cmd_install(a):
    game, host = _resolve_install(a)
    if not os.path.isfile(a.pod):
        sys.exit("no such archive: %s" % a.pod)
    with pod.Pod(a.pod) as src:
        n, onward = len(src.entries), src.next_pod
        mine = {e['name'].lower() for e in src.entries}

    # The host is mounted first, so anything it already carries shadows our copy.
    with pod.Pod(host) as hp:
        shadowed = sorted({e['name'] for e in hp.entries if e['name'].lower() in mine})
    if shadowed:
        print("WARNING: %s already contains %d of these %d files. It is mounted BEFORE %s, so "
              "its copies win and this install will have no effect for them:"
              % (a.host, len(shadowed), n, a.name), file=sys.stderr)
        for name in shadowed[:8]:
            print("  %s" % name, file=sys.stderr)
        if len(shadowed) > 8:
            print("  ... and %d more" % (len(shadowed) - 8), file=sys.stderr)
        print("Restore a clean %s first (e.g. from %s.orig, or via the Mod Manager), then "
              "re-run install." % (a.host, a.host), file=sys.stderr)

    dest = os.path.join(game, a.name)
    if _abspath(a.pod) == _abspath(dest):
        sys.exit("source and destination are the same file: %s" % dest)
    old_md5 = _md5(dest) if os.path.isfile(dest) else None
    # The archive being replaced may itself link onward (ONLINE.POD, a mod
    # manager's MODS.POD): remember it so --keep-chain can carry it over.
    prev_onward = ''
    if a.keep_chain and old_md5:
        try:
            with pod.Pod(dest) as _prev:
                prev_onward = _prev.next_pod or ''
        except (OSError, ValueError):
            prev_onward = ''
    shutil.copyfile(a.pod, dest)
    new_md5 = _md5(dest)
    if new_md5 != _md5(a.pod):
        sys.exit("copy of %s did not verify" % dest)
    print("installed %s: %d entries, md5 %s%s"
          % (dest, n, new_md5, ("  (was %s)" % old_md5) if old_md5 else "  (new file)"))
    if onward:
        print("  %s chains onward to %s" % (a.name, onward))

    try:
        old, new, how = pod.set_chain(host, a.name, allow_game_dir=True)
    except ValueError as exc:
        sys.exit("could not chain %s -> %s: %s" % (a.host, a.name, exc))
    print("%s: chain %s -> %s  (%s, md5 %s)"
          % (a.host, old or "(none)", new, how, _md5(host)))
    if a.keep_chain:
        # Insert rather than truncate: whatever the host mounted next still
        # mounts, after us, unless the archive names its own onward link.
        if not onward and prev_onward and prev_onward.lower() != a.name.lower():
            try:
                pod.set_chain(dest, prev_onward, allow_game_dir=True)
                onward = prev_onward
                print("  %s chains onward to %s (kept from the replaced archive)" % (a.name, prev_onward))
            except ValueError as exc:
                print("  WARNING: could not chain %s -> %s: %s" % (a.name, prev_onward, exc), file=sys.stderr)
        if old and old.lower() != a.name.lower() and not onward:
            try:
                pod.set_chain(dest, old, allow_game_dir=True)
                print("  %s chains onward to %s (kept from the previous chain)" % (a.name, old))
            except ValueError as exc:
                print("  WARNING: could not chain %s -> %s: %s" % (a.name, old, exc), file=sys.stderr)
    names, note = pod.walk_chain(host)
    print("  mount chain: " + " -> ".join(names) + (" -> (end)" if note is None else ""))
    if note:
        print("  note: %s" % note)


def cmd_uninstall(a):
    game, host = _resolve_install(a)
    try:
        old, _, how = pod.set_chain(host, '', allow_game_dir=True)
    except ValueError as exc:
        sys.exit("could not clear %s's chain: %s" % (a.host, exc))
    print("%s: chain %s -> (none)  (%s, md5 %s)" % (a.host, old or "(none)", how, _md5(host)))
    dest = os.path.join(game, a.name)
    if os.path.isfile(dest):
        os.unlink(dest)
        print("removed %s" % dest)
    else:
        print("%s was not installed" % dest)


def cmd_build(a):
    game_patch = os.path.join(a.game, PATCH_NAME)
    if not os.path.isfile(game_patch):
        sys.exit("no %s under %s" % (PATCH_NAME, a.game))
    if not os.path.isdir(a.mods):
        sys.exit("no such mods directory: %s" % a.mods)

    out_abs = _abspath(a.out)
    if _under(out_abs, a.game):
        sys.exit("refusing to write inside the game directory: %s" % a.out)

    orig = pod.Pod(game_patch)
    revision = a.revision if a.revision is not None else orig.revision
    next_pod = a.next if a.next is not None else orig.next_pod
    norig = len(orig.entries)

    # Original order preserved; overlay wins by name, matched case-insensitively
    # but stored with the shipped archive's casing.
    final = []              # list of dicts: name, source ('orig'|'file'), + payload info
    by_lower = {}            # lname -> index into final[]
    for e in orig.entries:
        by_lower[e['name'].lower()] = len(final)
        final.append(dict(name=e['name'], copy=(orig, e)))

    mod_files = _walk_mod_files(a.mods)
    if not mod_files:
        print("warning: no files found under %s" % a.mods, file=sys.stderr)

    added = replaced = 0
    for rel, full in mod_files:
        lname = rel.lower()
        if lname in by_lower:
            idx = by_lower[lname]
            final[idx] = dict(name=final[idx]['name'], path=full)  # keep original casing
            replaced += 1
        else:
            by_lower[lname] = len(final)
            final.append(dict(name=rel, path=full))
            added += 1

    index = pod.write_pod(a.out, final, revision=revision, next_pod=next_pod, level=a.level)
    orig.close()
    print("wrote %s: %d entries (%d from original, %d replaced, %d added), revision %d%s"
          % (a.out, len(index), norig - replaced, replaced, added, revision,
             ("  chain -> %s" % next_pod) if next_pod else ""))


def _sha1_stream(p, e, chunk=1 << 20):
    p.f.seek(e['off'])
    remaining = e['csize']
    h = hashlib.sha1()
    while remaining:
        n = p.f.read(min(chunk, remaining))
        if not n:
            break
        h.update(n)
        remaining -= len(n)
    return h.hexdigest()


def cmd_verify(a):
    with pod.Pod(a.archive) as p:
        _verify(a, p)


def _verify(a, p):
    print("%s  POD6  %d files  revision %d  index @0x%X%s"
          % (os.path.basename(a.archive), p.count, p.revision, p.index_off,
             ("  chain -> %s" % p.next_pod) if p.next_pod else ""))
    ok = bad = 0
    for e in p.entries:
        try:
            data = p.read(e)  # decompresses + checks len(data) == usize internally
            digest = hashlib.sha1(data).hexdigest()[:12]
            ok += 1
            if a.verbose:
                print("  OK   %-52s %10d -> %10d  m=%d  sha1=%s"
                      % (e['name'], e['csize'], e['usize'], e['method'], digest))
        except Exception as exc:
            bad += 1
            print("  FAIL %-52s %s" % (e['name'], exc))
    print("verify: %d/%d streams OK%s" % (ok, p.count, ("  (%d FAILED)" % bad) if bad else ""))
    if bad:
        sys.exit(1)


def cmd_diff(a):
    with pod.Pod(a.a) as pa, pod.Pod(a.b) as pb:
        _diff(a, pa, pb)


def _diff(a, pa, pb):
    ea = {e['name']: e for e in pa.entries}
    eb = {e['name']: e for e in pb.entries}
    added = sorted(set(eb) - set(ea))
    removed = sorted(set(ea) - set(eb))
    common = sorted(set(ea) & set(eb))

    changed, unchanged = [], []
    for name in common:
        x, y = ea[name], eb[name]
        if x['usize'] != y['usize'] or x['method'] != y['method'] or x['csize'] != y['csize']:
            changed.append(name)
            continue
        # same sizes -- confirm the compressed bytes actually match before calling it unchanged
        if _sha1_stream(pa, x) != _sha1_stream(pb, y):
            changed.append(name)
        else:
            unchanged.append(name)

    print("%s vs %s" % (os.path.basename(a.a), os.path.basename(a.b)))
    if pa.next_pod != pb.next_pod:
        print("  chain: %s -> %s" % (pa.next_pod or "(none)", pb.next_pod or "(none)"))
    for name in added:
        print("  + %s  (%d bytes)" % (name, eb[name]['usize']))
    for name in removed:
        print("  - %s  (%d bytes)" % (name, ea[name]['usize']))
    for name in changed:
        x, y = ea[name], eb[name]
        print("  ~ %s  (usize %d -> %d, csize %d -> %d, method %d -> %d)"
              % (name, x['usize'], y['usize'], x['csize'], y['csize'], x['method'], y['method']))
    print("summary: %d added, %d removed, %d changed, %d unchanged"
          % (len(added), len(removed), len(changed), len(unchanged)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('content', help='build a standalone mod archive to chain from PATCH.POD')
    p.add_argument('--mods', required=True, help='directory of mod files, POD-relative tree')
    p.add_argument('-o', '--out', required=True, help='output .POD path (not inside the game dir)')
    p.add_argument('--game', default=None, help='game directory: read-only, used to report what each file overrides')
    p.add_argument('--next', default=None, help='chain onward to a further archive')
    p.add_argument('--revision', type=int, default=1000)
    p.add_argument('--level', type=int, default=9)
    p.set_defaults(fn=cmd_content)

    p = sub.add_parser('install', help='copy a mod archive into the game dir and chain PATCH.POD to it')
    p.add_argument('--game', required=True)
    p.add_argument('--pod', required=True, help='the archive built by `content`')
    p.add_argument('--as', dest='name', default=CONTENT_NAME, help='name to install it under (default %s)' % CONTENT_NAME)
    p.add_argument('--host', default=PATCH_NAME, help='archive whose chain field points at it (default %s)' % PATCH_NAME)
    p.add_argument('--force', action='store_true', help='skip the ghost.exe-is-running check')
    p.add_argument('--keep-chain', action='store_true',
                   help='insert into the mount chain instead of truncating it: chain the installed '
                        'archive onward to whatever the replaced archive (or the host) used to mount '
                        'next, e.g. a mod manager\'s MODS.POD or ONLINE.POD')
    p.set_defaults(fn=cmd_install)

    p = sub.add_parser('uninstall', help='clear the chain field and delete the installed mod archive')
    p.add_argument('--game', required=True)
    p.add_argument('--as', dest='name', default=CONTENT_NAME)
    p.add_argument('--host', default=PATCH_NAME)
    p.add_argument('--force', action='store_true')
    p.set_defaults(fn=cmd_uninstall)

    p = sub.add_parser('build', help='build a full replacement PATCH.POD from the original + a mods dir')
    p.add_argument('--game', required=True, help='game directory containing the original PATCH.POD (read-only)')
    p.add_argument('--mods', required=True, help='directory of mod files, POD-relative tree (e.g. world\\firehouse.dante)')
    p.add_argument('-o', '--out', required=True, help='output .POD path (must NOT be under --game)')
    p.add_argument('--revision', type=int, default=None, help='default: keep original PATCH.POD revision')
    p.add_argument('--next', default=None, help='default: keep the original PATCH.POD chain field')
    p.add_argument('--level', type=int, default=9, help='zlib compression level for new/changed files')
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser('verify', help='list + integrity-check every stream in a POD')
    p.add_argument('archive')
    p.add_argument('-v', '--verbose', action='store_true')
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser('diff', help='compare two PODs: added / removed / changed entries')
    p.add_argument('a', metavar='A.POD')
    p.add_argument('b', metavar='B.POD')
    p.set_defaults(fn=cmd_diff)

    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()

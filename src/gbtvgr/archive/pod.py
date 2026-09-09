#!/usr/bin/env python3
"""
POD6 archives: list, extract, create, and the mount chain.

Mounting one archive mounts whatever its header chains to, in order, and mount
order is lookup priority -- which is what makes a mod archive load at all.

Commands:
  list    ARCHIVE.POD [-v]
  extract ARCHIVE.POD -o OUTDIR [-f SUBSTRING]
  create  OUT.POD -d INPUTDIR [--level N] [--next NAME.POD]
  chain   ARCHIVE.POD [NEXT.POD]        no NEXT: print the chain; "" clears it
"""
import argparse, os, struct, sys, zlib

MAGIC = b'POD6'
HDR = 0x80          # full header the engine reads
HDR_FIXED = 20      # magic + the four uint32 fields
CHAIN_OFF = 0x14
CHAIN_ROOM = HDR - CHAIN_OFF   # 108 bytes of storage in the header
CHAIN_MAX = 79      # engine strcpy's it into an 80-byte buffer
ENTRY = 24
ALIGN = 16
DEFLATE = 8
MAX_MOUNTS = 100    # engine's mounted-pod array is 100 slots


def _decode_chain(hdr):
    """Extract the NUL-terminated next-archive name from a raw header."""
    raw = hdr[CHAIN_OFF:HDR]
    end = raw.find(b'\0')
    return raw[:end if end >= 0 else None].decode('ascii', 'replace')


def _lookup_key(name):
    """Entry names as the engine compares them: backslashes, case-insensitive."""
    return name.lower().replace('/', '\\')


def encode_chain(name):
    """Encode a next-archive name into the fixed 108-byte header field."""
    if name is None:
        name = ''
    blob = name.encode('ascii')
    if len(blob) > CHAIN_MAX:
        raise ValueError("chain name %r is %d bytes; the engine's buffer holds %d + NUL"
                         % (name, len(blob), CHAIN_MAX))
    if b'\0' in blob:
        raise ValueError("chain name must not contain NUL")
    return blob.ljust(CHAIN_ROOM, b'\0')


class Pod:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'rb')
        self._by_name = None
        hdr = self.f.read(HDR)
        if hdr[:4] != MAGIC:
            raise ValueError("%s: not a POD6 archive (magic %r)" % (path, hdr[:4]))
        if len(hdr) < HDR:
            raise ValueError("%s: truncated header (%d of %d bytes); the engine reads 0x80"
                             % (path, len(hdr), HDR))
        self.count, self.revision, self.index_off, self.name_size = struct.unpack_from('<IIII', hdr, 4)
        self.next_pod = _decode_chain(hdr)
        self.f.seek(self.index_off)
        index = self.f.read(self.count * ENTRY)
        self.f.seek(self.index_off + self.count * ENTRY)
        names = self.f.read(self.name_size)
        self.entries = []
        for i in range(self.count):
            no, csize, off, usize, method, flags = struct.unpack_from('<IIIIII', index, i * ENTRY)
            end = names.find(b'\0', no)
            name = names[no:end if end >= 0 else None].decode('ascii', 'replace')
            self.entries.append(dict(name=name, csize=csize, off=off,
                                     usize=usize, method=method, flags=flags))

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def names(self):
        """Entry paths in index order, keeping the game's backslash separator."""
        return [e['name'] for e in self.entries]

    def entry(self, name):
        """Index entry for a path; separator and case are ignored, first match wins."""
        if self._by_name is None:
            self._by_name = {}
            for e in self.entries:
                self._by_name.setdefault(_lookup_key(e['name']), e)
        try:
            return self._by_name[_lookup_key(name)]
        except KeyError:
            raise KeyError("%s: no entry named %r" % (self.path, name)) from None

    @property
    def body_start(self):
        """Lowest file offset the archive body uses -- the header's usable size."""
        return min([self.index_off] + [e['off'] for e in self.entries])

    @property
    def chain_writable(self):
        """True if the 108-byte chain field can be patched without clobbering content."""
        return self.body_start >= HDR

    def read(self, e):
        """Decompress one entry, named either by path or by its index dict."""
        if not isinstance(e, dict):
            e = self.entry(e)
        self.f.seek(e['off'])
        blob = self.f.read(e['csize'])
        if e['method'] == DEFLATE:
            data = zlib.decompress(blob)
        elif e['method'] == 0:
            data = blob
        else:
            raise ValueError("%s: unknown compression method %d" % (e['name'], e['method']))
        if e['usize'] and len(data) != e['usize']:
            raise ValueError("%s: expected %d bytes, got %d" % (e['name'], e['usize'], len(data)))
        return data


def write_pod(out_path, entries, revision=1000, next_pod='', level=9):
    """Each entry is {'name': ...} plus one of path=, data= or copy=(Pod, entry).
    `copy` moves the stored stream verbatim, so a rebuild never recompresses."""
    chain = encode_chain(next_pod)
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    index = []
    with open(out_path, 'wb') as out:
        out.write(b'\0' * HDR)
        for entry in entries:
            if 'copy' in entry:
                src, se = entry['copy']
                src.f.seek(se['off'])
                blob = src.f.read(se['csize'])
                usize, method, flags = se['usize'], se['method'], se['flags']
            else:
                if 'data' in entry:
                    raw = entry['data']
                else:
                    with open(entry['path'], 'rb') as fh:
                        raw = fh.read()
                blob = zlib.compress(raw, level)
                usize, method, flags = len(raw), DEFLATE, 0
            while out.tell() % ALIGN:
                out.write(b'\0')
            off = out.tell()
            out.write(blob)
            index.append(dict(name=entry['name'], csize=len(blob), off=off,
                              usize=usize, method=method, flags=flags))

        while out.tell() % ALIGN:
            out.write(b'\0')
        index_off = out.tell()

        names, name_off = bytearray(), {}
        for e in index:
            name_off[e['name']] = len(names)
            names += e['name'].encode('ascii') + b'\0'
        while len(names) % 4:
            names.append(0)

        for e in index:
            out.write(struct.pack('<IIIIII', name_off[e['name']], e['csize'],
                                  e['off'], e['usize'], e['method'], e['flags']))
        out.write(bytes(names))
        out.seek(0)
        out.write(MAGIC)
        out.write(struct.pack('<IIII', len(index), revision, index_off, len(names)))
        out.write(chain)
    return index


# --- chain reading / writing -------------------------------------------------

def read_chain(path):
    """Return the next-archive name recorded in a POD's header ('' = end of chain)."""
    with open(path, 'rb') as fh:
        hdr = fh.read(HDR)
    if hdr[:4] != MAGIC:
        raise ValueError("%s: not a POD6 archive" % path)
    return _decode_chain(hdr)


def walk_chain(path, limit=MAX_MOUNTS):
    """Returns (names, note); names[0] is this POD's own basename.
    `note` is None on a clean end, else why the walk stopped early."""
    d = os.path.dirname(os.path.abspath(path))
    names, seen = [os.path.basename(path)], {os.path.basename(path).lower()}
    cur = path
    while True:
        try:
            nxt = read_chain(cur)
        except (OSError, ValueError) as exc:
            return names, str(exc)
        if not nxt:
            return names, None
        names.append(nxt)
        if nxt.lower() in seen:
            return names, "CYCLE -- the engine would loop forever mounting this"
        seen.add(nxt.lower())
        if len(names) > limit:
            return names, "longer than the engine's %d-slot mount array" % limit
        cur = os.path.join(d, nxt)
        if not os.path.isfile(cur):
            return names, "%s is not present (the engine silently stops here)" % nxt


def _rewrite_with_full_header(path, next_pod):
    """Rebuild an archive whose body starts before 0x80, preserving every stream."""
    src = Pod(path)
    tmp = path + ".chaintmp"
    write_pod(tmp, [dict(name=e['name'], copy=(src, e)) for e in src.entries],
              revision=src.revision, next_pod=next_pod)
    src.close()
    check = Pod(tmp)
    if [e['name'] for e in check.entries] != [e['name'] for e in src.entries]:
        check.close()
        os.unlink(tmp)
        raise ValueError("%s: rebuild lost entries; refusing to replace it" % path)
    check.close()
    os.replace(tmp, path)


def set_chain(path, next_pod, allow_game_dir=False, rewrite=True):
    """Point a POD at the next archive in its chain. Returns (old, new, how)."""
    self_name = os.path.basename(path)
    if next_pod and next_pod.lower() == self_name.lower():
        raise ValueError("%s cannot chain to itself -- the engine would loop forever" % self_name)
    encode_chain(next_pod)  # validate length/charset before touching the file

    d = os.path.dirname(os.path.abspath(path))
    if not allow_game_dir and os.path.exists(os.path.join(d, 'ghost.exe')):
        raise ValueError("%s sits next to ghost.exe; pass --allow-game-dir to modify a "
                         "POD inside the game directory" % path)

    if next_pod:
        target = os.path.join(d, next_pod)
        if os.path.isfile(target):
            names, note = walk_chain(target)
            if self_name.lower() in {n.lower() for n in names}:
                raise ValueError("chaining %s -> %s closes a cycle (%s); the engine would "
                                 "loop forever" % (self_name, next_pod, " -> ".join(names)))
            if note and note.startswith("CYCLE"):
                raise ValueError("%s already contains a cycle (%s)" % (next_pod, " -> ".join(names)))

    p = Pod(path)
    old, room, body = p.next_pod, p.chain_writable, p.body_start
    p.close()
    if old == next_pod:
        return old, next_pod, 'unchanged'
    if room:
        with open(path, 'r+b') as fh:
            fh.seek(CHAIN_OFF)
            fh.write(encode_chain(next_pod))
        how = 'stamped'
    else:
        if not rewrite:
            raise ValueError("%s has content at 0x%X, inside the 128-byte header; it must be "
                             "rebuilt to carry a chain field (drop --no-rewrite)"
                             % (path, body))
        _rewrite_with_full_header(path, next_pod)
        how = 'rebuilt'
    return old, next_pod, how


# --- subcommands -------------------------------------------------------------

def _chain_suffix(p):
    return ("  chain -> %s" % p.next_pod) if p.next_pod else ""


def cmd_list(a):
    with Pod(a.archive) as p:
        print("%s  POD6  %d files  revision %d  index @0x%X%s"
              % (os.path.basename(a.archive), p.count, p.revision, p.index_off, _chain_suffix(p)))
        if a.verbose:
            tot_c = tot_u = 0
            for e in p.entries:
                tot_c += e['csize']; tot_u += e['usize']
                print("  %-52s %10d -> %10d  m=%d" % (e['name'], e['csize'], e['usize'], e['method']))
            print("  %-52s %10d -> %10d" % ("TOTAL", tot_c, tot_u))
        else:
            ext = {}
            for e in p.entries:
                ext[os.path.splitext(e['name'])[1].lower()] = ext.get(os.path.splitext(e['name'])[1].lower(), 0) + 1
            for k, v in sorted(ext.items(), key=lambda x: -x[1]):
                print("  %-10s %d" % (k or '(none)', v))


def cmd_extract(a):
    n = bad = 0
    with Pod(a.archive) as p:
        for e in p.entries:
            if a.filter and a.filter.lower() not in e['name'].lower():
                continue
            dest = os.path.join(a.outdir, e['name'].replace('\\', os.sep))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                data = p.read(e)
            except (OSError, ValueError, zlib.error) as exc:
                print("  !! %s: %s" % (e['name'], exc), file=sys.stderr)
                bad += 1
                continue
            with open(dest, 'wb') as fh:
                fh.write(data)
            n += 1
            if a.verbose:
                print("  %s (%d bytes)" % (e['name'], len(data)))
    print("extracted %d file(s) to %s%s" % (n, a.outdir, ("  (%d failed)" % bad) if bad else ""))


def cmd_create(a):
    files = []
    for root, _, names in os.walk(a.inputdir):
        for nm in names:
            full = os.path.join(root, nm)
            rel = os.path.relpath(full, a.inputdir).replace(os.sep, '\\')
            files.append((rel, full))
    files.sort()
    if not files:
        sys.exit("no files under %s" % a.inputdir)

    index = write_pod(a.out, [dict(name=rel, path=full) for rel, full in files],
                      revision=a.revision, next_pod=a.next or '', level=a.level)
    print("wrote %s: %d files, %d bytes%s"
          % (a.out, len(index), os.path.getsize(a.out),
             ("  chain -> %s" % a.next) if a.next else ""))


def cmd_chain(a):
    if a.next is None:
        names, note = walk_chain(a.archive)
        print(" -> ".join(names) + (" -> (end)" if note is None else ""))
        if note:
            print("  note: %s" % note)
        return
    try:
        old, new, how = set_chain(a.archive, a.next, allow_game_dir=a.allow_game_dir,
                                  rewrite=not a.no_rewrite)
    except ValueError as exc:
        sys.exit("pod.py chain: %s" % exc)
    print("%s: chain %s -> %s  (%s)"
          % (os.path.basename(a.archive), old or "(none)", new or "(none)", how))
    names, note = walk_chain(a.archive)
    print("  " + " -> ".join(names) + (" -> (end)" if note is None else ""))
    if note:
        print("  note: %s" % note)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('list'); p.add_argument('archive'); p.add_argument('-v', '--verbose', action='store_true')
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser('extract'); p.add_argument('archive')
    p.add_argument('-o', '--outdir', required=True); p.add_argument('-f', '--filter')
    p.add_argument('-v', '--verbose', action='store_true'); p.set_defaults(fn=cmd_extract)

    p = sub.add_parser('create'); p.add_argument('out')
    p.add_argument('-d', '--inputdir', required=True)
    p.add_argument('--level', type=int, default=9)
    p.add_argument('--revision', type=int, default=1000)
    p.add_argument('--next', help='next archive in the chain, e.g. IMMORTAL.POD')
    p.set_defaults(fn=cmd_create)

    p = sub.add_parser('chain', help='print or set the next archive in a POD chain')
    p.add_argument('archive')
    p.add_argument('next', nargs='?', default=None,
                   help='name of the next archive; pass "" to clear. Omit to print the chain.')
    p.add_argument('--allow-game-dir', action='store_true',
                   help='permit modifying a POD that sits next to ghost.exe')
    p.add_argument('--no-rewrite', action='store_true',
                   help='fail instead of rebuilding an archive whose body starts before 0x80')
    p.set_defaults(fn=cmd_chain)

    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()

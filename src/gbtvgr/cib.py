#!/usr/bin/env python3
"""
.cib character definitions: inspect, clone, edit properties.

Tuning data is a flat property bag of ASCII key/value strings, so editing means
rewriting a value to occupy exactly the same number of bytes. This tool never
changes a file's length, so no other offset in the file is invalidated. A
character can hold several variants, and a key that recurs is disambiguated
with `name#<variant>` or, exactly, `name@<offset>` from `props`.

Commands:
    inspect <file.cib>                 header + all embedded strings, classified
    strings <file.cib>                 offset<TAB>kind<TAB>text, one per line
    clone <src.cib> <dst.cib> [--map]  byte copy; --map rewrites internal names
    props <file.cib>                   the property bag: variant, type, offset, name, value
    set <file.cib> name[#variant|@offset]=val [...] -o out.cib
                                       same-size in-place patch of numeric/bool properties
    diff <a.cib> <b.cib>               compare by named property, and raw bytes

`clone` without a map is a byte-for-byte copy: the engine resolves a character
by the path the .lvl gives, not by any name inside the file, so a copy under a
new path loads as a distinct character. `set` refuses rather than resize, so a
bool only toggles when both spellings already fit.

Exit codes: 0 ok, 1 bad args, 2 bad file.
"""
import os, struct, sys, re

STR_RE = re.compile(rb'[\x20-\x7e]{3,}')

# ---- property bag (props / set / diff) -------------------------------------

KEY_RE = re.compile(rb'^[a-zA-Z_][a-zA-Z0-9_]{3,}$')  # >=4 chars: shortest real key seen is "afx1"
NON_KEY_WORDS = {b'true', b'false', b'standard', b'none', b'null', b'void'}
MAX_TAG = 31          # real tags are 0-10 and 18; the rest is margin
MAX_VALUE_LEN = 200    # longest observed value string is well under this
# These appear exactly once per variant, which is how variant boundaries are
# inferred. Whichever one a given file has wins.
VARIANT_ANCHOR_KEYS = ('maxHitPoints', 'hasBlackSlime', 'freezeResistTime', 'remainFrozenTime')

def _prop_candidates(b):
    """Yield (offset, key_bytes) for every string token that looks like a property key
    (camelCase-ish identifier; excludes bare numbers/bools/enum words, which are values)."""
    for m in STR_RE.finditer(b):
        s = m.group()
        if KEY_RE.match(s) and s not in NON_KEY_WORDS and not re.fullmatch(rb'-?\d+(\.\d+)?', s):
            yield m.start(), s

def parse_properties(b):
    """Repeating `u32 typeTag; cstring key; cstring value; [cstring value2]`.
    Returns dicts sorted by file offset, each with a 0-based 'variant' index."""
    props = []
    for off, key in _prop_candidates(b):
        if off < 4:
            continue
        tag = struct.unpack_from('<I', b, off - 4)[0]
        if tag > MAX_TAG:
            continue
        key_end = off + len(key) + 1
        if key_end > len(b):
            continue
        nul = b.find(b'\x00', key_end)
        if nul == -1 or nul - key_end > MAX_VALUE_LEN:
            continue
        val = b[key_end:nul]
        if not all(0x20 <= c <= 0x7e for c in val):
            continue
        val2 = None
        v2start = nul + 1
        if v2start < len(b) and b[v2start] != 0:
            nul2 = b.find(b'\x00', v2start)
            if nul2 != -1 and nul2 - v2start <= MAX_VALUE_LEN:
                cand2 = b[v2start:nul2]
                if all(0x20 <= c <= 0x7e for c in cand2):
                    val2 = cand2.decode('latin-1')
        props.append({
            'key_off': off, 'tag': tag, 'key': key.decode('latin-1'),
            'val_off': key_end, 'val': val.decode('latin-1'), 'val2': val2,
        })
    props.sort(key=lambda p: p['key_off'])
    anchor = next((k for k in VARIANT_ANCHOR_KEYS if any(p['key'] == k for p in props)), None)
    variant, seen_anchor = 0, False
    for p in props:
        if anchor and p['key'] == anchor:
            if seen_anchor:
                variant += 1
            seen_anchor = True
        p['variant'] = variant
    return props

def _value_type(v):
    if v in ('true', 'false'):
        return 'bool'
    if re.fullmatch(r'-?\d+', v):
        return 'int'
    if re.fullmatch(r'-?(\d+\.\d*|\.\d+)([eE][-+]?\d+)?', v):
        return 'float'
    return 'string'

def _format_number(new_raw, target_len):
    """Fit a decimal number into exactly target_len ASCII bytes, or (None, None).
    exact=False means it had to be rounded or zero-padded to get there."""
    try:
        v = float(new_raw)
    except ValueError:
        return None, None
    candidates = []
    stripped = new_raw.strip()
    if re.fullmatch(r'-?\d+', stripped):
        candidates.append(stripped)
    for d in range(0, 12):
        candidates.append(f'{v:.{d}f}')
    exact = [c for c in candidates if len(c) == target_len and abs(float(c) - v) < 1e-9]
    if exact:
        return exact[0], True
    rounded = [c for c in candidates if len(c) == target_len]
    if rounded:
        return rounded[0], False
    shortest = min((c for c in candidates if len(c) <= target_len), key=len, default=None)
    if shortest is not None:
        sign, body = ('-', shortest[1:]) if shortest.startswith('-') else ('', shortest)
        pad = target_len - len(sign) - len(body)
        if pad >= 0:
            padded = sign + ('0' * pad) + body
            return padded, abs(float(padded) - v) < 1e-9
    return None, None

def _apply_set(props, key, variant_idx, target_off, new_raw):
    """Returns (edits, None) or (None, error). A key can recur even within one
    variant, so #<variant> may still be ambiguous and @<offset> is the exact pin."""
    matches = [p for p in props if p['key'] == key]
    if not matches:
        return None, f"no property named {key!r} in this file (see `cib.py props` for the list)"
    if target_off is not None:
        matches = [p for p in matches if p['val_off'] == target_off]
        if not matches:
            return None, f"{key!r} has no occurrence at offset {target_off:#06x} (see `cib.py props`)"
    elif variant_idx is not None:
        chosen = [p for p in matches if p['variant'] == variant_idx]
        if not chosen:
            have = sorted(set(p['variant'] for p in matches))
            return None, f"{key!r} has no occurrence in variant {variant_idx} (present in variant(s) {have})"
        matches = chosen
    if len(matches) > 1:
        if variant_idx is None:
            have = sorted(set(p['variant'] for p in matches))
            return None, f"{key!r} appears in {len(matches)} places, variant(s) {have}; disambiguate with {key}#<variant>=<value> or {key}@<offset>=<value>"
        offs = ', '.join(f"{p['val_off']:#06x}" for p in matches)
        return None, (f"{key!r} appears {len(matches)} times within variant {variant_idx} "
                       f"(offsets {offs}); disambiguate with {key}@<offset>=<value>")
    out = []
    for p in matches:
        old = p['val']
        vtype = _value_type(old)
        if vtype == 'string':
            return None, (f"{key!r} is a string/path property ({old!r}), not numeric/bool; "
                           f"`set` only patches numeric and boolean tuning values -- use `clone --map` to rename it")
        if vtype == 'bool':
            if new_raw not in ('true', 'false'):
                return None, f"{key!r} is a bool; value must be 'true' or 'false', got {new_raw!r}"
            if len(new_raw) != len(old):
                return None, (f"{key!r}: can't toggle {old!r} -> {new_raw!r} in place -- "
                               f"'true' (4 bytes) and 'false' (5 bytes) differ in length and this tool "
                               f"never changes file size")
            out.append((p, old, new_raw, True))
        else:
            new_text, exact = _format_number(new_raw, len(old))
            if new_text is None:
                return None, (f"{key!r}: can't represent {new_raw!r} in exactly {len(old)} byte(s) "
                               f"(old value {old!r}); try fewer significant digits or a smaller magnitude")
            out.append((p, old, new_text, exact))
    return out, None

def _read(path):
    with open(path, 'rb') as f: return f.read()

def _strings(b):
    """Yield (offset, text) for every printable run >= 3 chars."""
    for m in STR_RE.finditer(b):
        yield m.start(), m.group().decode('latin-1')

def _classify(s):
    """Best-effort kind label for an embedded string."""
    if re.search(r'\.(dfm|cib|cit|tfa|tfb|phys2|civh?)\b', s, re.I) or '\\' in s:
        return 'path'
    if re.fullmatch(r'-?\d+(\.\d+)?', s):
        return 'value'
    if s in ('true','false','standard','none','null','void'):
        return 'value'
    # property keys are camelCase identifiers, often long
    if re.fullmatch(r'[a-z][a-zA-Z0-9_]+', s) and len(s) >= 4:
        return 'key'
    return 'text'

def cmd_inspect(a):
    b = _read(a.file)
    if len(b) < 0x18:
        print("file too small to be a .cib", file=sys.stderr); return 2
    ver, varver = struct.unpack_from('<II', b, 0)
    hsh = b[8:0x18].hex()
    print(f"file:      {a.file}")
    print(f"size:      {len(b)} bytes")
    print(f"version:   {ver}")
    print(f"varVer:    {varver}")
    print(f"hash[16]:  {hsh}")
    print(f"strings:   {sum(1 for _ in _strings(b))}")
    print()
    print("embedded strings (offset  kind  text):")
    for off, s in _strings(b):
        print(f"  {off:#06x}  {_classify(s):5}  {s!r}")
    return 0

def cmd_strings(a):
    b = _read(a.file)
    for off, s in _strings(b):
        print(f"{off:#06x}\t{_classify(s)}\t{s}")
    return 0

def _load_map(path):
    m = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'): continue
            if '\t' not in line:
                print(f"bad map line (need Old<TAB>New): {line!r}", file=sys.stderr); return None
            old, new = line.split('\t', 1)
            if len(old.encode('latin-1')) != len(new.encode('latin-1')):
                print(f"length mismatch (offsets would break): {old!r} vs {new!r}", file=sys.stderr); return None
            m.append((old.encode('latin-1'), new.encode('latin-1')))
    return m

def cmd_clone(a):
    b = _read(a.src)
    out = bytearray(b)
    if a.map:
        m = _load_map(a.map)
        if m is None: return 1
        for old, new in m:
            n = out.count(old)
            if n == 0:
                print(f"warning: pattern not found in source: {old!r}", file=sys.stderr)
            out = out.replace(old, new)
            print(f"rewrite {n}x  {old!r} -> {new!r}")
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)) or '.', exist_ok=True)
    with open(a.dst, 'wb') as f:
        f.write(out)
    print(f"cloned {len(out)} bytes -> {a.dst}")
    return 0

def cmd_props(a):
    b = _read(a.file)
    if len(b) < 0x18:
        print("file too small to be a .cib", file=sys.stderr); return 2
    props = parse_properties(b)
    if a.key:
        props = [p for p in props if a.key.lower() in p['key'].lower()]
    if a.numeric:
        props = [p for p in props if _value_type(p['val']) in ('int', 'float', 'bool')]
    nvariants = 1 + max((p['variant'] for p in props), default=-1)
    print(f"file:    {a.file}")
    print(f"props:   {len(props)}  (variants: {nvariants if props else 0})")
    print()
    print(f"{'variant':7s} {'type':6s} {'offset':8s} {'name':40s} value")
    for p in props:
        vtype = _value_type(p['val'])
        val_disp = p['val'] + (f"  (2nd: {p['val2']!r})" if p['val2'] else '')
        print(f"{p['variant']:<7d} {vtype:6s} {p['val_off']:#08x} {p['key']:40s} {val_disp}")
    return 0

def cmd_set(a):
    src = _read(a.file)
    if len(src) < 0x18:
        print("file too small to be a .cib", file=sys.stderr); return 2
    props = parse_properties(src)
    out = bytearray(src)
    changes = []
    for assignment in a.assignments:
        if '=' not in assignment:
            print(f"bad assignment (need name=value or name#variant=value): {assignment!r}", file=sys.stderr)
            return 1
        namepart, new_raw = assignment.split('=', 1)
        variant_idx = None
        target_off = None
        if '@' in namepart:
            namepart, offstr = namepart.split('@', 1)
            try:
                target_off = int(offstr, 0)  # accepts "0x..." or decimal
            except ValueError:
                print(f"bad offset in {assignment!r}: {offstr!r}", file=sys.stderr)
                return 1
        elif '#' in namepart:
            namepart, vidx = namepart.split('#', 1)
            try:
                variant_idx = int(vidx)
            except ValueError:
                print(f"bad variant index in {assignment!r}: {vidx!r}", file=sys.stderr)
                return 1
        result, err = _apply_set(props, namepart, variant_idx, target_off, new_raw)
        if err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        changes.extend(result)
    for p, old, new_text, exact in changes:
        new_bytes = new_text.encode('latin-1')
        out[p['val_off']:p['val_off'] + len(new_bytes)] = new_bytes
    if len(out) != len(src):
        print("internal error: patch changed file length, refusing to write", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or '.', exist_ok=True)
    with open(a.output, 'wb') as f:
        f.write(out)
    for p, old, new_text, exact in changes:
        note = '' if exact else '  (rounded/padded to fit same byte length)'
        print(f"  {p['key']} (variant {p['variant']}, offset {p['val_off']:#06x}): {old!r} -> {new_text!r}{note}")
    print(f"wrote {len(out)} bytes -> {a.output}")
    return 0

def cmd_diff(a):
    ba, bb = _read(a.a), _read(a.b)
    if len(ba) != len(bb):
        print(f"size differs: {a.a} = {len(ba)} bytes, {a.b} = {len(bb)} bytes")
    else:
        print(f"same size: {len(ba)} bytes")
    props_a, props_b = parse_properties(ba), parse_properties(bb)
    # Group by (key, variant) into an ordered list: a key can legitimately recur
    # within one variant, and collapsing those would lose an edit.
    groups_a, groups_b = {}, {}
    for p in props_a:
        groups_a.setdefault((p['key'], p['variant']), []).append(p)
    for p in props_b:
        groups_b.setdefault((p['key'], p['variant']), []).append(p)
    covered = set()
    for p in props_a:
        covered.update(range(p['val_off'], p['val_off'] + len(p['val'].encode('latin-1'))))
    prop_changes = 0
    print()
    for k in sorted(set(groups_a) | set(groups_b)):
        name, variant = k
        la, lb = groups_a.get(k, []), groups_b.get(k, [])
        for i in range(max(len(la), len(lb))):
            va = la[i] if i < len(la) else None
            vb = lb[i] if i < len(lb) else None
            tag = f"{name} (variant {variant}" + (f", occurrence {i}" if max(len(la), len(lb)) > 1 else "") + ")"
            if va is None:
                print(f"  + {tag}: (absent) -> {vb['val']!r}")
                prop_changes += 1
            elif vb is None:
                print(f"  - {tag}: {va['val']!r} -> (absent)")
                prop_changes += 1
            elif va['val'] != vb['val']:
                print(f"  ~ {tag}, offset {va['val_off']:#06x}: {va['val']!r} -> {vb['val']!r}")
                prop_changes += 1
    n = min(len(ba), len(bb))
    raw_diff_bytes = 0
    unexplained = []
    i = 0
    while i < n:
        if ba[i] != bb[i]:
            j = i
            while j < n and ba[j] != bb[j]:
                j += 1
            raw_diff_bytes += j - i
            if not all(k in covered for k in range(i, j)):
                unexplained.append((i, j))
            i = j
        else:
            i += 1
    print()
    print(f"{prop_changes} named-property change(s); {raw_diff_bytes} differing byte(s) total "
          f"in the shared {n}-byte region")
    if unexplained:
        print("byte range(s) that differ but are NOT inside a recognized property value:")
        for i, j in unexplained:
            print(f"  [{i:#06x}..{j:#06x})  old={ba[i:j].hex()}  new={bb[i:j].hex()}")
    elif raw_diff_bytes:
        print("every differing byte falls inside a named property value listed above.")
    return 0

def main():
    ap = __import__('argparse').ArgumentParser(description=__doc__, formatter_class=__import__('argparse').RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('inspect'); p.add_argument('file'); p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser('strings'); p.add_argument('file'); p.set_defaults(fn=cmd_strings)
    p = sub.add_parser('clone'); p.add_argument('src'); p.add_argument('dst'); p.add_argument('--map'); p.set_defaults(fn=cmd_clone)
    p = sub.add_parser('props', help='list the property bag (tuning values) by name, type, offset, variant')
    p.add_argument('file')
    p.add_argument('--key', help='only show properties whose name contains this substring (case-insensitive)')
    p.add_argument('--numeric', action='store_true', help='only show int/float/bool properties')
    p.set_defaults(fn=cmd_props)
    p = sub.add_parser('set', help='patch one or more named numeric/bool properties in place (same file size)')
    p.add_argument('file')
    p.add_argument('assignments', nargs='+', metavar='name[#variant|@offset]=value')
    p.add_argument('-o', '--output', required=True)
    p.set_defaults(fn=cmd_set)
    p = sub.add_parser('diff', help='compare two .cib files by named property, plus a raw byte diff')
    p.add_argument('a'); p.add_argument('b'); p.set_defaults(fn=cmd_diff)
    a = ap.parse_args()
    sys.exit(a.fn(a))

if __name__ == '__main__':
    main()

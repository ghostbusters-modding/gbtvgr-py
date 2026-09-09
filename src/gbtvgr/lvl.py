#!/usr/bin/env python3
"""
.lvl actor tables: inspect, validate, diff, add/remove actors.

Plain-text level files, CRLF throughout, one statement per line, indented with
exactly `depth` tab characters. The writer regenerates indentation from tree
depth and still reproduces the shipped bytes exactly.

Commands:
  stats <file.lvl ...>                 counts per section
  actors <file.lvl>                    the actor table
  schema <corpus>                      per-class field schemas
  validate <file.lvl>                  invariants the engine enforces
  diff <a.lvl> <b.lvl>                 structural diff
  add-actor <file.lvl> ...             append an actor
  remove-actor <file.lvl> <name>       drop an actor and its references
  roundtrip-test <dir>                 parse->write every .lvl under dir
"""
import argparse, collections, copy, json, os, re, sys

from .archive import pod as _pod  # for corpus/library args that name a .POD

_HERE = os.path.dirname(os.path.abspath(__file__))
# $DANTE_API_JSON overrides the packaged table, so another build's can be
# dropped in without a reinstall.
DEFAULT_DANTE_API = os.environ.get(
    'DANTE_API_JSON', os.path.join(_HERE, 'data', 'dante_api.json'))

WINSEP = r'\\'  # the file's own path separator convention: two literal backslash chars


class LvlError(Exception):
    pass


# ============================================================================
# data model
# ============================================================================

def _read(path):
    with open(path, 'rb') as fh:
        return fh.read()


class KV:
    """A single ``key = value`` line. ``value`` is the raw text after " = ", verbatim."""
    __slots__ = ('key', 'value')

    def __init__(self, key, value):
        self.key = key
        self.value = value


class ListF:
    """A ``key = < ... >`` block of raw item lines; no shipped item nests or holds '='.
    Items carry their own extra_tabs so odd indentation round-trips as written."""
    __slots__ = ('key', 'items')

    def __init__(self, key, items):
        self.key = key
        self.items = items  # list[(int, str)]


class Block:
    """A ``<tag>`` ... ``</tag>`` block. The top-level document is a Block with tag=None."""
    __slots__ = ('tag', 'children')

    def __init__(self, tag, children=None):
        self.tag = tag
        self.children = children if children is not None else []


TAG_RE = re.compile(r'^<([^<>/][^<>]*)>$')


# ============================================================================
# parser
# ============================================================================

def parse_lvl(data: bytes) -> Block:
    text = data.decode('latin-1')
    if '\r\n' not in text:
        raise LvlError("not a .lvl file (no CRLF line endings found)")
    if not text.endswith('\r\n'):
        raise LvlError("file does not end with CRLF")
    raw_lines = text.split('\r\n')
    raw_lines.pop()  # trailing '' after the final CRLF
    if any('\n' in l or '\r' in l for l in raw_lines):
        raise LvlError("bare LF or CR found inside a line (mixed line endings)")

    lines = []
    for ln in raw_lines:
        n = 0
        while n < len(ln) and ln[n] == '\t':
            n += 1
        lines.append((n, ln[n:]))

    pos = [0]

    def parse_children(depth):
        children = []
        while pos[0] < len(lines):
            ind, rest = lines[pos[0]]
            if ind != depth:
                break
            if rest.startswith('</'):
                break
            m_tag = TAG_RE.match(rest)
            if m_tag:
                tag = m_tag.group(1)
                pos[0] += 1
                sub = parse_children(depth + 1)
                if pos[0] >= len(lines):
                    raise LvlError("line %d: unterminated block <%s>" % (pos[0], tag))
                cind, crest = lines[pos[0]]
                if cind != depth or crest != '</%s>' % tag:
                    raise LvlError("line %d: expected </%s>, got %r at depth %d"
                                    % (pos[0] + 1, tag, crest, cind))
                pos[0] += 1
                children.append(Block(tag, sub))
                continue
            sep = rest.find(' = ')
            if sep == -1:
                raise LvlError("line %d: no ' = ' found in %r" % (pos[0] + 1, rest))
            key = rest[:sep]
            value = rest[sep + 3:]
            if value == '<':
                pos[0] += 1
                items = []
                while True:
                    if pos[0] >= len(lines):
                        raise LvlError("unterminated list %r" % key)
                    iind, irest = lines[pos[0]]
                    if iind == depth and irest == '>':
                        pos[0] += 1
                        break
                    if iind < depth + 1:
                        raise LvlError("line %d: unterminated list %r (bad indent %d)"
                                        % (pos[0] + 1, key, iind))
                    items.append((iind - (depth + 1), irest))
                    pos[0] += 1
                children.append(ListF(key, items))
            else:
                children.append(KV(key, value))
                pos[0] += 1
        return children

    top = parse_children(0)
    if pos[0] != len(lines):
        ind, rest = lines[pos[0]]
        raise LvlError("line %d: trailing/unexpected content at depth %d: %r"
                        % (pos[0] + 1, ind, rest))
    return Block(None, top)


def serialize_lvl(doc: Block) -> bytes:
    out = []

    def emit(children, depth):
        indent = '\t' * depth
        for node in children:
            if isinstance(node, Block):
                out.append('%s<%s>' % (indent, node.tag))
                emit(node.children, depth + 1)
                out.append('%s</%s>' % (indent, node.tag))
            elif isinstance(node, ListF):
                out.append('%s%s = <' % (indent, node.key))
                item_indent = '\t' * (depth + 1)
                for extra, text in node.items:
                    out.append('%s%s%s' % (item_indent, '\t' * extra, text))
                out.append('%s>' % indent)
            elif isinstance(node, KV):
                out.append('%s%s = %s' % (indent, node.key, node.value))
            else:
                raise LvlError("unknown node type %r" % (node,))

    emit(doc.children, 0)
    text = '\r\n'.join(out) + '\r\n'
    return text.encode('latin-1')


# ============================================================================
# tree helpers
# ============================================================================

def top_kv(doc, key):
    for c in doc.children:
        if isinstance(c, KV) and c.key == key:
            return c
    return None


def top_list(doc, key):
    for c in doc.children:
        if isinstance(c, ListF) and c.key == key:
            return c
    return None


def top_block(doc, tag):
    for c in doc.children:
        if isinstance(c, Block) and c.tag == tag:
            return c
    return None


def actor_class_map(doc):
    """name -> class, from <actor-list>."""
    al = top_block(doc, 'actor-list')
    m = collections.OrderedDict()
    if al:
        for c in al.children:
            if isinstance(c, KV):
                m[c.key] = c.value
    return m


def actor_fields(actor_block):
    """dict of this actor's own top-level key/value fields (KV children only)."""
    return {c.key: c.value for c in actor_block.children if isinstance(c, KV)}


def iter_actor_field_names(actor_block):
    """Every field name anywhere in this actor's block, nested sub-blocks included."""
    def walk(node):
        for c in node.children:
            if isinstance(c, (KV, ListF)):
                yield c.key
            elif isinstance(c, Block):
                yield from walk(c)
    yield from walk(actor_block)


def dep_set(doc):
    lf = top_list(doc, 'Dependencies')
    return set(text for _, text in lf.items) if lf else set()


VEC_RE = re.compile(r'^-?\d+(\.\d+)?(, -?\d+(\.\d+)?){2}$')


def looks_like_vector(v):
    return bool(VEC_RE.match(v))


def dep_path_for_cit(cit):
    if cit.lower().startswith('data' + WINSEP):
        return cit
    return 'data' + WINSEP + cit


def civh_family_entry(dep_form, deps):
    """The .civh Dependencies entry beside dep_form, or None. The directory match is
    case-insensitive: shipped files mix 'Flyer_Med' and 'flyer_med'."""
    dirpart = dep_form.rsplit(WINSEP, 1)[0].lower()
    for d in deps:
        if d.lower().endswith('.civh') and d.rsplit(WINSEP, 1)[0].lower() == dirpart:
            return d
    return None


# ============================================================================
# corpus / library loading (dir of .lvl, a single .lvl, or a .POD)
# ============================================================================

def load_corpus_raw(path):
    """Returns list of (display_name, raw_bytes) for every .lvl found at path."""
    results = []
    if os.path.isdir(path):
        for root, _, names in os.walk(path):
            for n in sorted(names):
                if n.lower().endswith('.lvl'):
                    p = os.path.join(root, n)
                    results.append((os.path.relpath(p, path), _read(p)))
    elif path.lower().endswith('.pod'):
        with _pod.Pod(path) as p:
            for e in p.entries:
                if e['name'].lower().endswith('.lvl'):
                    results.append((e['name'], p.read(e)))
    elif path.lower().endswith('.lvl'):
        results.append((os.path.basename(path), _read(path)))
    else:
        raise LvlError("%s: not a directory, .lvl file, or .POD archive" % path)
    results.sort(key=lambda t: t[0])
    return results


# ============================================================================
# dante_api.json (registered-property schema)
# ============================================================================

def load_dante_api(path):
    with open(path) as fh:
        d = json.load(fh)
    by_field = collections.defaultdict(list)
    for p in d['properties']:
        if p.get('field'):
            by_field[p['field']].append(p)
    return d, by_field


_SUFFIX_RE = re.compile(r'^(.+)_(\d+)$')


def lookup_field(by_field, field):
    """Match a .lvl field name against the registered property table, which spells
    array slots ``base[N]`` where the .lvl text spells them ``base_N``."""
    if field in by_field:
        return by_field[field], field
    m = _SUFFIX_RE.match(field)
    if m:
        base, idx = m.group(1), m.group(2)
        if base in by_field:
            return by_field[base], base
        bracket = '%s[%s]' % (base, idx)
        if bracket in by_field:
            return by_field[bracket], bracket
        m2 = _SUFFIX_RE.match(base)
        if m2 and m2.group(1) in by_field:
            return by_field[m2.group(1)], m2.group(1)
    return None, None


SIZE_HINT = {1: 'bool/byte', 4: 'int/float/enum', 8: 'float/pointer/string(8)',
             12: 'Vector3', 24: 'vector-ish(24)', 32: 'String(32)', 40: 'event(40)'}


def size_hint(ptype):
    return SIZE_HINT.get(ptype, 'size %s' % ptype)


# ============================================================================
# number formatting (for add-actor's --pos)
# ============================================================================

def fmt_num(x):
    f = float(x)
    if f.is_integer():
        return str(int(f))
    s = '%.6f' % f
    s = s.rstrip('0').rstrip('.')
    return s


# ============================================================================
# subcommands
# ============================================================================

def cmd_stats(args):
    doc = parse_lvl(_read(args.file))
    deps = top_list(doc, 'Dependencies')
    di = top_list(doc, 'Dante-Includes')
    al = top_block(doc, 'actor-list')
    classes = collections.Counter()
    if al:
        for c in al.children:
            if isinstance(c, KV):
                classes[c.value] += 1
    version = top_kv(doc, 'version')
    setfn = top_kv(doc, 'set-filename')
    total = sum(classes.values())
    print(args.file)
    print("  version         = %s" % (version.value if version else '?'))
    print("  set-filename    = %s" % (setfn.value if setfn else '?'))
    print("  dependencies    = %d" % (len(deps.items) if deps else 0))
    print("  dante-includes  = %d" % (len(di.items) if di else 0))
    print("  actors          = %d  (%d distinct classes)" % (total, len(classes)))
    for cls, n in classes.most_common():
        print("    %5d  %s" % (n, cls))


def cmd_actors(args):
    doc = parse_lvl(_read(args.file))
    cls_map = actor_class_map(doc)
    actors_block = top_block(doc, 'actors')
    rows = []
    for a in (actors_block.children if actors_block else []):
        if not isinstance(a, Block):
            continue
        name = a.tag
        cls = cls_map.get(name, '?')
        if args.cls and cls != args.cls:
            continue
        if args.name and args.name.lower() not in name.lower():
            continue
        f = actor_fields(a)
        key_field = f.get('charInfoFilename') or f.get('definitionName') or ''
        rows.append((name, cls, f.get('pos', '-'), f.get('createStatus', '-'), key_field))
    if not rows:
        print("(no matching actors)")
        return
    w0 = max(len(r[0]) for r in rows)
    w1 = max(len(r[1]) for r in rows)
    w2 = max(len(r[2]) for r in rows)
    print("%-*s  %-*s  %-*s  %-2s  %s" % (w0, "name", w1, "class", w2, "pos", "cs", "cit/def"))
    for name, cls, pos, cs, kf in rows:
        print("%-*s  %-*s  %-*s  %-2s  %s" % (w0, name, w1, cls, w2, pos, cs, kf))
    print("(%d actors)" % len(rows))


def cmd_roundtrip_test(args):
    corpus = load_corpus_raw(args.path)
    if not corpus:
        print("no .lvl files found at %s" % args.path)
        return 1
    ok = 0
    fails = []
    for name, raw in corpus:
        try:
            doc = parse_lvl(raw)
            out = serialize_lvl(doc)
        except LvlError as e:
            fails.append((name, "parse error: %s" % e))
            continue
        if out == raw:
            ok += 1
            if args.verbose:
                print("OK    %s  (%d bytes)" % (name, len(raw)))
        else:
            i = 0
            m = min(len(out), len(raw))
            while i < m and out[i] == raw[i]:
                i += 1
            fails.append((name, "byte mismatch at offset %d (orig %d bytes, rebuilt %d bytes)"
                          % (i, len(raw), len(out))))
    for name, msg in fails:
        print("FAIL  %s: %s" % (name, msg))
    print("roundtrip: %d/%d passed" % (ok, len(corpus)))
    return 0 if not fails else 1


def cmd_schema(args):
    d, by_field = load_dante_api(args.dante_api)
    corpus_usage = collections.defaultdict(collections.Counter)
    all_usage = collections.Counter()
    if args.corpus:
        for name, raw in load_corpus_raw(args.corpus):
            try:
                doc = parse_lvl(raw)
            except LvlError as e:
                print("skipping %s: %s" % (name, e), file=sys.stderr)
                continue
            cls_map = actor_class_map(doc)
            actors_block = top_block(doc, 'actors')
            for a in (actors_block.children if actors_block else []):
                if not isinstance(a, Block):
                    continue
                cls = cls_map.get(a.tag, '?')
                for f in iter_actor_field_names(a):
                    corpus_usage[cls][f] += 1
                    all_usage[f] += 1

    if args.cls:
        fields = corpus_usage.get(args.cls)
        if fields:
            print("class %s: %d distinct fields, %d occurrences across corpus"
                  % (args.cls, len(fields), sum(fields.values())))
            for f, n in fields.most_common():
                props, matched_as = lookup_field(by_field, f)
                if props:
                    owners = sorted(set(p['owner'] for p in props))
                    print("  %6d  %-32s owner=%-14s size=%-16s disp=%r"
                          % (n, f, ','.join(owners), size_hint(props[0]['ptype']), props[0]['disp']))
                else:
                    print("  %6d  %-32s ** not in dante_api.json **" % (n, f))
        else:
            owned = sorted((p for p in d['properties'] if p['owner'] == args.cls),
                            key=lambda p: p['field'])
            if not owned and not args.corpus:
                print("class %s: no --corpus given, and no fields registered directly on "
                      "this class in dante_api.json (fields may be inherited -- pass "
                      "--corpus to see actually-observed fields)" % args.cls)
            else:
                print("class %s: no corpus usage (or class not present in --corpus); "
                      "%d fields registered directly on this class in dante_api.json:"
                      % (args.cls, len(owned)))
                for p in owned:
                    print("    %-32s size=%-16s disp=%r" % (p['field'], size_hint(p['ptype']), p['disp']))
    else:
        if not args.corpus:
            print("no --corpus given; pass a directory of .lvl files or a .POD to compute "
                  "observed-usage coverage. Showing dante_api.json summary only.")
            print("%d classes have directly-registered properties, %d properties total"
                  % (len(set(p['owner'] for p in d['properties'])), len(d['properties'])))
            return
        total = len(all_usage)
        found = sum(1 for f in all_usage if lookup_field(by_field, f)[0])
        print("%d actor classes observed across corpus" % len(corpus_usage))
        print("%d distinct field names observed; %d matched in dante_api.json (%.1f%%)"
              % (total, found, 100.0 * found / total if total else 0.0))
        print("unmatched field names (%d):" % (total - found))
        for f in sorted(all_usage):
            if not lookup_field(by_field, f)[0]:
                print("  %-32s (%d occurrences)" % (f, all_usage[f]))


def cmd_validate(args):
    raw = _read(args.file)
    doc = parse_lvl(raw)
    problems = []

    # A lone backslash reads as an escape and kills the whole level load, so
    # every backslash in the file has to be doubled.
    for ln, line in enumerate(raw.decode('latin1').split('\r\n'), 1):
        if '\\' in line.replace('\\\\', ''):
            problems.append('line %d: single (unescaped) backslash: %r'
                            % (ln, line.strip()[:70]))

    deps = dep_set(doc)
    name_to_class = collections.OrderedDict()
    dup_list_names = []
    al = top_block(doc, 'actor-list')
    if al:
        for c in al.children:
            if isinstance(c, KV):
                if c.key in name_to_class:
                    dup_list_names.append(c.key)
                name_to_class[c.key] = c.value
    for n in dup_list_names:
        problems.append("duplicate actor name in <actor-list>: %s" % n)

    by_field = {}
    if os.path.exists(args.dante_api):
        _, by_field = load_dante_api(args.dante_api)

    actors_block = top_block(doc, 'actors')
    seen_tags = set()
    unknown_field_counts = collections.Counter()
    for a in (actors_block.children if actors_block else []):
        if not isinstance(a, Block):
            continue
        if a.tag in seen_tags:
            problems.append("duplicate actor block: <%s>" % a.tag)
        seen_tags.add(a.tag)
        if a.tag not in name_to_class:
            problems.append("<%s> has no <actor-list> entry" % a.tag)

        f = actor_fields(a)
        cit = f.get('charInfoFilename')
        if cit and cit != '""':
            depform = dep_path_for_cit(cit)
            if depform not in deps:
                problems.append("%s: charInfoFilename %r -> missing Dependencies entry %r"
                                 % (a.tag, cit, depform))
            elif civh_family_entry(depform, deps) is None:
                problems.append("%s: charInfoFilename %r -> no sibling .civh in Dependencies "
                                 "(dir %r)" % (a.tag, cit, depform.rsplit(WINSEP, 1)[0]))

        for vk in ('pos', 'orient'):
            if vk in f and not looks_like_vector(f[vk]):
                problems.append("%s: malformed vector %s = %r" % (a.tag, vk, f[vk]))

        if by_field:
            for fld in iter_actor_field_names(a):
                if lookup_field(by_field, fld)[0] is None:
                    unknown_field_counts[fld] += 1

    for n in name_to_class:
        if n not in seen_tags:
            problems.append("<actor-list> entry %r has no matching <%s> block in <actors>" % (n, n))

    print(args.file)
    if not problems:
        print("  OK  -- dependency invariant, duplicate names, vector formatting all pass")
    else:
        for p in problems:
            print("  FAIL: %s" % p)
    print("  %d distinct field names not in dante_api.json (informational -- base actor "
          "bookkeeping fields like name/pos/orient aren't registered via the reflection "
          "system)" % len(unknown_field_counts))
    if args.verbose:
        for fld, n in unknown_field_counts.most_common():
            print("    %-32s x%d" % (fld, n))
    return 1 if problems else 0


def cmd_diff(args):
    doc_a = parse_lvl(_read(args.a))
    doc_b = parse_lvl(_read(args.b))

    def amap(doc):
        ab = top_block(doc, 'actors')
        cls_map = actor_class_map(doc)
        m = {}
        if ab:
            for a in ab.children:
                if isinstance(a, Block):
                    m[a.tag] = (cls_map.get(a.tag, '?'), actor_fields(a))
        return m

    am_a, am_b = amap(doc_a), amap(doc_b)
    added = sorted(set(am_b) - set(am_a))
    removed = sorted(set(am_a) - set(am_b))
    common = sorted(set(am_a) & set(am_b))
    changed = []
    for n in common:
        (cls_a, fa), (cls_b, fb) = am_a[n], am_b[n]
        diffs = {}
        if cls_a != cls_b:
            diffs['<class>'] = (cls_a, cls_b)
        for k in set(fa) | set(fb):
            va, vb = fa.get(k, '<absent>'), fb.get(k, '<absent>')
            if va != vb:
                diffs[k] = (va, vb)
        if diffs:
            changed.append((n, diffs))

    da, db = dep_set(doc_a), dep_set(doc_b)
    dep_added, dep_removed = sorted(db - da), sorted(da - db)

    print("--- %s" % args.a)
    print("+++ %s" % args.b)
    if added:
        print("actors added (%d):" % len(added))
        for n in added:
            print("  + %-32s class=%s" % (n, am_b[n][0]))
    if removed:
        print("actors removed (%d):" % len(removed))
        for n in removed:
            print("  - %-32s class=%s" % (n, am_a[n][0]))
    if changed:
        print("actors changed (%d):" % len(changed))
        for n, diffs in changed:
            print("  ~ %s" % n)
            for k in sorted(diffs):
                va, vb = diffs[k]
                print("      %-24s %r -> %r" % (k, va, vb))
    if dep_added or dep_removed:
        print("dependencies: +%d -%d" % (len(dep_added), len(dep_removed)))
        for d in dep_added:
            print("  + %s" % d)
        for d in dep_removed:
            print("  - %s" % d)
    if not (added or removed or changed or dep_added or dep_removed):
        print("no semantic differences (actors, dependencies)")


def _find_template(doc, doc_path, cls, template_name, library):
    """Returns (template_Block, source_doc) or raises LvlError."""
    actors_block = top_block(doc, 'actors')
    cls_map = actor_class_map(doc)

    def search(d):
        ab = top_block(d, 'actors')
        cmap = actor_class_map(d)
        if not ab:
            return None
        if template_name:
            for a in ab.children:
                if isinstance(a, Block) and a.tag == template_name:
                    return a
            return None
        for a in ab.children:
            if isinstance(a, Block) and cmap.get(a.tag) == cls:
                return a
        return None

    hit = search(doc)
    if hit is not None:
        return hit, doc, doc_path

    if library:
        for name, raw in load_corpus_raw(library):
            try:
                d2 = parse_lvl(raw)
            except LvlError:
                continue
            hit = search(d2)
            if hit is not None:
                return hit, d2, name

    if template_name:
        raise LvlError("no actor named %r found in %s%s" %
                        (template_name, doc_path, (" or --library %s" % library) if library else ""))
    raise LvlError("no actor of class %r found in %s%s" %
                    (cls, doc_path, (" or --library %s" % library) if library else ""))


def cmd_add_actor(args):
    raw = _read(args.file)
    doc = parse_lvl(raw)

    template, src_doc, src_name = _find_template(doc, args.file, args.cls, args.template, args.library)
    src_cls = actor_class_map(src_doc).get(template.tag, args.cls)
    cls = args.cls or src_cls
    if args.cls and src_cls != args.cls:
        print("note: template %r is class %s, not requested class %s -- using requested class"
              % (template.tag, src_cls, args.cls), file=sys.stderr)

    if args.name in actor_class_map(doc):
        raise LvlError("%s already has an actor named %r" % (args.file, args.name))

    new_actor = copy.deepcopy(template)
    new_actor.tag = args.name
    for c in new_actor.children:
        if isinstance(c, KV) and c.key == 'name':
            c.value = args.name
            break

    if args.pos is not None:
        x, y, z = args.pos
        posval = '%s, %s, %s' % (fmt_num(x), fmt_num(y), fmt_num(z))
        found = False
        for c in new_actor.children:
            if isinstance(c, KV) and c.key == 'pos':
                c.value = posval
                found = True
                break
        if not found:
            new_actor.children.insert(0, KV('pos', posval))

    overrides = {}
    for kv in (args.set or []):
        if '=' not in kv:
            raise LvlError("--set expects key=value, got %r" % kv)
        k, v = kv.split('=', 1)
        overrides[k] = v
    for k, v in overrides.items():
        found = False
        for c in new_actor.children:
            if isinstance(c, KV) and c.key == k:
                c.value = v
                found = True
                break
        if not found:
            print("note: %r is not an existing field on the template -- appending" % k, file=sys.stderr)
            new_actor.children.append(KV(k, v))

    actors_block = top_block(doc, 'actors')
    if actors_block is None:
        raise LvlError("%s has no <actors> block" % args.file)
    actors_block.children.append(new_actor)

    al = top_block(doc, 'actor-list')
    if al is None:
        raise LvlError("%s has no <actor-list> block" % args.file)
    al.children.append(KV(args.name, cls))

    di = top_list(doc, 'Dante-Includes')
    if di is not None:
        di.items.append((0, 'extern %s %s;' % (cls, args.name)))

    avl = top_block(doc, 'actor-version-list')
    if avl is not None and not any(isinstance(c, KV) and c.key == cls for c in avl.children):
        src_avl = top_block(src_doc, 'actor-version-list')
        ver = None
        if src_avl:
            for c in src_avl.children:
                if isinstance(c, KV) and c.key == cls:
                    ver = c.value
                    break
        if ver is not None:
            avl.children.append(KV(cls, ver))
            print("note: added actor-version-list entry %s = %s (copied from %s)" % (cls, ver, src_name),
                  file=sys.stderr)
        else:
            print("warning: class %s has no actor-version-list entry anywhere -- editor may "
                  "not recognize it" % cls, file=sys.stderr)

    deps_lf = top_list(doc, 'Dependencies')
    deps = dep_set(doc)
    new_fields = actor_fields(new_actor)
    cit = new_fields.get('charInfoFilename')
    added_deps = []
    if cit and cit != '""' and deps_lf is not None:
        depform = dep_path_for_cit(cit)
        if depform not in deps:
            deps_lf.items.append((0, depform))
            deps.add(depform)
            added_deps.append(depform)
        civh = civh_family_entry(depform, deps)
        if civh is None:
            src_deps = dep_set(src_doc)
            src_civh = civh_family_entry(depform, src_deps)
            if src_civh is not None:
                deps_lf.items.append((0, src_civh))
                deps.add(src_civh)
                added_deps.append(src_civh)
            else:
                print("warning: no .civh found for %r even in template source %s -- level may "
                      "fail to prepare" % (cit, src_name), file=sys.stderr)

    out = serialize_lvl(doc)
    with open(args.output, 'wb') as fh:
        fh.write(out)
    print("wrote %s: added actor %r (class %s, template %r from %s)"
          % (args.output, args.name, cls, template.tag, src_name))
    if added_deps:
        print("  added Dependencies: %s" % ', '.join(added_deps))


def cmd_remove_actor(args):
    raw = _read(args.file)
    doc = parse_lvl(raw)

    actors_block = top_block(doc, 'actors')
    al = top_block(doc, 'actor-list')
    if actors_block is None or al is None:
        raise LvlError("%s is missing <actors> or <actor-list>" % args.file)

    cls_map = actor_class_map(doc)
    if args.name not in cls_map:
        raise LvlError("%s: no actor named %r" % (args.file, args.name))
    cls = cls_map[args.name]

    before = len(actors_block.children)
    actors_block.children = [a for a in actors_block.children
                              if not (isinstance(a, Block) and a.tag == args.name)]
    removed_actor = before != len(actors_block.children)

    al.children = [c for c in al.children if not (isinstance(c, KV) and c.key == args.name)]

    di = top_list(doc, 'Dante-Includes')
    if di is not None:
        pat = re.compile(r'^extern\s+\S+\s+%s;$' % re.escape(args.name))
        di.items = [(e, t) for (e, t) in di.items if not pat.match(t)]

    loh = top_block(doc, 'list-of-heros')
    if loh is not None:
        before_h = len(loh.children)
        loh.children = [c for c in loh.children if not (isinstance(c, KV) and c.value == args.name)]
        if len(loh.children) != before_h:
            print("note: removed %r from <list-of-heros>" % args.name, file=sys.stderr)

    ag = top_block(doc, 'actor-groups')
    stale_groups = []
    if ag is not None:
        for c in ag.children:
            if isinstance(c, ListF):
                before_i = len(c.items)
                c.items = [(e, t) for (e, t) in c.items if t != args.name]
                if len(c.items) != before_i:
                    stale_groups.append(c.key)

    out = serialize_lvl(doc)
    with open(args.output, 'wb') as fh:
        fh.write(out)
    print("wrote %s: removed actor %r (class %s)%s"
          % (args.output, args.name, cls, "" if removed_actor else " (was not in <actors>!)"))
    if stale_groups:
        print("  also removed from actor-groups: %s" % ', '.join(stale_groups))
    print("  NOTE: Dependencies, actor-version-list left untouched (harmless if unused)")


# ============================================================================
# CLI
# ============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('stats', help='dependency/actor counts for one .lvl')
    p.add_argument('file')
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser('actors', help='tabular actor listing')
    p.add_argument('file')
    p.add_argument('--class', dest='cls', help='filter by exact class name')
    p.add_argument('--name', help='filter: substring match on actor name')
    p.set_defaults(func=cmd_actors)

    p = sub.add_parser('roundtrip-test', help='parse->write->byte-compare every .lvl in a dir or POD')
    p.add_argument('path', help='directory of .lvl files, a single .lvl, or a .POD archive')
    p.add_argument('-v', '--verbose', action='store_true')
    p.set_defaults(func=cmd_roundtrip_test)

    p = sub.add_parser('schema', help='per-class field schema from dante_api.json + corpus usage')
    p.add_argument('--class', dest='cls', help='restrict to one class')
    p.add_argument('--corpus', help='directory of .lvl files or a .POD, for observed usage counts')
    p.add_argument('--dante-api', default=DEFAULT_DANTE_API)
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser('validate', help='check the dependency invariant + structural sanity')
    p.add_argument('file')
    p.add_argument('--dante-api', default=DEFAULT_DANTE_API)
    p.add_argument('-v', '--verbose', action='store_true')
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser('diff', help='semantic diff of two .lvl files (actors, fields, deps)')
    p.add_argument('a')
    p.add_argument('b')
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser('add-actor', help='clone a template actor into a .lvl')
    p.add_argument('file')
    p.add_argument('--class', dest='cls', help='actor class (required unless --template names one)')
    p.add_argument('--name', required=True, help='new actor name (must be unique in the file)')
    p.add_argument('--pos', nargs=3, type=float, metavar=('X', 'Y', 'Z'))
    p.add_argument('--template', help='name of an existing actor to clone (in FILE or --library)')
    p.add_argument('--library', help='dir/.lvl/.POD to search for a template if FILE has none of --class')
    p.add_argument('--set', action='append', metavar='KEY=VALUE', help='override a field (repeatable)')
    p.add_argument('-o', '--output', required=True)
    p.set_defaults(func=cmd_add_actor)

    p = sub.add_parser('remove-actor', help='remove an actor and its cross-references')
    p.add_argument('file')
    p.add_argument('--name', required=True)
    p.add_argument('-o', '--output', required=True)
    p.set_defaults(func=cmd_remove_actor)

    args = ap.parse_args(argv)
    if args.cmd == 'add-actor' and not args.cls and not args.template:
        ap.error('add-actor requires --class (or a --template to infer it from)')
    try:
        rc = args.func(args)
    except LvlError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    return rc or 0


if __name__ == '__main__':
    sys.exit(main())

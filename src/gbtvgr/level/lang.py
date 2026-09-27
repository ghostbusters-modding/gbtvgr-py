"""Build a level's world\\en\\<stem>.txt from its own .dn dialogue externs and
the shipped generic callout tables."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import os
import re

BANKS = ('callouts.txt', 'callouts_peter.txt', 'splits.txt')
_EXTERN_RE = re.compile(r'^extern CDialogDatabaseEntry (\w+);', re.M)
_SPOKEN_RE = re.compile(r'(?:say|dbStartSay|dbSay)\(\w+, (Diag_\w+)')
_TAG_RE = re.compile(r'^(\w+),', re.M)


class LangError(ValueError):
    pass


def _load(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, 'rb').read().decode('latin1').split('\r\n'):
        m = re.match(r'^(Diag_\w+),\s?(.*)$', line)
        if m:
            out.setdefault(m.group(1), m.group(2))
    return out


def build_subtitles(src_dir, dn_path, out_dir, stem, header_lines, diag_lines):
    """Writes <out_dir>/<stem>.txt (header_lines + every Diag_* the module
    speaks, copied verbatim from the shipped callout tables) and
    <out_dir>/<stem>_diag.txt (diag_lines, this level's own recorded VO).

    THE ID LIST IS READ OUT OF THE MODULE, not kept by hand: it is every
    `extern CDialogDatabaseEntry ...` the .dn declares, and every one of
    those must actually be SPOKEN (say(Who, Id)) or the build fails -- a
    declared-but-unsaid line is dead weight, and the reverse (spoken but
    undeclared) is a D fixup the loader cannot resolve."""
    code = open(dn_path).read()
    ids, seen = [], set()
    for nm in re.findall(r'^extern CDialogDatabaseEntry (\w+);', code, re.M):
        if nm not in seen:
            seen.add(nm)
            ids.append(nm)
    if not ids:
        raise SystemExit('no Diag_* externs in %s' % dn_path)
    unspoken = [i for i in ids
                if not re.search(r'say\(\w+, %s\)' % re.escape(i), code)]
    if unspoken:
        raise SystemExit('%s declares dialogue it never speaks: %s'
                         % (stem, ', '.join(unspoken)))
    spoken = set(re.findall(r'say\(\w+, (Diag_\w+)\)', code))
    undeclared = sorted(spoken - seen)
    if undeclared:
        raise SystemExit('%s speaks dialogue it never declares: %s'
                         % (stem, ', '.join(undeclared)))

    table = {}
    for fn in ('callouts.txt', 'callouts_peter.txt'):
        table.update(_load(os.path.join(src_dir, fn)))
    if not table:
        raise SystemExit('no callouts under %s -- extract LANGUAGE.POD first'
                         % src_dir)
    missing = [i for i in ids if i not in table]
    if missing:
        raise SystemExit('dialogue ids not in the shipped callouts: %s'
                         % ', '.join(missing))
    lines = list(header_lines) + ['%s, %s' % (i, table[i]) for i in ids] + ['']
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, '%s.txt' % stem), 'wb').write(
        ('\r\n'.join(lines)).encode('latin1', 'replace'))
    open(os.path.join(out_dir, '%s_diag.txt' % stem), 'wb').write(
        ('\r\n'.join(diag_lines) + '\r\n').encode('latin1'))
    print('%s.txt: %d dialogue lines (all read out of the module and all '
          'spoken by it) + mission/objective/checkpoint text' % (stem, len(ids)))


# A level's own table. Mission, objective and checkpoint externs are read by the
# engine and never say()'d, so build_subtitles' spoken rule cannot cover them.

def table_tags(data):
    """Every tag a lang table defines (bytes in, latin1)."""
    return set(_TAG_RE.findall(data.decode('latin1')))


def load_callouts(library, banks=BANKS):
    """Diag_* id -> text from the shipped banks in LANGUAGE.POD, first bank wins."""
    out = {}
    for name in banks:
        data = library.read('world\\en\\' + name)
        for line in (data or b'').decode('latin1').split('\r\n'):
            m = re.match(r'^(Diag_\w+),\s?(.*)$', line)
            if m:
                out.setdefault(m.group(1), m.group(2))
    return out


def strip_comments(code):
    return re.sub(r'//[^\n]*', '', code)


def dialogue_externs(code):
    """The CDialogDatabaseEntry externs a .dn declares, in order, once each."""
    seen, out = set(), []
    for name in _EXTERN_RE.findall(strip_comments(code)):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def spoken_ids(code):
    return set(_SPOKEN_RE.findall(strip_comments(code)))


def check_dn_dialogue(code):
    """Every Diag_* declared is spoken and every one spoken is declared."""
    ids = dialogue_externs(code)
    spoken = spoken_ids(code)
    bad = ['%s is declared and never spoken' % i for i in ids
           if i.startswith('Diag_') and i not in spoken]
    bad += ['%s is spoken and never declared' % i for i in sorted(spoken - set(ids))]
    return bad


def _own_tags(own_lines):
    return {row[0] for row in own_lines if not isinstance(row, str) and row[0]}


def check_level_table(externs, callouts, own_lines, global_tags=()):
    """Every extern resolves: Diag_* through the banks, the rest through own_lines or
    global.txt; objShrt_/objLong_ come in pairs."""
    own = _own_tags(own_lines)
    bad = []
    for i in externs:
        if i.startswith('Diag_'):
            if i not in callouts:
                bad.append('%s is in no shipped callout bank' % i)
        elif i not in own and i not in global_tags:
            bad.append('%s.text is read and this table has no entry' % i)
    for tag in sorted(own):
        for a, b in (('objShrt_', 'objLong_'), ('objLong_', 'objShrt_')):
            if tag.startswith(a) and b + tag[len(a):] not in own:
                bad.append('%s has no %s partner' % (tag, b))
    return bad


def _quoted(text):
    return '"%s"' % text if ',' in text else text


def build_level_table(stem, externs, callouts, own_lines, global_tags=(), header=()):
    """world\\en\\<stem>.txt as bytes: header comments, the _diag include, own_lines
    (a str is written verbatim, a (tag, text) pair as a row), then the Diag_* rows."""
    bad = check_level_table(externs, callouts, own_lines, global_tags)
    if bad:
        raise LangError('; '.join(bad))
    lines = [''] + list(header) + ['#include "%s_diag.txt"' % stem, '']
    for row in own_lines:
        lines.append(row if isinstance(row, str) else '%s, %s' % (row[0], _quoted(row[1])))
    diag = [i for i in externs if i.startswith('Diag_')]
    lines += ['', '//callouts spoken by this level'] + ['%s, %s' % (i, callouts[i]) for i in diag]
    lines.append('')
    return '\r\n'.join(lines).encode('latin1', 'replace')


def diag_table(lines):
    """world\\en\\<stem>_diag.txt as bytes, one row per line."""
    return ('\r\n'.join(lines) + '\r\n').encode('latin1', 'replace')


# ============================================================================
# Level-based check: a wire.Level carries the compiled script (not the .dn
# source), so the declared/spoken trace above is not recoverable here -- only
# "does the shipped lang table cover what the script actually references" is,
# which is the .lang field's own reason to exist on the wire Level.
# ============================================================================

def dialogue_ids(mod):
    """Every CDialogDatabaseEntry name a compiled .dante references."""
    from . import checks
    return sorted({name for cls, name in checks.script_externs(mod)
                   if cls == 'CDialogDatabaseEntry'})


def check_lang_level(level, extra_tables=()):
    """Every CDialogDatabaseEntry id the compiled script references has an
    entry in every language table the level ships (level.lang), or in extra_tables."""
    if level.script is None:
        return ['no compiled script for this level']
    from dante import module as dante_module
    mod = dante_module.Dante(level.script.decode('latin-1'), level.stem)
    ids = dialogue_ids(mod)
    bad = []
    if not level.lang:
        return ['%d Diag_* ids referenced but the level ships no lang table'
                % len(ids)] if ids else []
    extra = set()
    for data in extra_tables:
        extra |= table_tags(data)
    for lang_name, data in level.lang.items():
        tags = table_tags(data) | extra
        missing = sorted(i for i in ids if i not in tags)
        if missing:
            bad.append('%s: %d referenced Diag_* ids have no entry (%s)'
                       % (lang_name, len(missing), ', '.join(missing[:5])))
    return bad

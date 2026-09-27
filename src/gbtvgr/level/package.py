"""Staging a mod's file tree and shipping it the way every mod here ships: a
chained archive built and installed by gbtvgr's patchpod."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import argparse
import io
import os
import sys
from contextlib import redirect_stdout

from gbtvgr.archive import patchpod, pod

GBHOOK_CMD = 'gbhook.cmd'


def stage(out_dir, set_name, set_bytes, level_stem, level_bytes, dante_bytes=None,
          assets=None, extra_files=None):
    """Write files/ as a POD-relative tree. Returns the files directory."""
    files = os.path.join(out_dir, 'files')
    written = []

    def put(rel, data):
        dst = os.path.join(files, *rel.replace('/', '\\').split('\\'))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'wb') as fh:
            fh.write(data)
        written.append(rel)

    put('sets\\%s.bst' % set_name, set_bytes)
    put('world\\%s.lvl' % level_stem, level_bytes)
    if dante_bytes is not None:
        put('world\\%s.dante' % level_stem, dante_bytes)
    for rel, data in (assets or {}).items():
        put(rel, data)
    for rel, path in (extra_files or {}).items():
        with open(path, 'rb') as fh:
            put(rel, fh.read())
    return files, written


def build_pod(files_dir, out_pod, game_dir=None, revision=1000):
    a = argparse.Namespace(mods=files_dir, out=out_pod, game=game_dir, next=None,
                           revision=revision, level=9)
    buf = io.StringIO()
    with redirect_stdout(buf):
        patchpod.cmd_content(a)
    return buf.getvalue()


def install_pod(game_dir, pod_path, name='EDITOR.POD', force=False):
    a = argparse.Namespace(game=game_dir, pod=pod_path, name=name, host='PATCH.POD',
                           force=force, keep_chain=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        patchpod.cmd_install(a)
    return buf.getvalue()


def uninstall_pod(game_dir, name='EDITOR.POD', force=False):
    a = argparse.Namespace(game=game_dir, name=name, host='PATCH.POD', force=force)
    buf = io.StringIO()
    with redirect_stdout(buf):
        patchpod.cmd_uninstall(a)
    return buf.getvalue()


def list_pod(pod_path):
    with pod.Pod(pod_path) as p:
        return p.names()


def mount_chain(game_dir):
    names, note = pod.walk_chain(os.path.join(game_dir, 'PATCH.POD'))
    return names, note


def level_command(game_dir, stem):
    """Queue `level <stem>` for gb-hook, which polls gbhook.cmd one line at a time."""
    path = os.path.join(game_dir, GBHOOK_CMD)
    with open(path, 'a', encoding='ascii') as fh:
        fh.write('level %s\n' % stem)
    return path


def game_running():
    return patchpod._game_is_running()


class SystemExitCapture:
    """patchpod reports refusals with sys.exit; turn them into exceptions."""

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is SystemExit:
            raise RuntimeError(str(ev))
        return False


def run_guarded(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except SystemExit as exc:
        raise RuntimeError(str(exc))

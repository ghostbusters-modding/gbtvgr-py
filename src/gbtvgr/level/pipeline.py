"""The whole export in one call: components, script, bake, navmesh, set, level,
staging, archive. Every stage reports what it verified and what it did not."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import os
import time

import numpy as np

from .mathutil import F32, transform_points
from . import components as gcomp
from . import bake as gbake
from . import levelfile as glevel
from . import navmesh as gnav
from . import package as gpkg
from . import scripts as gscripts
from . import sections as gsec
from . import setfile as gset

MAX_COLLISION_PARTS = 20      # a solid CProp over this breaks level prepare


class BuildOptions:
    def __init__(self, **kw):
        self.out_dir = None
        self.bake_vertex = True
        self.ao = True
        self.lightmaps = False
        self.probes = True
        self.navmesh = True
        self.pod = True
        self.install = False
        self.launch = False
        self.cubemap_fn = None            # (pos, size) -> six faces; the viewport's renderer
        self.revision = 1000
        self.force = False
        for k, v in kw.items():
            setattr(self, k, v)


class BuildResult:
    def __init__(self):
        self.report = []
        self.warnings = []
        self.errors = []
        self.files_dir = None
        self.pod_path = None
        self.dante_path = None
        self.set_build = None
        self.level_build = None
        self.nav = None
        self.seconds = 0.0

    @property
    def ok(self):
        return not self.errors

    def log(self, line):
        self.report.append(line)


def preflight(doc, library):
    """Warnings a build would ship with, before anything is written."""
    out = []
    if not doc.section_list():
        out.append('no sections: the set will not compile')
    if not any(n.kind in ('mesh', 'terrain') for n in doc.root.iter_descendants()):
        out.append('no geometry: the player would fall forever')
    for node in doc.nodes('component'):
        cd = doc.components.get(node.component_id)
        if cd is None:
            out.append('%s: refers to a missing component' % node.name)
            continue
        if node.props.get('solid', True) and library is not None:
            m = library.read_model(node.props.get('mesh') or cd.mesh)
            if m is not None and m.collision_part_count > MAX_COLLISION_PARTS:
                out.append('%s: %s has %d collision parts; a solid prop over %d breaks level prepare'
                           % (node.name, cd.mesh, m.collision_part_count, MAX_COLLISION_PARTS))
    for node in doc.nodes('actor'):
        if node.props.get('cls') == 'CProp' and node.props.get('solid') and node.model and library is not None:
            m = library.read_model(node.model)
            if m is not None and m.collision_part_count > MAX_COLLISION_PARTS:
                out.append('%s: %d collision parts on a solid prop' % (node.name, m.collision_part_count))
    names = [n.name for n in doc.nodes()]
    dupes = sorted({n for n in names if names.count(n) > 1 and n})
    if dupes:
        out.append('duplicate node names: %s' % ', '.join(dupes[:8]))
    for node in doc.nodes('trigger'):
        for prog in ('on_enter', 'on_leave'):
            for step in node.props.get(prog, []):
                for k, v in step.get('params', {}).items():
                    if isinstance(v, str) and v.startswith('@') and doc.find(v[1:]) is None:
                        out.append('%s: block references a missing node %s' % (node.name, v))
    return out


def world_colored_triangles(doc):
    """Every mesh node as world-space vertex-coloured triangles, for probe renders."""
    pos, col, idx, base = [], [], [], 0
    for n in doc.mesh_nodes():
        geo = n.mesh if n.kind == 'mesh' else (n.terrain.mesh() if n.terrain else None)
        if geo is None or geo.nfaces == 0:
            continue
        p, _nrm, _uv, i, cmap, _first = geo.split_corners()
        wm = n.world_matrix()
        pos.append(transform_points(wm, p))
        colors = geo.colors if geo.colors is not None else getattr(getattr(n, 'terrain', None), 'colors', None)
        if colors is not None and len(colors) == geo.nverts:
            col.append(np.asarray(colors, np.uint8)[cmap])
        else:
            col.append(np.full((len(p), 4), 200, np.uint8))
        idx.append(i.astype(np.uint32) + base)
        base += len(p)
    if not pos:
        return np.zeros((0, 3), F32), np.zeros((0, 4), np.uint8), np.zeros((0, 3), np.uint32)
    return np.concatenate(pos), np.concatenate(col), np.concatenate(idx)


def _flatten_handlers(doc):
    """Only the two slots shipped levels fill; the other stubs are exported anyway."""
    out = {}
    for name, fields in gscripts.handler_names(doc).items():
        for field in ('actorEnterEvent', 'actorLeaveEvent'):
            if field in fields:
                out[(name, field)] = fields[field]
    return out


def build_all(doc, library, options=None, log=None, progress=None):
    opt = options or BuildOptions()
    res = BuildResult()
    t0 = time.time()

    def say(line):
        res.log(line)
        if log:
            log(line)

    def step(name, frac):
        if progress:
            progress(name, frac)

    s = doc.settings
    set_name = s.get('set_name', 'level')
    stem = s.get('level_stem', set_name + '1')
    if not opt.out_dir:
        raise ValueError('BuildOptions.out_dir is required')
    out_dir = opt.out_dir
    files_dir = os.path.join(out_dir, 'files')
    os.makedirs(files_dir, exist_ok=True)
    res.files_dir = files_dir
    res.warnings.extend(preflight(doc, library))

    # 1. components -----------------------------------------------------------------
    step('components', 0.05)
    plan = gcomp.resolve_all(doc)
    model_refs = {}
    try:
        gcomp.write_plan(plan, library, files_dir, log=say)
        for node, key in plan.instances:
            model_refs[node.id] = plan.model_ref[key]
        say('components: %d instances, %d unique variants, %d material clones'
            % (len(plan.instances), len(plan.variants), len(plan.material_clones)))
    except (FileNotFoundError, ValueError) as exc:
        res.errors.append('components: %s' % exc)

    # 2. script ----------------------------------------------------------------------
    step('script', 0.15)
    handlers = {}
    if s.get('keep_script'):
        say('script: keeping the shipped world\\%s.dante' % stem)
    else:
        try:
            res.dante_path = gscripts.compile_document(doc, library, out_dir, log=say)
            handlers = _flatten_handlers(doc)
        except gscripts.ScriptError as exc:
            res.errors.append('script: %s' % exc)

    # 3. bake ---------------------------------------------------------------------------
    step('bake', 0.25)
    settings = gbake.settings_of(doc, {'ao': bool(opt.ao)})
    if opt.bake_vertex:
        # the baked colour is what shows a set light, so the grid must exist before the bake
        max_edge = doc.settings.get('max_edge', gset.DEFAULT_MAX_EDGE)
        if max_edge and not opt.lightmaps:
            for n in doc.nodes('mesh'):
                if n.mesh is not None and len(n.mesh.faces):
                    n.mesh = gsec.tessellate_mesh(n.mesh, max_edge)
        try:
            gbake.bake_vertex_colors(doc, settings)
            say('bake: vertex colours over %d nodes' % len(doc.mesh_nodes()))
        except Exception as exc:  # noqa: BLE001
            res.errors.append('bake: %s' % exc)
    templates = gbake.donor_templates(library) if library is not None else {}
    lightmaps = {}
    if opt.lightmaps:
        for sec in doc.section_list():
            try:
                plan_lm = gbake.plan_lightmaps(doc, sec, settings=settings)
                lightmaps[sec.id] = gbake.bake_lightmaps(doc, sec, plan_lm, settings,
                                                         template_tile=templates.get('tile'))
                say('bake: lightmap %dx%d for section %s' % (lightmaps[sec.id]['size'][0],
                                                            lightmaps[sec.id]['size'][1], sec.name))
            except Exception as exc:  # noqa: BLE001
                res.errors.append('lightmap %s: %s' % (sec.name, exc))
    probe_tiles = None
    probes = [n for n in doc.probes.children if n.kind == 'probe']
    if opt.probes and probes:
        fn = opt.cubemap_fn
        raster = None
        if fn is None:
            try:
                from .raster import Rasteriser
                raster = Rasteriser()
                raster.set_geometry(*world_colored_triangles(doc))
                fn = raster.cubemap
            except Exception as exc:  # noqa: BLE001
                res.warnings.append('probes: no offscreen renderer (%s); tiles not baked' % exc)
        if fn is not None:
            try:
                probe_tiles = gbake.bake_probes(probes, fn, templates.get('ambient'), templates.get('env'))
                say('bake: %d light probes' % len(probe_tiles))
            except Exception as exc:  # noqa: BLE001
                res.errors.append('probes: %s' % exc)
        if raster is not None:
            raster.close()

    # 4. navmesh -----------------------------------------------------------------------
    step('navmesh', 0.45)
    nav = None
    if opt.navmesh:
        try:
            p = doc.nav.props
            world = gnav.WalkableWorld.from_document(doc, library)
            res.nav = gnav.build_navmesh(world, cell=float(p.get('cell', 6.0)),
                                         max_step=float(p.get('max_step', 2.0)),
                                         max_slope_deg=float(p.get('max_slope_deg', 40.0)),
                                         agent_height=float(p.get('agent_height', 8.0)))
            nav = res.nav.nav
            say('navmesh: %s' % (res.nav.report.strip().splitlines()[0] if res.nav.report else
                                 '%d nodes' % len(res.nav.nodes)))
            for prob in gnav.nav_selftest(res.nav):
                res.warnings.append('navmesh: %s' % prob)
        except Exception as exc:  # noqa: BLE001
            res.errors.append('navmesh: %s' % exc)

    # 5. set ---------------------------------------------------------------------------------
    step('set', 0.6)
    sb = gset.compile_set(doc, library, nav=nav, lightmaps=lightmaps, probe_tiles=probe_tiles, log=say)
    res.set_build = sb
    res.warnings.extend(sb.warnings)
    res.errors.extend(sb.errors)

    # 6. level -------------------------------------------------------------------------------
    step('level', 0.75)
    lb = glevel.compile_level(doc, library, handlers, model_refs, log=say)
    res.level_build = lb
    res.warnings.extend(lb.warnings)
    res.errors.extend(lb.errors)
    if res.errors:
        res.seconds = time.time() - t0
        return res

    # 7. stage and archive ---------------------------------------------------------------------
    step('package', 0.85)
    dante_bytes = None
    if res.dante_path:
        with open(res.dante_path, 'rb') as fh:
            dante_bytes = fh.read()
    # the .lvl names its .sec layers, so they ship beside it or the engine stops on the first
    assets = dict(sb.assets)
    assets.update({'world\\%s.sec' % n: b for n, b in getattr(lb, 'secs', [])})
    gpkg.stage(out_dir, set_name, sb.data, stem, lb.data, dante_bytes, assets=assets)
    say('staged %d files under %s' % (len(assets) + 3, files_dir))
    if opt.pod:
        res.pod_path = os.path.join(out_dir, s.get('archive_name', 'EDITOR.POD'))
        try:
            say(gpkg.run_guarded(gpkg.build_pod, files_dir, res.pod_path,
                                 library.game_dir if library else None, opt.revision).strip())
        except RuntimeError as exc:
            res.errors.append('pod: %s' % exc)
    if opt.install and res.pod_path and library is not None and res.ok:
        step('install', 0.95)
        try:
            say(gpkg.run_guarded(gpkg.install_pod, library.game_dir, res.pod_path,
                                 s.get('archive_name', 'EDITOR.POD'), opt.force).strip())
        except RuntimeError as exc:
            res.errors.append('install: %s' % exc)
    if opt.launch and res.ok and library is not None:
        gpkg.level_command(library.game_dir, stem)
        say('queued `level %s` in gbhook.cmd' % stem)
    res.seconds = time.time() - t0
    step('done', 1.0)
    return res

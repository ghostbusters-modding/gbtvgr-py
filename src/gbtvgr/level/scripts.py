"""Trigger block programs -> a `.dn` level module -> `.dante` through the real
toolchain. The shapes are the shipped ones: stubs for every CTrigger event
slot, per-class spawn wrappers, one blocking level thread."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import json
import os
import re

# CTrigger event field -> handler suffix; a missing export is a fatal prepare error
TRIG_EVENTS = [
    ('onEvent', 'onEvent'),
    ('risingEdgeEvent', 'onRisingEdge'),
    ('fallingEdgeEvent', 'onFallingEdge'),
    ('actorEnterEvent', 'onActorEnter'),
    ('actorInsideEvent', 'onActorInside'),
    ('actorLeaveEvent', 'onActorLeave'),
    ('actorEnterUpStreamEvent', 'onActorEnterUpStream'),
    ('actorEnterDownStreamEvent', 'onActorEnterDownStream'),
    ('actorLeaveUpStreamEvent', 'onActorLeaveUpStream'),
    ('actorLeaveDownStreamEvent', 'onActorLeaveDownStream'),
]
TRIG_PROTO = 'void %s_%s(@CTrigger,@CActor)'
SPAWN_PROTO = 'void %s_onActivate(@CSpawn)'
SPAWN_CLASSES = ('CBiped', 'CBipedLarge', 'CFloater', 'CFlyerMedium', 'CFlyerSmall', 'CScuttler')
BLOCKING = {'wait', 'camera_orbit'}
HERO = 'Ghostbuster0'
SQUAD = ('Winston', 'Egon', 'Peter', 'Ray')

_BLOCKS = None


class ScriptError(Exception):
    pass


def load_blocks(data_dir=None):
    """`data_dir` holds blocks.json; defaults to this package's own bundled copy."""
    global _BLOCKS
    if _BLOCKS is None:
        path = os.path.join(data_dir or os.path.join(os.path.dirname(__file__), 'data'), 'blocks.json')
        with open(path, 'r', encoding='utf-8') as fh:
            d = json.load(fh)
        _BLOCKS = {b['id']: b for b in d['blocks']}
    return _BLOCKS


def block_params(block_id, params):
    """The block's parameters with defaults filled in."""
    b = load_blocks().get(block_id)
    if b is None:
        raise ScriptError('unknown block %r' % block_id)
    out = {p['name']: p.get('default') for p in b['params']}
    out.update(params or {})
    return out


def ident(name):
    s = re.sub(r'[^A-Za-z0-9_]', '_', str(name or ''))
    if not s or s[0].isdigit():
        s = '_' + s
    return s


def dn_str(s):
    s = str(s if s is not None else '')
    if '\\\\' in s:
        s = s.replace('\\\\', '\\')
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'


def dn_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = 0.0
    s = ('%.6f' % f).rstrip('0').rstrip('.')
    if '.' not in s:
        s += '.0'
    if s == '-0.0':
        s = '0.0'
    return s


def dn_vec(v):
    v = list(v or (0, 0, 0)) + [0, 0, 0]
    return 'getVector(%s, %s, %s)' % tuple(dn_float(x) for x in v[:3])


def dn_bool(v):
    if isinstance(v, str):
        return 'true' if v.strip().lower() in ('1', 'true', 'yes') else 'false'
    return 'true' if v else 'false'


class ScriptPlan:
    """Everything the module declares, collected before any code is written."""

    def __init__(self, doc):
        self.doc = doc
        self.stem = ident(doc.settings.get('level_stem') or 'level')
        self.warnings = []
        self.names = {}                 # node name -> identifier
        self.used = set()
        self.actor_cls = {}             # node name -> class
        self.group_nodes = set()
        self.actor_externs = []         # (cls, ident) in first-use order
        self.group_externs = []
        self.script_groups = []         # value-object CActorGroup globals
        self.externs = set()            # global.dante exports used
        self.engine = set()             # gTimeSlice, gMainView
        self.spawn_classes = []
        self.effects, self.sfx = [], []
        self.checkpoints = {}           # name -> text
        self.triggers = []              # TriggerNode list
        self.spawners = []              # CSpawn ActorNode names
        self.heroes = []
        self._scan()

    def _register(self, name):
        if name in self.names:
            return self.names[name]
        base = ident(name)
        cand, n = base, 2
        while cand in self.used:
            cand = '%s_%d' % (base, n)
            n += 1
        if cand != base:
            self.warnings.append('name %r collides; script calls it %s' % (name, cand))
        self.used.add(cand)
        self.names[name] = cand
        return cand

    def _scan(self):
        doc = self.doc
        for n in doc.nodes('actor'):
            self.actor_cls[n.name] = n.cls
            if n.cls == 'CSpawn':
                self.spawners.append(n.name)
            if n.cls == 'CGhostbuster':
                self.heroes.append(n.name)
        for n in doc.nodes('group'):
            self.group_nodes.add(n.name)
        for n in doc.nodes('trigger'):
            self.triggers.append(n)
            self._register(n.name)
        for name in self.spawners:
            self._register(name)
        if HERO not in self.heroes:
            self.heroes.insert(0, HERO)
        # the hero and squad come from the template level, not actor nodes; a block naming
        # one must not declare it a second time as a CActor (the engine refuses the script)
        for name in [HERO] + list(SQUAD):
            self.actor_cls.setdefault(name, 'CGhostbuster')
        for h in self.heroes:
            self.actor(h, 'CGhostbuster')
        for t in self.triggers:
            for prog in ('on_enter', 'on_leave', 'on_state'):
                for step in t.props.get(prog) or []:
                    self._scan_step(step)
            st = t.props.get('state')
            if st:
                self._scan_state(st)

    def _scan_state(self, st):
        p = block_params(st['block'], st.get('params'))
        b = st['block']
        if b == 'actor_group_dead':
            self.group(p['group'])
            self.externs.add('bool isActorGroupDead(@CActorGroup)')
        elif b == 'actor_dead':
            self.actor(p['actor'])
        elif b == 'timer':
            self.engine.add('gTimeSlice')
        elif b == 'checkpoint_loaded':
            pass

    def _scan_step(self, step):
        b = step.get('block')
        p = block_params(b, step.get('params'))
        if b == 'spawn_character':
            self.actor(p['spawner'], 'CSpawn')
            cls = p['cls'] or 'CBiped'
            if cls not in self.spawn_classes:
                self.spawn_classes.append(cls)
            if p['group']:
                self.group(p['group'])
            self.externs.add('Vector getVector(float,float,float)')
        elif b == 'kill_group':
            self.group(p['group'])
            self.externs.add('void killGroupMembers(@CActorGroup)')
        elif b in ('enable_actor', 'disable_actor'):
            self.actor(p['actor'])
        elif b == 'warp_actor':
            self.actor(p['actor'])
            self.externs.add('Vector getVector(float,float,float)')
        elif b == 'camera_fixed':
            self.actor(p['camera'], 'CCameraPathActor')
            self.actor(p['target'])
            self.engine.add('gMainView')
        elif b == 'camera_orbit':
            self.actor(p['target'])
            self.engine.add('gMainView')
        elif b == 'camera_normal':
            self.engine.add('gMainView')
        elif b == 'start_effect':
            if p['at']:
                self.actor(p['at'])
            else:
                self.externs.add('Vector getVector(float,float,float)')
            if p['effect'] and p['effect'] not in self.effects:
                self.effects.append(p['effect'])
        elif b == 'play_sfx':
            if p['name'] and p['name'] not in self.sfx:
                self.sfx.append(p['name'])
        elif b == 'wait':
            self.externs.add('void wait(float)')
        elif b == 'define_checkpoint':
            self.checkpoints[ident(p['name'])] = p['text']
        elif b == 'enable_trigger':
            self.actor(p['trigger'], 'CTrigger')

    def actor(self, name, cls=None):
        """An actor extern; returns its identifier."""
        if not name:
            self.warnings.append('a block names no actor')
            return HERO
        cls = self.actor_cls.get(name) or cls
        if cls is None:
            self.warnings.append('actor %r is not in the level; declared as CActor' % name)
            cls = 'CActor'
        i = self._register(name)
        if (cls, i) not in self.actor_externs:
            self.actor_externs.append((cls, i))
        return i

    def group(self, name):
        if not name:
            self.warnings.append('a block names no group')
            return 'g_editor'
        i = self._register(name)
        if name in self.group_nodes:
            if i not in self.group_externs:
                self.group_externs.append(i)
        elif i not in self.script_groups:
            self.script_groups.append(i)
        return i

    def id_of(self, name):
        return self.names.get(name) or self._register(name)


class Generator:
    def __init__(self, doc):
        self.doc = doc
        self.plan = ScriptPlan(doc)
        self.out = []
        self.tmp = 0

    # -- helpers ----------------------------------------------------------------------
    def line(self, s=''):
        self.out.append(s)

    def temp(self, base):
        self.tmp += 1
        return '%s_%d' % (base, self.tmp)

    def is_blocking(self, steps):
        return any(s.get('block') in BLOCKING for s in steps or [])

    # -- the pieces -------------------------------------------------------------------
    def emit_header(self):
        p = self.plan
        self.line('// Generated by gbtvgr-editor from the trigger blocks of %s.' % self.doc.settings.get('set_name', ''))
        self.line('module %s;' % p.stem)
        self.line()
        for e in sorted(p.externs):
            self.line('extern "%s";' % e)
        if 'gTimeSlice' in p.engine:
            self.line('extern @float gTimeSlice;')
        if 'gMainView' in p.engine:
            self.line('extern @CGameView gMainView;')
        self.line()
        for cls, i in p.actor_externs:
            self.line('extern %s %s;' % (cls, i))
        for i in p.group_externs:
            self.line('extern CActorGroup %s;' % i)
        for i in p.script_groups:
            self.line('CActorGroup %s;' % i)
        self.line()
        self.line('bool loadingFromCheckpoint = false;')
        for t in p.triggers:
            i = p.id_of(t.name)
            if t.props.get('once', True):
                self.line('bool %s_done = false;' % i)
            if t.props.get('state') and not t.props.get('once', True):
                self.line('bool %s_was = false;' % i)
            if t.props.get('state') and t.props['state']['block'] == 'timer':
                self.line('float %s_clock = 0.0;' % i)
        self.line()

    def emit_spawn_wrappers(self):
        for cls in self.plan.spawn_classes:
            self.line('@%s spawn%s(@CSpawn spawn, String citFilename, String variant)' % (cls, cls[1:]))
            self.line('{')
            self.line('    SSpawnInfo spawnInfo;')
            self.line('    spawnInfo.classTypeName = "%s";' % cls)
            self.line('    spawnInfo.citFilename = citFilename;')
            self.line('    spawnInfo.variantName = variant;')
            self.line('    return (@%s) spawn.spawnCharacter(spawnInfo);' % cls)
            self.line('}')
            self.line()

    def emit_step(self, step, indent='    '):
        p = self.plan
        b = step.get('block')
        pr = block_params(b, step.get('params'))
        L = lambda s: self.line(indent + s)  # noqa: E731
        if b == 'spawn_character':
            cls = pr['cls'] or 'CBiped'
            c = self.temp('spawned')
            L('{')
            L('    @%s %s = spawn%s(%s, %s, %s);' % (cls, c, cls[1:], p.id_of(pr['spawner']),
                                                    dn_str(pr['cit']), dn_str(pr['variant'] or 'standard')))
            L('    if (%s != null) {' % c)
            if pr['victim'] in (True, 'true', '1', 1):
                L('        %s.setVictimUponSpawn(%s);' % (c, HERO))
            if pr['group']:
                L('        %s.add(%s);' % (p.id_of(pr['group']), c))
            L('    }')
            L('}')
        elif b == 'kill_group':
            L('killGroupMembers(%s);' % p.id_of(pr['group']))
        elif b == 'enable_actor':
            L('%s.enable(true);' % p.id_of(pr['actor']))
        elif b == 'disable_actor':
            L('%s.enable(false);' % p.id_of(pr['actor']))
        elif b == 'warp_actor':
            L('%s.warpTo(%s, %s);' % (p.id_of(pr['actor']), dn_vec(pr['pos']), dn_vec(pr['orient'])))
        elif b == 'camera_fixed':
            L('gMainView.setCameraModeFixed(%s, %s, %s);' % (p.id_of(pr['camera']), p.id_of(pr['target']),
                                                            dn_float(pr['blend'])))
        elif b == 'camera_orbit':
            L('gMainView.setCameraModeOrbit(%s, %s, 0.05, 0.45, 5.0, 0.7, %s);'
              % (p.id_of(pr['target']), dn_float(pr['radius']), dn_float(pr['seconds'])))
            L('wait(%s);' % dn_float(pr['seconds']))
            L('gMainView.setCameraModeNormal(0.0, 0.0);')
        elif b == 'camera_normal':
            L('gMainView.setCameraModeNormal(0.0, 0.0);')
        elif b == 'display_message':
            L('displayMessage(eHudMessage_ObjectivesUpdated, %s, %s);' % (dn_str(pr['text']), dn_float(pr['seconds'])))
        elif b == 'set_objective':
            L('setCurrentObjective(%s);' % dn_str(pr['text']))
            L('displayMessage(eHudMessage_ObjectivesUpdated, %s, -1.0);' % dn_str(pr['short'] or pr['text']))
        elif b == 'start_effect':
            if pr['at']:
                a = p.id_of(pr['at'])
                L('startEffect(%s, %s.getPos(), %s.getOrient());' % (dn_str(pr['effect']), a, a))
            else:
                L('startEffect(%s, %s, getVector(0.0, 0.0, 0.0));' % (dn_str(pr['effect']), dn_vec(pr['pos'])))
        elif b == 'play_sfx':
            L('startSfx(%s);' % dn_str(pr['name']))
        elif b == 'set_music':
            L('setMusic(%s);' % dn_str(pr['cue']))
        elif b == 'wait':
            L('wait(%s);' % dn_float(pr['seconds']))
        elif b == 'save_checkpoint':
            name = ident(pr['name'])
            text = p.checkpoints.get(name)
            if text is None:
                p.warnings.append('save_checkpoint %r has no define_checkpoint' % pr['name'])
                text = pr['name']
            L('saveCheckpoint(%s);' % dn_str(text))
        elif b == 'define_checkpoint':
            pass
        elif b == 'enable_trigger':
            L('%s.enable(%s);' % (p.id_of(pr['trigger']), dn_bool(pr['enabled'])))
        elif b == 'run_dn':
            for raw in str(pr['code'] or '').splitlines():
                L(raw)
        else:
            p.warnings.append('unknown block %r skipped' % b)

    def emit_program(self, steps, indent='    '):
        for s in steps or []:
            self.emit_step(s, indent)

    def program_call(self, tid, prog, steps, indent='    '):
        """Inline the program, or hand it to a thread when it blocks."""
        if self.is_blocking(steps):
            self.line('%sbeginThread(%s_%s, false);' % (indent, tid, prog))
        else:
            self.emit_program(steps, indent)

    def emit_thread_functions(self):
        for t in self.plan.triggers:
            tid = self.plan.id_of(t.name)
            for prog in ('on_enter', 'on_leave', 'on_state'):
                steps = t.props.get(prog) or []
                if steps and self.is_blocking(steps):
                    self.line('void %s_%s()' % (tid, prog))
                    self.line('{')
                    self.emit_program(steps)
                    self.line('}')
                    self.line()

    def emit_setup(self):
        p = self.plan
        self.line('void setupLevel()')
        self.line('{')
        for name, text in p.checkpoints.items():
            self.line('    defineCheckpoint(%s, %s);' % (dn_str('void checkpoint_%s()' % name), dn_str(text)))
        desc = self.doc.settings.get('description')
        if desc:
            self.line('    setLevelDescription(%s);' % dn_str(desc))
        for fx in p.effects:
            self.line('    cacheEffect(%s);' % dn_str(fx))
        for s in p.sfx:
            self.line('    cacheSfx(%s);' % dn_str(s))
        for t in p.triggers:
            st = t.props.get('state')
            if st and st['block'] == 'actor_dead':
                pr = block_params('actor_dead', st.get('params'))
                self.line('    %s.setDeathEvent(%s_onDeath);' % (p.id_of(pr['actor']), p.id_of(t.name)))
        self.line('}')
        self.line()
        for name in p.checkpoints:
            self.line('void checkpoint_%s()' % name)
            self.line('{')
            self.line('    loadingFromCheckpoint = true;')
            for t in p.triggers:
                st = t.props.get('state')
                if st and st['block'] == 'checkpoint_loaded':
                    pr = block_params('checkpoint_loaded', st.get('params'))
                    if ident(pr['checkpoint']) == name:
                        self.program_call(p.id_of(t.name), 'on_state', t.props.get('on_state'))
            self.line('}')
            self.line()

    def state_condition(self, t):
        p = self.plan
        st = t.props['state']
        pr = block_params(st['block'], st.get('params'))
        b = st['block']
        tid = p.id_of(t.name)
        if b == 'actor_group_dead':
            return 'isActorGroupDead(%s)' % p.id_of(pr['group'])
        if b == 'players_downed':
            return ' && '.join('%s.getHitPointsPct() <= 0.0' % p.id_of(h) for h in p.heroes)
        if b == 'timer':
            return '%s_clock >= %s' % (tid, dn_float(pr['seconds']))
        return None

    def emit_main(self):
        p = self.plan
        polled = [t for t in p.triggers if t.props.get('state')
                  and t.props['state']['block'] in ('actor_group_dead', 'players_downed', 'timer')]
        self.line('void main()')
        self.line('{')
        if polled:
            self.line('    beginThread(stateThread, false);')
        self.line('    for (; ; ) {')
        self.line('        idle();')
        self.line('    }')
        self.line('}')
        self.line()
        if not polled:
            return
        self.line('void stateThread()')
        self.line('{')
        self.line('    for (; ; ) {')
        for t in polled:
            tid = p.id_of(t.name)
            if t.props['state']['block'] == 'timer':
                self.line('        %s_clock += *gTimeSlice;' % tid)
            cond = self.state_condition(t)
            if t.props.get('once', True):
                self.line('        if (!%s_done && (%s)) {' % (tid, cond))
                self.line('            %s_done = true;' % tid)
                self.program_call(tid, 'on_state', t.props.get('on_state'), '            ')
                self.line('        }')
            else:
                self.line('        if (%s) {' % cond)
                self.line('            if (!%s_was) {' % tid)
                self.line('                %s_was = true;' % tid)
                self.program_call(tid, 'on_state', t.props.get('on_state'), '                ')
                self.line('            }')
                self.line('        } else {')
                self.line('            %s_was = false;' % tid)
                self.line('        }')
        self.line('        idle();')
        self.line('    }')
        self.line('}')
        self.line()

    def emit_handlers(self):
        p = self.plan
        for t in p.triggers:
            tid = p.id_of(t.name)
            once = t.props.get('once', True)
            for field, suffix in TRIG_EVENTS:
                prog = {'onActorEnter': 'on_enter', 'onActorLeave': 'on_leave'}.get(suffix)
                steps = t.props.get(prog) if prog else None
                if not steps:
                    self.line('void %s_%s(@CTrigger t, @CActor a) { }' % (tid, suffix))
                    continue
                self.line('void %s_%s(@CTrigger trigger, @CActor actor)' % (tid, suffix))
                self.line('{')
                if once:
                    self.line('    if (%s_done) return;' % tid)
                    self.line('    %s_done = true;' % tid)
                self.program_call(tid, prog, steps)
                self.line('}')
            st = t.props.get('state')
            if st and st['block'] == 'actor_dead':
                self.line('void %s_onDeath(@CCharacter character)' % tid)
                self.line('{')
                if once:
                    self.line('    if (%s_done) return;' % tid)
                    self.line('    %s_done = true;' % tid)
                self.program_call(tid, 'on_state', t.props.get('on_state'))
                self.line('}')
            self.line()
        for name in p.spawners:
            self.line('void %s_onActivate(@CSpawn spawn) { }' % p.id_of(name))
        if p.spawners:
            self.line()

    def generate(self):
        self.emit_header()
        self.emit_spawn_wrappers()
        self.emit_thread_functions()
        self.emit_setup()
        self.emit_main()
        self.emit_handlers()
        user = self.doc.settings.get('user_dn') or ''
        if user.strip():
            self.line('// user')
            self.line(user.rstrip('\n'))
            self.line()
        return '\n'.join(self.out) + '\n'


def generate_source(doc):
    return Generator(doc).generate()


def plan_for(doc):
    return ScriptPlan(doc)


def handler_names(doc):
    """{trigger name: {.lvl event field: prototype text}} for every trigger."""
    p = ScriptPlan(doc)
    out = {}
    for t in p.triggers:
        tid = p.id_of(t.name)
        out[t.name] = {field: TRIG_PROTO % (tid, suffix) for field, suffix in TRIG_EVENTS}
    return out


def spawner_handlers(doc):
    p = ScriptPlan(doc)
    return {name: SPAWN_PROTO % p.id_of(name) for name in p.spawners}


def global_lib_path(library, cache_dir):
    """world\\global.dante out of the archives, cached under `cache_dir` for the compiler."""
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, 'global.dante')
    blob = library.read('world\\global.dante')
    if blob is None:
        raise ScriptError('world\\global.dante is not in the archives')
    if not os.path.isfile(dst) or os.path.getsize(dst) != len(blob):
        with open(dst, 'wb') as fh:
            fh.write(blob)
    return dst


def compile_document(doc, library, out_dir, log=None):
    """Write world/<stem>.dn and compile it to world/<stem>.dante; returns the .dante path."""
    import dante
    gen = Generator(doc)
    src = gen.generate()
    stem = gen.plan.stem
    wdir = os.path.join(out_dir, 'world')
    os.makedirs(wdir, exist_ok=True)
    dn_path = os.path.join(wdir, stem + '.dn')
    with open(dn_path, 'w', encoding='latin-1', newline='\n') as fh:
        fh.write(src)
    for w in gen.plan.warnings:
        if log:
            log('script: ' + w)
    lib = global_lib_path(library, out_dir)
    try:
        built = dante.compile_source(dn_path, libs=[lib], module=stem)
    except dante.CompileError as ex:
        raise ScriptError('%s: %s' % (dn_path, ex))
    out = os.path.join(wdir, stem + '.dante')
    with open(out, 'wb') as fh:
        fh.write(built.emit().encode('latin-1'))
    if log:
        log('compiled %s' % out)
    return out

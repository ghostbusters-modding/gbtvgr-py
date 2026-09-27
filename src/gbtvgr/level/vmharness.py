"""The headless run_vm harness: pooled spawn, bounded runner, walk driver and
the GameWorld subclass every level's offline VM proof reuses."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import signal

from dante import module as dante_module
from dante.vm import machine as dantevm
from dante.vm.world import GameWorld

from .. import lvl as _lvl
from . import wire


def read_pool(lvl_path):
    """The level's dormant spawn stock, as the engine sees it: every
    createStatus=0 CCharacter with a loaded .cib, keyed by the triple
    CSpawn::spawnCharacter matches on."""
    import re
    raw = open(lvl_path, 'rb').read().decode('latin1')
    cls_of = dict(re.findall(r'^\t(\w+) = (\w+)\r$',
                             re.search(r'<actor-list>\r\n(.*?)</actor-list>',
                                       raw, re.S).group(1), re.M))
    pool = []
    for m in re.finditer(r'\t<(\w+)>\r\n(.*?)\t</\1>\r\n', raw, re.S):
        name, body = m.group(1), m.group(2)
        f = dict(re.findall(r'^\t\t(\w+) = (.*)\r$', body, re.M))
        if 'charInfoFilename' not in f or f.get('createStatus') != '0':
            continue
        cit = f['charInfoFilename'].replace('\\\\', '\\')
        if cit.lower().startswith('gb_'):
            continue          # a parked hero is not spawn stock
        pool.append({'name': name, 'cls': cls_of.get(name, '?'),
                     'cit': cit.lower(),
                     'variant': (f.get('charInfoVariantName') or '').lower(),
                     'busy': None})
    return pool


def read_pool_level(level):
    """`read_pool`, off a wire.Level: the dormant pool across .lvl + every
    .sec (the engine's flat actor table), not a single .lvl path."""
    maps = wire.actor_classes(level)
    pool = []
    for si, actor in wire.actors(level):
        f = _lvl.actor_fields(actor)
        if 'charInfoFilename' not in f or f.get('createStatus') != '0':
            continue
        cit = f['charInfoFilename'].replace('\\\\', '\\')
        if cit.lower().startswith('gb_'):
            continue
        pool.append({'name': actor.tag, 'cls': maps[si].get(actor.tag, '?'),
                     'cit': cit.lower(),
                     'variant': (f.get('charInfoVariantName') or '').lower(),
                     'busy': None})
    return pool


class Hang(Exception):
    pass


def run_bounded(vm, ticks, timeout_s=60):
    """A script that never reaches idle() spins run_thread() forever, exactly
    as it would hang the game, so the test itself must be bounded rather
    than hanging."""
    def _on_alarm(signum, frame):
        raise Hang('no idle() within %ds' % timeout_s)
    old = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(timeout_s)
    try:
        vm.run(ticks=ticks)
        return True
    except Hang:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


class ScenarioWorld(GameWorld):
    """A headless level, walked and fired by a route/waypoint list.  Kills
    every spawn `kill_after` ticks after it appears and records every
    INTERESTING native call for the assertions to read back.

    `member_of` maps an `extern CActorGroup g_x;` name to the fake member
    actor its .enable() calls should be visible on (a door group's member
    carries the door prop's own name); `interesting` is the native-call
    names worth recording at all."""

    def __init__(self, member_of, interesting, route=(), kill_after=45,
                 spawns_fail=False, verbose=True, pool=None, walk=None,
                 hero_pos=(0.0, 0.0, 0.0)):
        GameWorld.__init__(self)
        self.member_of = member_of
        self.interesting = interesting
        # WHERE THE HERO IS STANDING matters twice: a resume guard reads it
        # (a "resume" with the hero at the muster is a fresh level the
        # engine attached an old save slot to), and watchThread() point-tests
        # it against every beat volume at 30 Hz.
        self.walk = list(walk or [])
        self.hero_pos = hero_pos
        self.kill_after = kill_after
        self.verbose = verbose
        self.events = []          # (tick, native name, args, who)
        self.route = list(route)
        self.fired = []
        self.pool = pool or []
        self.spawn_misses = []    # (cls, cit, variant) the pool cannot answer
        # THE CAMERA MODE.  The base world has no model for CGameView, so the
        # three mode calls the level makes are stubbed here and the mode is
        # tracked: setCameraModeNormal is the only way out of an orbit or a
        # fixed post, and the run must end in normal mode.
        self.camera = 'normal'
        self.natives['void CGameView::setCameraModeFixed{@CGameView}'
                     '(@CCameraPathActor,@CActor,float)'] = \
            lambda vm, this, cam, target, blend: self._cam('fixed')
        self.natives['void CGameView::setCameraModeOrbit{@CGameView}'
                     '(@CActor,float,float,float,float,float,float)'] = \
            lambda vm, this, *a: self._cam('orbit')
        self.natives['void CGameView::setCameraModeNormal{@CGameView}'
                     '(float,float)'] = lambda vm, this, *a: self._cam('normal')
        if spawns_fail:
            self.natives['@CCharacter CSpawn::spawnCharacter'
                         '{@CSpawn}(@SSpawnInfo)'] = \
                lambda vm, this, info: None
        elif self.pool:
            self.natives['@CCharacter CSpawn::spawnCharacter'
                         '{@CSpawn}(@SSpawnInfo)'] = self._pooled_spawn

    def _cam(self, mode):
        self.camera = mode

    def make_global(self, vm, typ, name):
        """An `extern CActorGroup g_x;` comes up with ONE member, so that
        enableActorGroup(g_x, b) -- `.enable(b)` over every member -- shows
        in the event log as an enable() on that member."""
        if typ == 'CActorGroup':
            obj = self.make_object(vm, typ, name)
            member = self.make_object(vm, 'CProp',
                                      self.member_of.get(name, name + '#m'))
            obj.props['members'] = [dantevm.Ptr(member, 0)]
            return obj
        return GameWorld.make_global(self, vm, typ, name)

    def call_native(self, vm, thread, proto, this, args):
        ret = GameWorld.call_native(self, vm, thread, proto, this, args)
        if proto.name in self.interesting:
            who = getattr(getattr(this, 'obj', None), 'name', None)
            self.events.append((vm.tick_no, proto.name, tuple(
                a if isinstance(a, (str, int, float)) else
                (a.obj.name if getattr(a, 'obj', None) is not None else '?')
                for a in args), who))
            if self.verbose:
                print('  t=%5d %-20s %s%s'
                      % (vm.tick_no, proto.name,
                         ('%s.' % who) if who else '', self.events[-1][2]))
        return ret

    def _pooled_spawn(self, vm, this, info):
        """CSpawn::spawnCharacter as the engine implements it: walk the level
        actor list for a FREE pooled CCharacter whose class, .cit and variant
        match the request (stricmp), and hand that one back.  No match, or
        every match already alive, means null -- the same silence the level
        got in-game."""
        cls = vm.struct_get(info, 'SSpawnInfo', 'classTypeName') or ''
        cit = vm.struct_get(info, 'SSpawnInfo', 'citFilename') or ''
        var = vm.struct_get(info, 'SSpawnInfo', 'variantName') or ''
        slot = None
        for e in self.pool:
            if (e['cls'].lower() == cls.lower()
                    and e['cit'] == cit.lower()
                    and e['variant'] == var.lower()):
                if e['busy'] is None:
                    slot = e
                    break
                slot = slot or False          # matched, but all of them alive
        if not slot:
            self.spawn_misses.append((cls, cit, var, slot is False))
            return None
        obj = GameWorld._spawn_character(self, vm, this, info)
        slot['busy'] = obj.obj if isinstance(obj, dantevm.Ptr) else obj
        obj.obj.props['pool_slot'] = slot['name']
        return obj

    def place_hero(self, vm, pos):
        hero = vm.resolve_global('CGhostbuster Ghostbuster0', None)
        self.state(hero)['pos'] = tuple(float(v) for v in pos)
        self.hero_pos = tuple(float(v) for v in pos)

    def on_tick(self, vm):
        GameWorld.on_tick(self, vm)
        while self.walk and self.walk[0][0] <= vm.tick_no:
            _t, pos = self.walk.pop(0)
            if self.verbose:
                print('  t=%5d [walk] %s' % (vm.tick_no, (pos,)))
            self.place_hero(vm, pos)
        for e in self.pool:            # a dead enemy goes back in the pool
            if e['busy'] is not None and e['busy'].props.get('dead'):
                e['busy'] = None
        while self.route and self.route[0][0] <= vm.tick_no:
            _t, name = self.route.pop(0)
            self.fire(vm, name)
        for obj in list(self.spawned):
            if obj.props.get('dead'):
                continue
            obj.props.setdefault('seen_tick', vm.tick_no)
            if vm.tick_no - obj.props['seen_tick'] >= self.kill_after:
                self.kill(obj)

    def fire(self, vm, name):
        """The engine calls a CTrigger's actorEnterEvent by export name; do
        the same, with the trigger and the hero as its two arguments.  A
        `trinket_` route entry is the other engine event a level can depend
        on -- CTrinket's pickedUpEvent."""
        hero = vm.resolve_global('CGhostbuster Ghostbuster0', None)
        if name.startswith('trinket_'):
            this = vm.resolve_global('CTrinket %s' % name, None)
            proto = 'void %s_pickedUp(@CTrinket,@CGhostbuster)' % name
        else:
            this = vm.resolve_global('CTrigger %s' % name, None)
            proto = 'void %s_onActorEnter(@CTrigger,@CActor)' % name
        if self.verbose:
            print('  t=%5d [player] %s' % (vm.tick_no, name))
        vm.call(proto, this, hero)
        self.fired.append(name)


def check(cond, what, fails):
    print('  %s %s' % ('ok  ' if cond else 'FAIL', what))
    if not cond:
        fails.append(what)


def build(module, libs, world):
    vm = dantevm.VM([dantevm.load(p) for p in libs + [module]], world=world)
    vm.init_modules()
    return vm


def build_level(level, lib_paths, world):
    """`build`, off a wire.Level's compiled script instead of a module path;
    `lib_paths` (global.dante and friends) are still real files on disk."""
    mod = dante_module.Dante(level.script.decode('latin-1'), level.stem)
    vm = dantevm.VM([dantevm.load(p) for p in lib_paths] + [mod], world=world)
    vm.init_modules()
    return vm


def names(world, native):
    return [e for e in world.events if e[1] == native]


def enables(events, who):
    """[(tick, bool)] of every enable() on the actor `who`."""
    return [(e[0], bool(e[2][0])) for e in events
            if e[1] == 'enable' and e[3] == who]

def set_global(vm, sym, value, module=None):
    """Write a module global such as 'int gSpawnMarks' before a run: the shipped
    default stays quiet, the offline proof still reads the HUD marks."""
    mod = module or [m for m in vm.modules if m.name != 'global'][-1]
    vm.resolve_global(sym, mod).set(value)

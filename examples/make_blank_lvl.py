#!/usr/bin/env python3
"""
Author world\\immortal1.lvl from scratch -- every line generated.

Not a donor derivation: the file is composed section by section. Only three
things are lifted verbatim, deliberately -- the class schema-version constants
(engine-wide, not level content), the player and spawner actor blocks with
their event fields blanked, and the dependency lines for the asset families.

The set-filename still names cemetery2's .bst because a set is mandatory; the
level hides it, with everything at y=+100 wrapped in dense fog.

Usage: make_blank_lvl.py <donor.lvl> <manifest.json> [<manifest2.json>...] -o <out.lvl>
"""
import json
import re
import sys

CLASSES = ('CProp', 'CGhostbuster', 'CSpawn')
COAL_CLASSES = ('CGolemPartBase', 'CPhysicsObject', 'CEmitter')
SET_NAME = 'cemetery2.pst'
DEP_PATS = ('gb_player', 'ghostbuster', 'scuttler', 'gravestone', 'weap',
            'proton', 'trap', 'civh', 'goggle', 'slimeblower', 'fiend',
            'crawler', 'dirt_explosion')
# Preload lines copied verbatim from the shipped levels that use them.
# possessed_human is never .lvl-dep'd: possessors spawn them at runtime.
EXTRA_DEPS = [
    'data\\\\scuttler\\\\Mini.cit', 'data\\\\scuttler\\\\Mini.civh',
    'fx\\\\mini_impact.tfa', 'fx\\\\explosion_puff_mini.tfa',
    'fx\\\\fire_marshmallow_mini.tfa', 'fx\\\\fire_mini_throw.tfa',
    'materials\\\\marshmallow_mini_burning.mtl',
    'materials\\\\marshmallow_mini_burnt.mtl',
    'data\\\\scuttler\\\\crawler_spider.cit',
    'data\\\\biped1\\\\fiend_wb2.cit',
    'data\\\\floater\\\\Lobber_Civil_War_Union.cit',
    'data\\\\floater\\\\Pummeler_Civil_War_Confederate.cit',
    'fx\\\\glow_civil_war_ghost_musket.tfa', 'fx\\\\glow_civil_war_ghost_sword.tfa',
    'physics\\\\civil_war_head.phys2',
    'skeletal\\\\floater\\\\civil_war_ghost_musket.dfm',
    'skeletal\\\\floater\\\\civil_war_ghost_sword.dfm',
    'data\\\\flyer_small\\\\Zoomer_Flying_Skull.cit',
    'skeletal\\\\flyer_med\\\\flying_skull.dfm',
    # stone angel pool (flyer_med family, same donor as the flying skulls)
    'data\\\\flyer_med\\\\Tosser.civh',
    'data\\\\Flyer_Med\\\\Tosser_Stone_Angel.cit',
    'skeletal\\\\flyer_med\\\\stone_angel.dfm',
    # DEP_PATS already pulls the shared ghostbuster assets in; only the per-hero
    # .cit lines are missed. Listed anyway -- 'seen' dedupes.
    'data\\\\gb_stantz.cit', 'skeletal\\\\ghostbuster\\\\stantz.dfm',
    'skeletal\\\\ghostbuster\\\\stantz.jug',
    'animations\\\\ghostbuster_mocap\\\\stantz\\\\facefx\\\\gb_stantz.fxa',
    'data\\\\gb_spengler.cit', 'skeletal\\\\ghostbuster\\\\spengler.dfm',
    'skeletal\\\\ghostbuster\\\\spengler.jug',
    'animations\\\\ghostbuster_mocap\\\\spengler\\\\facefx\\\\gb_spengler.fxa',
    'data\\\\gb_zeddemore.cit', 'skeletal\\\\ghostbuster\\\\zeddemore.dfm',
    'skeletal\\\\ghostbuster\\\\zeddemore.jug',
    'animations\\\\ghostbuster_mocap\\\\zeddemore\\\\facefx\\\\gb_zeddemore.fxa',
    'data\\\\gb_venkman.cit', 'skeletal\\\\ghostbuster\\\\venkman.dfm',
    'skeletal\\\\ghostbuster\\\\venkman.jug',
    'animations\\\\ghostbuster_mocap\\\\venkman\\\\facefx\\\\gb_venkman.fxa',
    # Deliberately not the ghost trap: a spinning weap_trap reads as "a trap you
    # can pick up", the one thing it is not. These are collectible-artifact props.
    'models\\\\artifacts\\\\envCollect_40.smf',   # mayan armour  -> BARRIER
    'models\\\\artifacts\\\\envCollect_51.smf',   # ecto jar      -> RECHARGE
    'models\\\\artifacts\\\\envCollect_41.smf',   # civil-war guns-> TORPEDO
    'models\\\\artifacts\\\\envCollect_64.smf',   # swiss clock   -> STASIS
    # the coal fires burning in the mounds (the coal-donor dep filter only
    # takes lines with 'coal'/'golem' in them, and this one has neither)
    'fx\\\\fire_pillar.tfa',
    # The POD stores .smb while .lvl deps and modelInstance name the sibling
    # .smf -- the same convention weap_trap follows above.
    'models\\\\graveyard\\\\fence_boyd.smf',
    'models\\\\graveyard\\\\redeyes.smf',
    'models\\\\graveyard\\\\spike_bottom.smf',
]
HERO_POS = '0.00, 100.60, 0.00'
SPAWNER_POS = '8.00, 100.05, -8.00'
# Companion ghostbusters, from donor blocks already hero-versioned. They muster
# off-centre because the arena centre is the Great Furnace.
COMPANIONS = [
    ('Ray', '6.00, 200.60, -24.00', '-135, 0, 0'),
    ('Egon', '-6.00, 200.60, -24.00', '135, 0, 0'),
    ('Winston', '6.00, 200.60, -36.00', '-45, 0, 0'),
    ('Peter', '-6.00, 200.60, -36.00', '45, 0, 0'),
]
# Powerup stations (actor, pos, model): two on the plateaus, a lap away, two on
# the ziggurat plaza. The script keeps them disabled until a kill count earns one.
POWERUPS = [
    ('imPow_NE', '72.00, 210.90, 0.00', 'artifacts\\\\envCollect_40.smf'),
    ('imPow_NW', '-72.00, 210.90, 0.00', 'artifacts\\\\envCollect_51.smf'),
    ('imPow_SE', '48.00, 200.90, -72.00', 'artifacts\\\\envCollect_41.smf'),
    ('imPow_SW', '-48.00, 200.90, -72.00', 'artifacts\\\\envCollect_64.smf'),
]
# The donor declares CSpinner in its actor-version-list but ships no actor block
# to lift, so this is hardcoded from two levels that do.
SPINNER_VERSION = '9'
# Hardcoded (name, model, pos, yaw): no donor block exists to lift. The shipped
# gravestones are physics chunks, so these are the closest static CProp models.
IMMORTAL_DECOR = [
    ('imDecorGrave1', 'graveyard\\fence_boyd.smf', (-80.00, 199.00, -87.00), '15'),
    ('imDecorGrave2', 'graveyard\\spike_bottom.smf', (-48.00, 199.00, -88.00), '200'),
    ('imDecorGrave3', 'graveyard\\redeyes.smf', (-16.00, 199.00, -87.00), '90'),
    ('imDecorGrave4', 'graveyard\\fence_boyd.smf', (18.00, 199.00, -88.00), '260'),
    ('imDecorGrave5', 'graveyard\\redeyes.smf', (82.00, 199.00, -86.00), '330'),
    ('imDecorGrave6', 'graveyard\\fence_boyd.smf', (-87.00, 199.00, -60.00), '90'),
    ('imDecorGrave7', 'graveyard\\spike_bottom.smf', (-88.00, 199.00, -10.00), '95'),
    ('imDecorGrave8', 'graveyard\\redeyes.smf', (-87.00, 199.00, 45.00), '85'),
    ('imDecorGrave9', 'graveyard\\fence_boyd.smf', (28.00, 199.00, -60.00), '180'),
    ('imDecorGrave10', 'graveyard\\spike_bottom.smf', (-28.00, 199.00, -60.00), '0'),
]


def block(raw, name):
    m = re.search(r'^\t<%s>\r\n.*?^\t</%s>\r\n' % (name, name), raw, re.S | re.M)
    return m.group(0)


def set_field(blk, field, value):
    # function replacement: value is LITERAL (re.sub would eat backslashes)
    return re.sub(r'^(\t\t%s = )[^\r\n]*' % field,
                  lambda mm: mm.group(1) + value, blk, flags=re.M)


def blank_events(blk):
    return re.sub(r'^(\t\t\w+ = )[A-Za-z@][^=\r\n]*\([^\r\n]*\)(?=\r?$)', r'\g<1>""',
                  blk, flags=re.M)


def prop_block(name, pos, yaw, model):
    # restOnActorValid must be 0: 1 settles the prop onto whatever is below it,
    # which on a void platform means a 100-unit drop onto the donor terrain.
    f = [('name', name), ('createStatus', '1'),
         ('pos', '%.2f, %.2f, %.2f' % tuple(pos)),
         ('orient', '0, %s, 0' % yaw), ('processRadius', '250'),
         ('damageFilterEvent', '""'), ('restOnActorValid', '0'),
         ('onSnaredStatusChangedEvent', '""'),
         ('modelInstance', model.replace('\\', '\\\\')),
         ('useCollisionParts', '1'), ('propFlags', '0')]
    return ('\t<%s>\r\n' % name + ''.join('\t\t%s = %s\r\n' % kv for kv in f)
            + '\t</%s>\r\n' % name)


def spinner_block(name, pos, model):
    # field set verified against the shipped boss_sp_side.lvl / abyss.lvl
    # CSpinner actors (no CSpinner block exists in cemetery3.lvl to donate)
    f = [('name', name), ('createStatus', '1'), ('pos', pos),
         ('orient', '0, 0, 0'), ('processRadius', '250'),
         ('damageFilterEvent', '""'), ('restOnActorValid', '0'),
         ('onSnaredStatusChangedEvent', '""'), ('modelType', '0'),
         ('modelInstance', model),
         ('spinUpWavName', '""'), ('loopWavName', '""'),
         ('spinDownWavName', '""'), ('axisOfRotationType', '0'),
         ('axisOfRotation', '0, 1, 0'), ('rpm', '25'),
         ('spinUpTime', '0'), ('spinDownTime', '0'), ('groundType', '0'),
         ('plotInShadowFlag', '0'), ('collisionType', '0'),
         ('emitLight', '0'), ('spotTex', 'lights\\\\default_light.tga'),
         ('spotFOV', '28'), ('blendMode', '1')]
    return ('\t<%s>\r\n' % name + ''.join('\t\t%s = %s\r\n' % kv for kv in f)
            + '\t</%s>\r\n' % name)


POOL_SEEDS = [
    # (basename, count, class, donor block, cit override, variant). createStatus=0
    # actors are the spawn stock: spawning recycles one matching class/cit/variant.
    ('imPoolCrawler', 10, 'CScuttler', 'Scuttler3', None, None),
    ('imPoolMini', 16, 'CScuttler', 'Scuttler3', 'scuttler\\\\Mini.cit', None),
    ('imPoolSpider', 10, 'CScuttler', 'Scuttler3',
     'scuttler\\\\crawler_spider.cit', None),
    ('imPoolFiend', 8, 'CBiped', 'biped_GraveFiend1', None, None),
    # Not possessed_human: that is the scripted possession-fight entity and is
    # unkillable by weapons. fiend_wb2 is the webby fiend clone.
    ('imPoolWebby', 5, 'CBiped', 'biped_GraveFiend1',
     'biped1\\\\fiend_wb2.cit', None),
    ('imPoolUnion', 5, 'CFloater', 'floater_Cultist1',
     'floater\\\\Lobber_Civil_War_Union.cit', None),
    ('imPoolReb', 5, 'CFloater', 'floater_Cultist1',
     'floater\\\\Pummeler_Civil_War_Confederate.cit', None),
    # The skull donor ships variant cultist_skulls, but the cib also has standard,
    # which is what the script actually asks for.
    ('imPoolSkull', 6, 'CFlyerSmall', 'flyerSmall_Skull5', None, 'standard'),
    # stone angels: donor block already carries the right cit + variant
    # (Flyer_Med\\Tosser_Stone_Angel.cit / cemetery2) -- no overrides needed
    ('imPoolAngel', 5, 'CFlyerMedium', 'flyerMed_StoneAngel3', None, None),
]

def pool_seeds(raw):
    acts = []
    n = 0
    for base, count, cls, donor_tag, cit, variant in POOL_SEEDS:
        for i in range(count):
            name = '%s%d' % (base, i + 1)
            b = block(raw, donor_tag).replace(donor_tag, name)
            b = blank_events(b)
            b = set_field(b, 'createStatus', '0')
            b = set_field(b, 'pos', '%.2f, 199.30, %.2f'
                          % (-76.0 + 8.0 * (n % 20), 92.0 - 3.0 * (n // 20)))
            if cit:
                b = set_field(b, 'charInfoFilename', cit)
            if variant:
                b = set_field(b, 'charInfoVariantName', variant)
            acts.append((name, cls, b))
            n += 1
    return acts


# ---- the coal economy.  480 chunks, ~320 of them lit -- about 4x the biggest
# shipped field, so these are the numbers to turn down first if frames suffer.
COAL_CORNERS = ((75.0, 75.0), (-75.0, 75.0), (75.0, -75.0), (-75.0, -75.0))
COAL_MOUNDS_PER_CORNER = 2
COAL_SCALE = 4
COAL_PER_MOUND = 15
COAL_LIT_NUM, COAL_LIT_DEN = 2, 3      # 2 of every 3 chunks are lit = red
# The centre furnace is a bed of coal, not a building: ankle-height physics
# props have no nav footprint, so the crossroads stays walkable.
COAL_CENTRE_MOUNDS = 2
COAL_CENTRE_RADIUS = 10.0


def coal_field(coal_raw):
    """Coal economy blocks, arena-local: a field, a parts cache and a fire per
    corner, so each golem wakes up standing in its own fuel."""
    import math, random
    rng = random.Random(20260831)
    acts = []
    phys_t = block(coal_raw, 'physO_CoalB_P01')

    def chunk(name, pos, defn):
        b = phys_t.replace('physO_CoalB_P01', name)
        b = set_field(b, 'pos', '%.2f, %.2f, %.2f' % pos)
        b = set_field(b, 'orient', '%d, 0, 0' % rng.randrange(0, 360))
        b = set_field(b, 'definitionName', defn)
        b = set_field(b, 'restOnActorValid', '1')
        return b

    n = 0
    nmound = COAL_MOUNDS_PER_CORNER * COAL_SCALE
    for mx, mz in COAL_CORNERS:
        for m in range(nmound):
            # Sub-mound centres on a 10ft ring, so the field stays inside x,z 59..91
            # of its corner -- clear of the pillars, beams, plateaus and walls.
            ang = 6.28318 * m / nmound
            sx = mx + 10.0 * math.cos(ang)
            sz = mz + 10.0 * math.sin(ang)
            for i in range(COAL_PER_MOUND):
                a = rng.uniform(0, 6.28318)
                r = rng.uniform(0.5, 6.0)
                lay = 0.35 + 0.75 * (i % 2)
                n += 1
                lit = (n % COAL_LIT_DEN) < COAL_LIT_NUM
                kind = 'lit' if lit else 'plain'
                idx = (n % (4 if lit else 6)) + 1
                defn = 'coal_golem\\\\coal_%s_%02d.phys2' % (kind, idx)
                nm = 'imCoal%03d' % n
                acts.append((nm, 'CPhysicsObject',
                             chunk(nm, (sx + r * math.cos(a), 199.0 + lay,
                                        sz + r * math.sin(a)), defn)))
    # the centre bed: the fires that used to sit on the Great Furnace now
    # burn here, in a ring of coal you can walk straight through
    for m in range(COAL_CENTRE_MOUNDS):
        ang = 6.28318 * m / COAL_CENTRE_MOUNDS
        sx = COAL_CENTRE_RADIUS * math.cos(ang)
        sz = COAL_CENTRE_RADIUS * math.sin(ang)
        for i in range(COAL_PER_MOUND):
            a = rng.uniform(0, 6.28318)
            r = rng.uniform(0.5, 6.0)
            lay = 0.35 + 0.75 * (i % 2)
            n += 1
            lit = (n % COAL_LIT_DEN) < COAL_LIT_NUM
            kind = 'lit' if lit else 'plain'
            idx = (n % (4 if lit else 6)) + 1
            defn = 'coal_golem\\\\coal_%s_%02d.phys2' % (kind, idx)
            nm = 'imCoal%03d' % n
            acts.append((nm, 'CPhysicsObject',
                         chunk(nm, (sx + r * math.cos(a), 199.0 + lay,
                                    sz + r * math.sin(a)), defn)))
    # One standard-part cache per golem, along that corner's outer wall, so every
    # golem has an unclaimed set within reach of its own furnace.
    g = p = l = 0
    for ci, (mx, mz) in enumerate(COAL_CORNERS):
        row_z = 92.0 if mz > 0 else -92.0
        k = 0
        for src_name in ('golemPart_CoalGrate1', 'golemPart_CoalGrate2'):
            g += 1
            nm = 'imGrate%d' % g
            b = block(coal_raw, src_name).replace(src_name, nm)
            b = set_field(b, 'pos', '%.2f, 199.30, %.2f'
                          % (mx + (k - 5.5) * 2.5, row_z))
            acts.append((nm, 'CGolemPartBase', b))
            k += 1
        for i in range(6):
            p += 1
            nm = 'imTplP%d' % p
            src_name = 'golemPart_CoalPlain%d' % (i + 1)
            b = block(coal_raw, src_name).replace(src_name, nm)
            b = set_field(b, 'pos', '%.2f, 199.30, %.2f'
                          % (mx + (k - 5.5) * 2.5, row_z))
            acts.append((nm, 'CPhysicsObject', b))
            k += 1
        for i in range(4):
            l += 1
            nm = 'imTplL%d' % l
            src_name = 'golemPart_CoalLit%d' % (i + 1)
            b = block(coal_raw, src_name).replace(src_name, nm)
            b = set_field(b, 'pos', '%.2f, 199.30, %.2f'
                          % (mx + (k - 5.5) * 2.5, row_z))
            acts.append((nm, 'CPhysicsObject', b))
            k += 1
    # The donor ships createStatus = 0 because it lights its coal from a trigger.
    # These are created with the level instead: one per corner, one at the centre.
    if '<emit_CoalFire1>' in coal_raw:
        fire_t = block(coal_raw, 'emit_CoalFire1')
        fires = [('%.2f, 199.60, %.2f' % (mx, mz)) for mx, mz in COAL_CORNERS]
        for m in range(COAL_CENTRE_MOUNDS):
            ang = 6.28318 * m / COAL_CENTRE_MOUNDS
            fires.append('%.2f, 199.60, %.2f'
                         % (COAL_CENTRE_RADIUS * math.cos(ang),
                            COAL_CENTRE_RADIUS * math.sin(ang)))
        for i, pos in enumerate(fires):
            nm = 'imCoalFire%d' % (i + 1)
            b = fire_t.replace('emit_CoalFire1', nm)
            b = set_field(b, 'createStatus', '1')
            b = set_field(b, 'pos', pos)
            acts.append((nm, 'CEmitter', b))
    return acts


def main():
    args = sys.argv[1:]
    out = args[args.index('-o') + 1]
    set_name = SET_NAME
    if '--set' in args:
        set_name = args[args.index('--set') + 1]
    hero_pos = HERO_POS
    if '--hero-pos' in args:
        hero_pos = args[args.index('--hero-pos') + 1]
    no_spawner = '--no-spawner' in args
    spawners = []
    if '--spawners' in args:
        for tok in args[args.index('--spawners') + 1].split(';'):
            nm, pos = tok.split('@')
            spawners.append((nm, pos))
    coal_donor = None
    if '--coal-donor' in args:
        coal_donor = args[args.index('--coal-donor') + 1]
    skip = {out, set_name, hero_pos}
    if '--spawners' in args: skip.add(args[args.index('--spawners') + 1])
    if coal_donor: skip.add(coal_donor)
    donor = args[0]
    manifests = [a for a in args[1:] if not a.startswith('-') and a not in skip]

    raw = open(donor, 'rb').read().decode('latin1')

    # verbatim engine constants: schema versions for the classes we instantiate
    vers = {}
    for c in CLASSES:
        m = re.search(r'^\t%s = (\d+)\r$' % c, raw, re.M)
        vers[c] = m.group(1)

    # dependency families
    dep_start = raw.index('Dependencies = <')
    dep_end = raw.index('>', dep_start)
    donor_deps = [l.strip() for l in raw[dep_start:dep_end].split('\r\n')[1:] if l.strip()]
    deps, seen = [], set()
    for d in donor_deps:
        if any(p in d.lower() for p in DEP_PATS) and d not in seen:
            deps.append(d); seen.add(d)

    # imported actor blocks
    hero = blank_events(set_field(block(raw, 'Ghostbuster0'), 'pos', hero_pos))
    # Session hero binding is a name lookup at level prepare, so a second named
    # CGhostbuster is all a level needs to host two players.
    hx, hy, hz = [float(v) for v in hero_pos.split(',')]
    hero2 = blank_events(set_field(block(raw, 'Ghostbuster0'), 'pos',
                                   '%.2f, %.2f, %.2f' % (hx + 3.0, hy, hz)))
    # Invulnerable until health sync exists: an unmanned second hero dying to the
    # waves can fail the mission, and a puppet's health belongs to its peer.
    hero2 = set_field(hero2, 'invulnerableFlag', '1')
    hero2 = hero2.replace('Ghostbuster0', 'Ghostbuster1')
    sp = block(raw, 'spEmit_FirstWaveScuttlers1')
    sp = sp.replace('spEmit_FirstWaveScuttlers1', 'immortalSpawner')
    sp = blank_events(set_field(sp, 'pos', SPAWNER_POS))
    sp = set_field(sp, 'restOnActorValid', '0')      # do not settle into the void

    # our actors from the manifest(s)
    actors = [('Ghostbuster0', 'CGhostbuster', hero),
              ('Ghostbuster1', 'CGhostbuster', hero2)]
    if not no_spawner:
        actors.append(('immortalSpawner', 'CSpawn', sp))
    for nm, pos in spawners:
        b = block(raw, 'spEmit_FirstWaveScuttlers1')
        b = b.replace('spEmit_FirstWaveScuttlers1', nm)
        b = blank_events(set_field(b, 'pos', pos))
        b = set_field(b, 'restOnActorValid', '0')
        actors.append((nm, 'CSpawn', b))
    seed_classes = ('CScuttler', 'CBiped', 'CFloater', 'CFlyerSmall',
                     'CBipedLarge', 'CFlyerMedium', 'CWayPoint')
    for c in seed_classes:
        mm = re.search(r'^\t%s = (\d+)\r$' % c, raw, re.M)
        vers[c] = mm.group(1)
    vers['CSpinner'] = SPINNER_VERSION
    actors += pool_seeds(raw)

    # companion ghostbusters: lift Ray/Egon/Winston/Peter verbatim, park
    # createStatus=1, and place them around the hero
    for nm, pos, orient in COMPANIONS:
        b = blank_events(block(raw, nm))
        b = set_field(b, 'createStatus', '1')
        b = set_field(b, 'pos', pos)
        b = set_field(b, 'orient', orient)
        actors.append((nm, 'CGhostbuster', b))

    # super-trap waypoint
    wp = block(raw, 'wp_CoffinWaypoint1').replace('wp_CoffinWaypoint1', 'wpSuperTrap')
    wp = blank_events(wp)
    wp = set_field(wp, 'createStatus', '1')
    wp = set_field(wp, 'pos', '24.00, 199.30, -30.00')
    wp = set_field(wp, 'orient', '0, 0, 0')
    actors.append(('wpSuperTrap', 'CWayPoint', wp))

    # powerup stations (a different model per kind, none of them a trap)
    for nm, spos, model in POWERUPS:
        actors.append((nm, 'CSpinner', spinner_block(nm, spos, model)))

    # decorative graveyard props -- static, non-colliding (deco must not
    # block movement or the navmesh, which does not exclude these spots)
    for nm, model, pos, yaw in IMMORTAL_DECOR:
        b = prop_block(nm, pos, yaw, model)
        b = set_field(b, 'useCollisionParts', '0')
        actors.append((nm, 'CProp', b))

    coal_classes_used = ()
    if coal_donor:
        coal_raw = open(coal_donor, 'rb').read().decode('latin1')
        for c in COAL_CLASSES:
            mm = re.search(r'^\t%s = (\d+)\r$' % c, coal_raw, re.M)
            vers[c] = mm.group(1)
        coal_classes_used = COAL_CLASSES
        dep_s = coal_raw.index('Dependencies = <')
        for l in coal_raw[dep_s:coal_raw.index('>', dep_s)].split('\r\n')[1:]:
            l = l.strip()
            if l and ('coal' in l.lower() or 'golem' in l.lower()) and l not in seen:
                deps.append(l); seen.add(l)
        for l in EXTRA_DEPS:
            if l not in seen:
                deps.append(l); seen.add(l)
        # Four dormant golems, one per corner field. The engine matches the pool
        # triple by the cit each one resolves, not by this clone path.
        for gi, (mx, mz) in enumerate(COAL_CORNERS):
            nm = 'imPoolGolem%d' % (gi + 1)
            b = block(coal_raw, 'BipedLarge_CoalClone1')
            b = blank_events(b.replace('BipedLarge_CoalClone1', nm))
            b = set_field(b, 'pos', '%.2f, 199.30, %.2f' % (mx, mz))
            actors.append((nm, 'CBipedLarge', b))
        actors += coal_field(coal_raw)
    for mf in manifests:
        man = json.load(open(mf))
        for m in man.get('models', []):
            line = 'models\\\\' + m.replace('\\', '\\\\')
            if line not in seen:
                deps.append(line); seen.add(line)
        for a in man['actors']:
            model = a['fields'].get('modelInstance', '')
            actors.append((a['name'], 'CProp',
                           prop_block(a['name'], a['pos'], a['yaw'], model)))

    L = []
    L.append('version = 19')
    L.append('set-filename = %s' % set_name)
    L.append('Dependencies = <')
    L += ['\t' + d for d in deps]
    L.append('>')
    L.append('Dante-Includes = <')
    L += ['\textern %s %s;' % (cls, name) for name, cls, _ in actors]
    L.append('>')
    L.append('<actor-version-list>')
    L += ['\t%s = %s' % (c, vers[c])
          for c in CLASSES + seed_classes + ('CSpinner',) + tuple(coal_classes_used)]
    L.append('</actor-version-list>')
    L.append('<actor-list>')
    L += ['\t%s = %s' % (name, cls) for name, cls, _ in actors]
    L.append('</actor-list>')
    L.append('<list-of-heros>')
    L.append('\thero_0 = Ghostbuster0')
    L.append('</list-of-heros>')
    txt = '\r\n'.join(L) + '\r\n'
    txt += '<actors>\r\n'
    for _, _, blk in actors:
        txt += blk
    txt += '</actors>\r\n'
    for tag in ('actor-groups', 'selection-sets-for-editor-only',
                'cinemat-version-list', 'cinemat-list', 'cinemats'):
        txt += '<%s>\r\n</%s>\r\n' % (tag, tag)
    open(out, 'wb').write(txt.encode('latin1'))
    print('blank level: %d deps, %d actors -> %s' % (len(deps), len(actors), out))


if __name__ == '__main__':
    main()

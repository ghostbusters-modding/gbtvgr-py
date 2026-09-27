"""The placement API: floors off a shipped BVT, prop bounds, actor overrides, the squad,
the lang table builder, compile_level's new knobs and the register-indirect member rule."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import glob
import os
import subprocess
import sys

import pytest

from gbtvgr import lvl as glvl
from gbtvgr.level import checks, lang, levelfile, place, placement, wire
from gbtvgr.sets import bst

# cemetery1 section 42 (Room25) in its own frame: the graft1 hero, PKE and trigger floors,
# read back out of the shipped graft1.lvl minus their clearances and the (-400, 30, 150) lift
KNOWN_FLOORS = [((77.0, -106.0), -30.068345), ((71.0, -120.0), -26.363774),
                ((68.0, -132.5), -25.802831)]


class _StubLibrary:
    """read_level/level_stems over a loose world/ dir, so no POD chain is needed."""

    def __init__(self, world_dir):
        self.world_dir = world_dir

    def read_level(self, stem):
        p = os.path.join(self.world_dir, stem + '.lvl')
        if not os.path.isfile(p):
            return None
        with open(p, 'rb') as fh:
            return fh.read()

    def level_stems(self):
        return sorted(f[:-4] for f in os.listdir(self.world_dir) if f.lower().endswith('.lvl'))


@pytest.fixture(scope='module')
def templates(lvl_corpus):
    return levelfile.TemplateLibrary(_StubLibrary(os.path.join(lvl_corpus, 'world')))


@pytest.fixture(scope='module')
def cemetery1(bst_corpus):
    path = os.path.join(bst_corpus, 'cemetery1.bst')
    if not os.path.isfile(path):
        pytest.skip('needs cemetery1.bst in the .bst corpus')
    with open(path, 'rb') as fh:
        return bst.parse(fh.read())


def _actor(doc, name):
    ab = glvl.top_block(doc, 'actors')
    return next(a for a in ab.children if isinstance(a, glvl.Block) and a.tag == name)


# ---- floors ---------------------------------------------------------------------------------
def test_floor_at_known_points(cemetery1):
    for (x, z), y in KNOWN_FLOORS:
        assert abs(place.floor_at(cemetery1, 42, x, z) - y) < 1e-4, (x, z)
    floor = place.Floor(cemetery1, 42)
    assert floor(77.0, -106.0) == place.floor_at(cemetery1, 42, 77.0, -106.0)
    assert place.floor_at(cemetery1, 42, 5000.0, 5000.0) is None
    assert place.floor_at(cemetery1, 42, 77.0, -106.0, below=-31.0) is None
    with pytest.raises(place.PlaceError):
        floor.need('x', 5000.0, 5000.0)
    assert place.section_at(cemetery1, 77.0, -30.0, -106.0) == 42


# ---- props ----------------------------------------------------------------------------------
def test_place_prop_bounds(smb_corpus, cemetery1, templates):
    smb_dir = os.path.join(smb_corpus, 'models')
    model = 'graveyard\\spike_bottom'
    if placement.smb_meta_dir(smb_dir, model) is None:
        pytest.skip('%s.smb not under %s' % (model, smb_dir))
    level = place.new_level('proptest', 'proptest', templates, template_level='cemetery1',
                            bst=cemetery1)
    lo, hi = place.section_bbox(cemetery1, 42)[:3], place.section_bbox(cemetery1, 42)[3:]
    cx, cy, cz = [(lo[k] + hi[k]) / 2.0 for k in range(3)]
    blk = place.place_prop(level, model, (cx, cy, cz), 90.0, 'spike_a', templates, smb_dir)
    f = glvl.actor_fields(blk)
    assert f['modelInstance'] == 'graveyard\\\\spike_bottom'
    assert f['orient'] == '90, 0, 0' and f['useCollisionParts'] in ('0', '1')
    assert glvl.actor_class_map(level.lvl)['spike_a'] == 'CProp'
    assert 'models\\\\graveyard\\\\spike_bottom.smf' in glvl.dep_set(level.lvl)
    with pytest.raises(place.PlaceError):
        place.place_prop(level, model, (cx + 5000.0, cy, cz), 0.0, 'spike_b', templates, smb_dir)
    with pytest.raises(place.PlaceError):
        place.place_prop(level, model, (hi[0] - 0.05, cy, cz), 0.0, 'spike_c', templates,
                         smb_dir, bst_section=42)
    with pytest.raises(place.PlaceError):
        place.place_prop(level, model, (cx, cy, cz), 0.0, 'spike_a', templates, smb_dir)
    assert place.resolve_yaw(smb_dir, model, 'W') in (0.0, 90.0, 180.0, 270.0)
    assert levelfile.validate(glvl.serialize_lvl(level.lvl)) == []


# ---- actors ----------------------------------------------------------------------------------
def test_place_actor_overrides_roundtrip(templates, tmp_path):
    level = place.new_level('acttest', 'acttest', templates, template_level='abyss')
    place.place_actor(level, 'CPKESource', 'pke_Key', 'pke_x', (1.0, 2.5, 3.0),
                      {'range': '2', 'rolloverDisplayTag': 'pke_x_Name'},
                      library=templates, template_level='cemetery1', created=False)
    place.place_actor(level, 'CTrigger', 'trig_undergroundContainer1', 'trig_x',
                      (4.0, 5.0, 6.0),
                      {'triggerSize': '24, 20, 20', 'oneShot': '1',
                       'actorEnterEvent': 'void trig_x_onActorEnter(@CTrigger,@CActor)'},
                      library=templates, template_level='cemetery1', yaw=45.0)
    with pytest.raises(place.PlaceError):
        place.place_actor(level, 'CTrigger', 'no_such_actor', 'trig_y', (0, 0, 0),
                          library=templates)
    with pytest.raises(place.PlaceError, match='is a CPKESource, not a CTrigger'):
        place.place_actor(level, 'CTrigger', 'pke_Key', 'trig_y', (0, 0, 0), library=templates,
                          template_level='cemetery1')
    with pytest.raises(place.PlaceError):
        place.place_actor(level, 'CTrigger', 'trig_undergroundContainer1', 'trig_x', (0, 0, 0),
                          library=templates)
    place.emit_section_list(level.lvl, [])
    wire.write_level(level, str(tmp_path))
    back = wire.open_level(str(tmp_path), 'acttest', load_set=False)
    f = glvl.actor_fields(_actor(back.lvl, 'pke_x'))
    assert f['pos'] == '1, 2.5, 3' and f['range'] == '2' and f['createStatus'] == '0'
    assert f['rolloverDisplayTag'] == 'pke_x_Name'
    f = glvl.actor_fields(_actor(back.lvl, 'trig_x'))
    assert f['orient'] == '45, 0, 0' and f['triggerSize'] == '24, 20, 20'
    assert f['actorEnterEvent'] == 'void trig_x_onActorEnter(@CTrigger,@CActor)'
    assert f['onEvent'] == '""', 'other events are blanked'
    assert glvl.actor_class_map(back.lvl) == {'pke_x': 'CPKESource', 'trig_x': 'CTrigger'}
    avl = glvl.top_block(back.lvl, 'actor-version-list')
    assert [c.key for c in avl.children] == ['CPKESource', 'CTrigger']
    assert wire.sec_names(back.lvl) == []
    assert levelfile.validate(glvl.serialize_lvl(back.lvl)) == []


def test_place_squad_heights(cemetery1, templates):
    level = place.new_level('squadtest', 'squadtest', templates, template_level='abyss',
                            bst=cemetery1)
    floor = place.Floor(cemetery1, 42)
    hero = (77.0, floor(77.0, -106.0) + place.STAND, -106.0)
    placed = place.place_squad(level, hero, 180.0, floor, library=templates,
                               fields={'successfulScanEvent': 'void h_scan(@CCharacter,@CActor)'})
    assert placed[0] == 'Ghostbuster0' and set(placed[1:]) == set(place.SQUAD_LAYOUT)
    heroes = glvl.top_block(level.lvl, 'list-of-heros')
    assert [(c.key, c.value) for c in heroes.children] == [('hero_0', 'Ghostbuster0')]
    f = glvl.actor_fields(_actor(level.lvl, 'Ghostbuster0'))
    assert f['successfulScanEvent'] == 'void h_scan(@CCharacter,@CActor)'
    assert f['orient'] == '180, 0, 0'
    for name in placed[1:]:
        x, y, z = [float(v) for v in glvl.actor_fields(_actor(level.lvl, name))['pos'].split(',')]
        assert abs(y - floor(x, z)) < 0.3, name
    # yaw 180 faces -z, so Ray (8 right, 4 back) lands 8 west and 4 north of the hero
    x, _y, z = [float(v) for v in glvl.actor_fields(_actor(level.lvl, 'Ray'))['pos'].split(',')]
    assert (x, z) == (69.0, -102.0)
    assert 'data\\\\gb_player.cit' in glvl.dep_set(level.lvl)
    with pytest.raises(place.PlaceError):
        place.place_squad(level, (5000.0, 0.0, 5000.0), 0.0, floor, library=templates)


# ---- lang ------------------------------------------------------------------------------------
def test_build_level_table_covers_externs():
    externs = ['MissionDescription', 'objShrt_a', 'objLong_a', 'checkpointFH_From_Lost_Island',
               'Diag_Egon_GEN_C_241']
    callouts = {'Diag_Egon_GEN_C_241': 'Hmm, strong signal.'}
    own = [('MissionDescription', 'The Test'), '', '//objectives', ('objShrt_a', 'Do it.'),
           ('objLong_a', 'Do it, then leave.')]
    data = lang.build_level_table('t', externs, callouts, own,
                                  global_tags={'checkpointFH_From_Lost_Island'}, header=['// t'])
    tags = lang.table_tags(data)
    assert all(i in tags for i in externs if i != 'checkpointFH_From_Lost_Island')
    text = data.decode('latin1')
    assert text.startswith('\r\n// t\r\n#include "t_diag.txt"\r\n\r\n')
    assert 'objLong_a, "Do it, then leave."\r\n' in text and text.endswith('\r\n')
    with pytest.raises(lang.LangError):
        lang.build_level_table('t', externs, callouts, own)
    with pytest.raises(lang.LangError):
        lang.build_level_table('t', externs, {}, own, global_tags={'checkpointFH_From_Lost_Island'})
    assert lang.check_level_table(['x'], {}, [('objShrt_b', 'b')]) == [
        'x.text is read and this table has no entry', 'objShrt_b has no objLong_ partner']
    code = ('extern CDialogDatabaseEntry Diag_a; // x\nextern CDialogDatabaseEntry objShrt_q;\n'
            'void f() { dbStartSay(Ray, Diag_a, false); say(Egon, Diag_b); }\n')
    assert lang.dialogue_externs(code) == ['Diag_a', 'objShrt_q']
    assert lang.check_dn_dialogue(code) == ['Diag_b is spoken and never declared']
    assert lang.diag_table(['// none']) == b'// none\r\n'


def test_load_callouts_and_extra_tables(game_dir, lvl_corpus):
    from gbtvgr.level.library import Library
    with Library(game_dir).open() as lib:
        callouts = lang.load_callouts(lib)
        assert callouts.get('Diag_Egon_GEN_C_241') == 'Hmm, strong signal.'
        glob_txt = lib.read('world\\en\\global.txt')
    lv = wire.open_level(lvl_corpus, 'abyss', load_set=False)
    lv.lang = {'en': b'\r\n'}
    missing = lang.check_lang_level(lv)
    assert missing and 'en:' in missing[0]
    assert len(lang.check_lang_level(lv, extra_tables=(glob_txt,))) <= len(missing)


# ---- compile_level's new knobs -------------------------------------------------------------------
def test_compile_level_knobs(templates):
    from gbtvgr.level.scene import Document, TriggerNode
    doc = Document('knobs')
    doc.settings.update(template_level='abyss', hero_start=[10.0, 1.0, 20.0], hero_yaw=90.0,
                        hero_events={'successfulScanEvent': 'void h_scan(@CCharacter,@CActor)'},
                        companions={'Ray': (1.0, 2.0, 3.0), 'Egon': (4.0, 5.0, 6.0)})
    t = TriggerNode('trig_k')
    t.pos[:] = (7, 8, 9)
    t.props.update(template='trig_undergroundContainer1', template_level='cemetery1',
                   fields={'filterHeroExclusive': '1', 'playerFocusDistance': '0'})
    doc.add_node(t)
    build = levelfile.compile_level(doc, templates.library)
    assert build.errors == [], build.errors
    parsed = glvl.parse_lvl(build.data)
    f = glvl.actor_fields(_actor(parsed, 'Ghostbuster0'))
    assert f['successfulScanEvent'] == 'void h_scan(@CCharacter,@CActor)'
    assert f['orient'] == '90, 0, 0'
    assert glvl.actor_fields(_actor(parsed, 'Ray'))['pos'] == '1, 2, 3'
    assert glvl.actor_fields(_actor(parsed, 'Egon'))['pos'] == '4, 5, 6'
    assert glvl.actor_fields(_actor(parsed, 'Peter'))['pos'] != '4, 5, 6', 'unlisted: old rule'
    f = glvl.actor_fields(_actor(parsed, 'trig_k'))
    assert f['filterHeroExclusive'] == '1' and f['playerFocusDistance'] == '0'
    assert f['filterBaseName'] == '*', 'cloned from the named cemetery1 trigger'
    sl = glvl.top_block(parsed, 'section-list')
    assert sl is not None and sl.children == []
    kids = [c.tag for c in parsed.children if isinstance(c, glvl.Block)]
    assert kids.index('section-list') == kids.index('selection-sets-for-editor-only') + 1
    doc2 = glvl.parse_lvl(build.data)
    levelfile.emit_section_list(doc2, ['knobsA', 'knobsB'])
    assert wire.sec_names(doc2) == ['knobsA', 'knobsB']


# ---- the register-indirect member rule ---------------------------------------------------
OLD_SHAPE = '''module oldshape;
struct CDialogDatabaseEntry { String text; }
enum EHudMessage { eHudMessage_ObjectivesUpdated }
extern CDialogDatabaseEntry objShrt_a;
extern CDialogDatabaseEntry objLong_a;
void objective(@CDialogDatabaseEntry shortE, @CDialogDatabaseEntry longE)
{
    setCurrentObjective(longE.text);
    displayMessage(eHudMessage_ObjectivesUpdated, shortE.text, -1.0);
}
void main() { objective(objShrt_a, objLong_a); for (; ; ) { idle(); } }
'''
NEW_SHAPE = '''module newshape;
struct CDialogDatabaseEntry { String text; }
enum EHudMessage { eHudMessage_ObjectivesUpdated }
extern CDialogDatabaseEntry objShrt_a;
extern CDialogDatabaseEntry objLong_a;
void main()
{
    setCurrentObjective(objLong_a.text);
    displayMessage(eHudMessage_ObjectivesUpdated, objShrt_a.text, -1.0);
    for (; ; ) { idle(); }
}
'''


def _compile(tmp_path, stem, source, *flags):
    src = tmp_path / (stem + '.dn')
    out = tmp_path / (stem + '.dante')
    src.write_text(source)
    r = subprocess.run([sys.executable, '-m', 'dante', 'compile', str(src), '-o', str(out)]
                       + list(flags), capture_output=True, text=True)
    return r, out


def test_indirect_member_rule(tmp_path, lvl_corpus):
    import dante
    # the compiler refuses the shape outright now; the rule catches a module that got past it
    r, _out = _compile(tmp_path, 'oldshape', OLD_SHAPE)
    assert r.returncode != 0 and 'read through a parameter or local' in r.stdout + r.stderr
    r, out = _compile(tmp_path, 'oldshape', OLD_SHAPE, '--allow-indirect-members')
    assert r.returncode == 0, r.stdout + r.stderr
    old = checks.check_indirect_members(dante.load(str(out)))
    assert len(old) == 1 and 'CDialogDatabaseEntry::text' in old[0] and '@7' in old[0]
    r, out = _compile(tmp_path, 'newshape', NEW_SHAPE)
    assert r.returncode == 0, r.stdout + r.stderr
    assert checks.check_indirect_members(dante.load(str(out))) == []
    lv = wire.open_level(lvl_corpus, 'hotel1a', load_set=False)
    assert checks.indirect_member_check(lv) == [], 'hotel1a reads .tag through a register'
    # the one shipped site that is not .tag: isLastActorSpawnedDead(@CSpawn) in global.dante
    hits = {}
    for path in sorted(glob.glob(os.path.join(lvl_corpus, 'world', '*.dante'))):
        mod = dante.load(path)
        assert checks.check_indirect_members(mod) == [], path
        bad = checks.check_indirect_members(mod, allowed=frozenset(('tag',)))
        if bad:
            hits[os.path.basename(path)] = bad
    assert list(hits) == ['global.dante'] and len(hits['global.dante']) == 1, hits
    assert 'CSpawn::lastActorSpawned' in hits['global.dante'][0]

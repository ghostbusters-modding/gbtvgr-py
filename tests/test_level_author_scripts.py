"""Trigger blocks -> .dn -> .dante through the real compiler."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import os

import pytest

from gbtvgr.level import scripts
from gbtvgr.level.scene import ActorNode, Document, GroupNode, TriggerNode


@pytest.fixture(scope="module")
def library(game_dir):
    from gbtvgr.level.library import Library
    lib = Library(game_dir)
    lib.open()
    yield lib
    lib.close()


def make_doc():
    doc = Document('blocktest')
    doc.settings['hero_start'] = [0.0, 0.0, 0.0]
    doc.settings['description'] = 'A test level.'
    sp = ActorNode('spEmit_a')
    sp.props['cls'] = 'CSpawn'
    sp.pos[:] = (10.0, 0.0, 10.0)
    sp.fields['name'] = 'spEmit_a'
    doc.add_node(sp)
    b = ActorNode('biped_1')
    b.props['cls'] = 'CBiped'
    b.pos[:] = (-40.0, 0.0, -40.0)
    b.fields['createStatus'] = '0'
    b.fields['charInfoFilename'] = 'Biped1\\\\Fiend_web.cit'
    b.fields['charInfoVariantName'] = 'standard'
    doc.add_node(b)
    g = GroupNode('g_wave')
    g.members.append(b.id)
    doc.add_node(g)
    t = TriggerNode('trig_a')
    t.pos[:] = (30.0, 5.0, 30.0)
    t.props['size'] = [10.0, 10.0, 10.0]
    t.props['once'] = True
    t.props['on_enter'] = [
        {'block': 'spawn_character', 'params': {'spawner': 'spEmit_a', 'cls': 'CBiped',
                                                'cit': 'Biped1\\Fiend_web.cit', 'variant': 'standard',
                                                'group': 'g_wave'}},
        {'block': 'display_message', 'params': {'text': 'hi', 'seconds': 4.0}},
    ]
    doc.add_node(t)
    return doc


def test_generate_source_shape():
    src = scripts.generate_source(make_doc())
    assert src.startswith('// Generated')
    assert 'module blocktest1;' in src
    for _field, suffix in scripts.TRIG_EVENTS:
        assert 'void trig_a_%s(@CTrigger' % suffix in src
    assert 'spawnBiped(spEmit_a, "Biped1\\\\Fiend_web.cit", "standard")' in src
    assert 'g_wave.add(' in src
    assert 'extern CActorGroup g_wave;' in src
    assert 'extern CSpawn spEmit_a;' in src
    assert 'extern CGhostbuster Ghostbuster0;' in src
    assert 'setLevelDescription("A test level.");' in src
    assert 'void spEmit_a_onActivate(@CSpawn spawn) { }' in src


def test_handler_names():
    h = scripts.handler_names(make_doc())
    assert h['trig_a']['actorEnterEvent'] == 'void trig_a_onActorEnter(@CTrigger,@CActor)'
    assert len(h['trig_a']) == 10
    assert scripts.spawner_handlers(make_doc()) == {'spEmit_a': 'void spEmit_a_onActivate(@CSpawn)'}


def test_ident_and_literals():
    assert scripts.ident('trig a-1') == 'trig_a_1'
    assert scripts.ident('9x') == '_9x'
    assert scripts.dn_str('a\\b') == '"a\\\\b"'
    assert scripts.dn_str('a\\\\b') == '"a\\\\b"'
    assert scripts.dn_float(2) == '2.0' and scripts.dn_float(-2.5) == '-2.5'


def test_state_triggers_generate():
    doc = make_doc()
    t = TriggerNode('trig_state')
    t.props['state'] = {'block': 'actor_group_dead', 'params': {'group': 'g_wave'}}
    t.props['on_state'] = [{'block': 'display_message', 'params': {'text': 'dead'}}]
    doc.add_node(t)
    t2 = TriggerNode('trig_timer')
    t2.props['once'] = False
    t2.props['state'] = {'block': 'timer', 'params': {'seconds': 3}}
    t2.props['on_state'] = [{'block': 'wait', 'params': {'seconds': 1}},
                            {'block': 'camera_normal', 'params': {}}]
    doc.add_node(t2)
    src = scripts.generate_source(doc)
    assert 'void stateThread()' in src
    assert 'isActorGroupDead(g_wave)' in src
    assert 'trig_timer_clock += *gTimeSlice;' in src
    assert 'beginThread(trig_timer_on_state, false);' in src
    assert 'extern @CGameView gMainView;' in src


def test_compiles(library, tmp_path):
    doc = make_doc()
    out = scripts.compile_document(doc, library, str(tmp_path))
    assert os.path.isfile(out)
    import dante
    d = dante.load(out)
    exports = {p for k, _o, p in d.exports if k == 'C'}
    for _field, suffix in scripts.TRIG_EVENTS:
        assert 'void trig_a_%s(@CTrigger,@CActor)' % suffix in exports
    assert 'void setupLevel()' in exports and 'void main()' in exports
    assert 'void spEmit_a_onActivate(@CSpawn)' in exports
    assert dante.verify_one(d) == [], 'structural verification is clean'


def test_state_triggers_compile(library, tmp_path):
    doc = make_doc()
    t = TriggerNode('trig_state')
    t.props['state'] = {'block': 'actor_group_dead', 'params': {'group': 'g_wave'}}
    t.props['on_state'] = [{'block': 'kill_group', 'params': {'group': 'g_wave'}},
                           {'block': 'set_objective', 'params': {'text': 'long', 'short': 'short'}},
                           {'block': 'define_checkpoint', 'params': {'name': 'Mid', 'text': 'Midway'}},
                           {'block': 'save_checkpoint', 'params': {'name': 'Mid'}},
                           {'block': 'start_effect', 'params': {'effect': 'rubblesmoke.tfa', 'at': 'spEmit_a'}},
                           {'block': 'play_sfx', 'params': {'name': 'sfx/door'}},
                           {'block': 'set_music', 'params': {'cue': ''}},
                           {'block': 'warp_actor', 'params': {'actor': 'biped_1', 'pos': [1, 2, 3]}},
                           {'block': 'enable_trigger', 'params': {'trigger': 'trig_a', 'enabled': False}}]
    doc.add_node(t)
    t2 = TriggerNode('trig_timer')
    t2.props['once'] = False
    t2.props['state'] = {'block': 'timer', 'params': {'seconds': 3}}
    t2.props['on_state'] = [{'block': 'wait', 'params': {'seconds': 1}},
                            {'block': 'camera_orbit', 'params': {'target': 'biped_1'}},
                            {'block': 'disable_actor', 'params': {'actor': 'biped_1'}}]
    doc.add_node(t2)
    t3 = TriggerNode('trig_down')
    t3.props['state'] = {'block': 'players_downed', 'params': {}}
    t3.props['on_state'] = [{'block': 'display_message', 'params': {'text': 'down'}}]
    doc.add_node(t3)
    t4 = TriggerNode('trig_cp')
    t4.props['state'] = {'block': 'checkpoint_loaded', 'params': {'checkpoint': 'Mid'}}
    t4.props['on_state'] = [{'block': 'enable_actor', 'params': {'actor': 'spEmit_a'}}]
    doc.add_node(t4)
    t5 = TriggerNode('trig_death')
    t5.props['state'] = {'block': 'actor_dead', 'params': {'actor': 'biped_1'}}
    t5.props['on_state'] = [{'block': 'display_message', 'params': {'text': 'died'}}]
    doc.add_node(t5)
    out = scripts.compile_document(doc, library, str(tmp_path))
    import dante
    d = dante.load(out)
    exports = {p for k, _o, p in d.exports if k == 'C'}
    assert 'void checkpoint_Mid()' in exports
    assert 'void stateThread()' in exports
    assert 'void trig_death_onDeath(@CCharacter)' in exports


def test_compile_error_is_clear(library, tmp_path):
    doc = make_doc()
    doc.settings['user_dn'] = 'void broken( {'
    with pytest.raises(scripts.ScriptError):
        scripts.compile_document(doc, library, str(tmp_path))

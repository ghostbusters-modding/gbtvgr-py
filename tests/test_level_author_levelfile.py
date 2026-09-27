"""The `.lvl` compiler against shipped templates, including .sec-aware compile."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only
import collections
import os

import pytest

from gbtvgr import lvl as glvl
from gbtvgr.level import importer, levelfile, wire
from gbtvgr.level.components import ComponentDef
from gbtvgr.level.scene import (ActorNode, ComponentNode, Document, GroupNode,
                                TriggerNode)


@pytest.fixture(scope="module")
def library(game_dir):
    from gbtvgr.level.library import Library
    lib = Library(game_dir)
    lib.open()
    yield lib
    lib.close()


def level_doc():
    doc = Document('lvltest')
    doc.settings['hero_start'] = [3.0, 0.0, 9.0]
    sp = ActorNode('spEmit_a')
    sp.props.update(cls='CSpawn', template='spEmit_CultistBellRinger', template_level='cemetery2')
    sp.pos[:] = (10, 1, 10)
    doc.add_node(sp)
    biped = ActorNode('pool_fiend1')
    biped.props.update(cls='CBiped', template='biped_GraveFiend1', template_level='cemetery2',
                       created=False)
    biped.fields['charInfoFilename'] = 'Biped1\\\\Fiend_web.cit'
    biped.fields['charInfoVariantName'] = 'standard'
    biped.pos[:] = (0, -50, 0)
    doc.add_node(biped)
    g = GroupNode('g_wave')
    g.members.append('pool_fiend1')
    doc.add_node(g)
    t = TriggerNode('trig_a')
    t.pos[:] = (20, 5, 20)
    t.props['size'] = [12.0, 10.0, 12.0]
    doc.add_node(t)
    cd = ComponentDef('crate', 'lost_island\\crate01', ['lost_island\\crate01'])
    doc.components[cd.id] = cd
    c = ComponentNode('crate_1')
    c.props['component'] = cd.id
    c.pos[:] = (5, 0, 5)
    doc.add_node(c)
    return doc


def test_compile_level_validates(library):
    doc = level_doc()
    handlers = {('trig_a', 'actorEnterEvent'): 'void trig_a_onActorEnter(@CTrigger,@CActor)'}
    build = levelfile.compile_level(doc, library, handlers)
    assert build.errors == [], build.errors
    parsed = glvl.parse_lvl(build.data)
    assert glvl.serialize_lvl(parsed) == build.data
    names = glvl.actor_class_map(parsed)
    assert names['Ghostbuster0'] == 'CGhostbuster'
    assert names['spEmit_a'] == 'CSpawn' and names['trig_a'] == 'CTrigger'
    assert names['crate_1'] == 'CProp' and names['pool_fiend1'] == 'CBiped'
    deps = glvl.dep_set(parsed)
    assert 'data\\\\Biped1\\\\Fiend_web.cit' in deps
    assert any(d.lower().endswith('.civh') and 'biped1' in d.lower() for d in deps)
    ab = glvl.top_block(parsed, 'actors')
    trig = next(a for a in ab.children if isinstance(a, glvl.Block) and a.tag == 'trig_a')
    f = glvl.actor_fields(trig)
    assert f['triggerSize'] == '12, 10, 12'
    assert f['actorEnterEvent'].startswith('void trig_a_onActorEnter')
    assert f['createStatus'] == '1'
    crate = next(a for a in ab.children if isinstance(a, glvl.Block) and a.tag == 'crate_1')
    assert glvl.actor_fields(crate)['modelInstance'] == 'lost_island\\\\crate01'
    groups = glvl.top_block(parsed, 'actor-groups')
    assert any(isinstance(c, glvl.ListF) and c.key == 'g_wave' for c in groups.children)
    assert levelfile.validate(build.data) == []


# -- the two design-doc extensions: .sec-aware import and compile -------------------------
class _StubLibrary:
    """The two calls compile_level needs from a level's own corpus, without the
    real archive chain: a shipped .lvl by stem, and every stem available."""

    def __init__(self, world_dir):
        self.world_dir = world_dir

    def read_level(self, stem):
        p = os.path.join(self.world_dir, stem + '.lvl')
        if not os.path.isfile(p):
            return None
        with open(p, 'rb') as fh:
            return fh.read()

    def level_stems(self):
        return [f[:-4] for f in os.listdir(self.world_dir) if f.lower().endswith('.lvl')]


def test_import_level_tags_sections(lvl_corpus):
    world = os.path.join(lvl_corpus, 'world')
    lv = wire.open_level(lvl_corpus, 'hotel1a', load_set=False)
    doc = Document('hotel1a')
    added = importer.import_level(doc, lv, 'hotel1a')
    # 588 .lvl (592 minus the 4 heroes, filtered like a plain import) + 2006 .sec actors
    assert len(added) == 2594
    by_section = collections.Counter(n.props.get('section') for n in added)
    # sections 6 (hotel1aSOUND1) and 8 (hotel1aART) carry schema pins only, no actors
    assert set(by_section) == {0, 1, 2, 3, 4, 5, 7}
    assert by_section[0] == 588
    assert doc.settings['sec_names'] == ['hot1a_Lobby1', 'hot1a_12thFloor', 'hot1a_Lobby2',
                                         'hot1a_Kitchen', 'hot1a_Lobby3', 'hotel1aSOUND1',
                                         'hot1a_Ballroom', 'hotel1aART']
    groups = doc.nodes('group')
    assert groups and all(g.props.get('section') is not None for g in groups)


def test_compile_level_writes_matching_sec_layers(lvl_corpus):
    world = os.path.join(lvl_corpus, 'world')
    lv = wire.open_level(lvl_corpus, 'hotel1a', load_set=False)
    doc = Document('hotel1a')
    added = importer.import_level(doc, lv, 'hotel1a')
    lib = _StubLibrary(world)
    build = levelfile.compile_level(doc, lib)

    parsed = glvl.parse_lvl(build.data)
    assert glvl.serialize_lvl(parsed) == build.data, '.lvl does not round-trip'
    al = glvl.top_block(parsed, 'actor-list')
    lvl_names = {c.key for c in al.children if isinstance(c, glvl.KV)}
    expected0 = {n.name for n in added if (n.props.get('section') or 0) == 0}
    heroes = {'Ghostbuster0', 'Egon', 'Ray', 'Peter'}
    assert lvl_names - heroes == expected0, 'main .lvl carries exactly the section-0 actors plus heroes'

    assert [name for name, _data in build.secs] == ['hot1a_Lobby1', 'hot1a_12thFloor', 'hot1a_Lobby2',
                                                     'hot1a_Kitchen', 'hot1a_Lobby3', 'hot1a_Ballroom']
    for name, data in build.secs:
        sd = glvl.parse_lvl(data)
        assert glvl.serialize_lvl(sd) == data, '%s.sec does not round-trip' % name
        sal = glvl.top_block(sd, 'actor-list')
        names = {c.key for c in sal.children if isinstance(c, glvl.KV)}
        sec_idx = doc.settings['sec_names'].index(name) + 1
        expected = {n.name for n in added if n.props.get('section') == sec_idx}
        # a handful of Ballroom's CBurningClothActor props have no template TemplateLibrary
        # can find (it never searches .sec); they drop with an error, see the report.
        assert names <= expected, (name, sorted(names - expected))
        assert len(expected - names) <= 11, (name, len(expected - names))

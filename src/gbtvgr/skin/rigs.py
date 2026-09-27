"""Bone vocabularies. The common human set is the Ghostbusters' own bone names, which the
cast, Janine and every human NPC share; sources map into it, targets read it directly."""
# Copyright (C) 2026 Colin Sullivan and contributors
# SPDX-License-Identifier: GPL-2.0-only

FINGERS = ('index', 'middle', 'ring', 'pinkie', 'thumb')

HUMAN = ['bone_hips', 'bone_spine1', 'bone_spine2', 'bone_spine3', 'bone_neck', 'bone_head', 'bone_head_top']
for _s in 'LR':
    HUMAN += ['bone_%s_%s' % (_s, b) for b in ('clavicle1', 'upperarm', 'forearm', 'hand',
                                               'thigh', 'calf', 'ankle', 'foot', 'toe')]
    HUMAN += ['bone_%s_%s%d' % (_s, f, k) for f in FINGERS for k in range(1, 5)]

# the child a bone is aimed along when it has several; None = never rotate
AIM = {'bone_hips': None, 'bone_spine3': 'bone_neck',
       'bone_L_hand': 'bone_L_middle1', 'bone_R_hand': 'bone_R_middle1'}
# fitting lengths here squashes helmets and hoods: move and turn only
RIGID = {'bone_neck', 'bone_head'}


def _mixamo():
    m = {'Hips': 'bone_hips', 'Spine': 'bone_spine1', 'Spine1': 'bone_spine2', 'Spine2': 'bone_spine3',
         'Neck': 'bone_neck', 'Head': 'bone_head', 'HeadTop_End': 'bone_head_top'}
    for S, s in (('Left', 'L'), ('Right', 'R')):
        m.update({S + 'Shoulder': 'bone_%s_clavicle1' % s, S + 'Arm': 'bone_%s_upperarm' % s,
                  S + 'ForeArm': 'bone_%s_forearm' % s, S + 'Hand': 'bone_%s_hand' % s,
                  S + 'UpLeg': 'bone_%s_thigh' % s, S + 'Leg': 'bone_%s_calf' % s,
                  S + 'Foot': 'bone_%s_ankle' % s, S + 'ToeBase': 'bone_%s_foot' % s,
                  S + 'Toe_End': 'bone_%s_toe' % s})
        for F, f in (('Index', 'index'), ('Middle', 'middle'), ('Ring', 'ring'), ('Pinky', 'pinkie'),
                     ('Thumb', 'thumb')):
            for k in range(1, 5):
                m['%sHand%s%d' % (S, F, k)] = 'bone_%s_%s%d' % (s, f, k)
    return m


SOURCE_RIGS = {'mixamo': _mixamo(), 'human': {b: b for b in HUMAN}}


def _gb_parts():
    p = {'bone_hips': 'part_torso_bottom', 'bone_spine1': 'part_waist', 'bone_spine2': 'part_torso_top',
         'bone_spine3': 'part_torso_top', 'bone_neck': 'part_head', 'bone_head': 'part_head'}
    for s in 'LR':
        p.update({'bone_%s_clavicle1' % s: 'part_torso_top', 'bone_%s_upperarm' % s: 'part_arm_%s' % s,
                  'bone_%s_forearm' % s: 'part_forearm_%s' % s, 'bone_%s_hand' % s: 'part_forearm_%s' % s,
                  'bone_%s_thigh' % s: 'part_thigh_%s' % s, 'bone_%s_calf' % s: 'part_shin_%s' % s,
                  'bone_%s_ankle' % s: 'part_foot_%s' % s, 'bone_%s_foot' % s: 'part_foot_%s' % s})
    return p


_GB = {'dir': 'ghostbuster', 'parts': _gb_parts(),
       'keep': ('acc_',), 'drop': ('acc_collar_', 'acc_tucked_', 'acc_untucked_', 'acc_goggles_rookie')}

# character -> the shipped meshes it is drawn with, and how to fill them
TARGETS = {
    'rookie': dict(_GB, files=['gb_player', 'gb_player_nopack']),
    'peter': dict(_GB, files=['venkman', 'venkman_nopack']),
    'ray': dict(_GB, files=['stantz', 'stantz_nopack']),
    'egon': dict(_GB, files=['spengler', 'spengler_nopack']),
    'winston': dict(_GB, files=['zeddemore', 'zeddemore_nopack']),
    'janine': {'dir': 'npc', 'files': ['janine', 'janine_sweater'], 'parts': None, 'keep': (), 'drop': ()},
    # the museum's walking enemy: human bone names, 8.75 ft; armour, slime and ice variants dropped
    'mannequin': {'dir': 'biped1', 'files': ['possessed_mannequin'], 'parts': None, 'keep': (), 'drop': ()},
}

import csv


import os


import uuid


import torch


RELATION_TB_GROUPS = (
    ('00_main', (
        'placement/episode_final_rate', 'placement/episode_ever_rate',
        'placement/post_first_retention', 'current_success_state', 'holding/satisfied',
    )),
    ('01_placement', (
        'goal_xy_error', 'goal_z_error', 'at/satisfied',
        'placement/first_seconds', 'placement/longest_hold_seconds',
        'placement/post_first_seconds',
    )),
    ('02_reward', (
        'saturation_active', 'current_success_reward', 'task_relation_total',
        'holding/state_reward', 'at/state_reward',
        'holding/progress_reward', 'at/progress_reward',
        'holding/raw_state_reward', 'at/raw_state_reward',
        'holding/raw_progress_reward', 'at/raw_progress_reward',
        'success_bonus', 'first_success_bonus',
    )),
    ('03_samples', (
        'placement/eligible_completed_count', 'placement/reached_count',
        'placement/initially_placed_count', 'placement/initially_placed_fraction',
        'placement/completed_count', 'placement/initially_placed_final_rate',
    )),
    ('04_state', (
        'holding/phi', 'at/phi', 'holding/gate', 'at/gate',
        'holding/achieved', 'at/achieved', 'current_target_valid',
        'scene_current_all_valid', 'done', 'scene_all_done', 'first_success',
        'holding/threshold_up', 'holding/threshold_down',
        'at/threshold_up', 'at/threshold_down', 'active',
    )),
    ('05_motion', (
        'root_box_distance_xy', 'hand_midpoint_distance',
        'right_hand_center_distance', 'left_hand_center_distance',
        'box_speed', 'box_bottom_height_proxy', 'target_xy_crossing',
        'progress_holding', 'progress_at',
        'holding/progress_bar', 'at/progress_bar', 'holding/approach', 'at/approach',
    )),
)


_RELATION_TB_TAGS = {
    key: 'relation/{}/{:02d}_{}'.format(group, index, key.replace('/', '_'))
    for group, keys in RELATION_TB_GROUPS
    for index, key in enumerate(keys, 1)
}


def relation_tensorboard_tag(key):
    """Presentation only; preserve unlisted metrics in a trailing debug group."""
    if key.startswith(('edge/', 'agent/', 'goal/', 'penalty/', 'sampling/', 'sharing/',
                       'ontop/', 'sit/', 'climb/')):
        return 'relation/' + key
    if key.startswith('scenario/'):
        return 'relation/06_scenarios/' + key[len('scenario/'):]
    return _RELATION_TB_TAGS.get(key, 'relation/90_debug/' + key)

"""Straight goals require translation along the initial box face normal only."""
from isaacgym import gymapi  # before torch
from copy import deepcopy
from pathlib import Path
import json
import math
import pytest
import torch
import yaml
from utils.door_reference import rotate
from utils.open_push_scene import goal_travel_limit, sample_goal_offsets, sample_boxes
from utils.open_push_spec import validate_env
from utils.open_push_rsi import bank_contract

NAME = 'push_fixed130_point_zband_heightmatch_straight_goal2to5'


def test_goal_is_initial_face_normal_with_uniform_distance_and_unchanged_box():
    spec = dict(sampling='box_forward_uniform', distance_range=[2., 5.])
    boxes = torch.zeros(10000, 2, 13)
    yaw = torch.linspace(-math.pi, math.pi, 20000).reshape(10000, 2)
    boxes[..., 5] = (yaw/2).sin(); boxes[..., 6] = (yaw/2).cos()
    boxes[..., :3] = torch.randn(10000, 2, 3)
    before = boxes.clone()
    torch.manual_seed(42); offsets = sample_goal_offsets(boxes, spec)
    torch.manual_seed(42); torch.testing.assert_close(sample_goal_offsets(boxes, spec), offsets)
    torch.testing.assert_close(boxes, before, rtol=0, atol=0)
    distance = offsets.norm(dim=-1)
    assert distance.min() >= 2. and distance.max() <= 5.
    assert abs(distance.mean().item()-3.5) < .025
    local_forward = rotate(boxes[..., 3:7], boxes.new_tensor([0., 1., 0.]).expand(10000, 2, 3))
    torch.testing.assert_close(offsets/distance[..., None], local_forward, atol=1e-6, rtol=0)
    assert offsets[..., 2].count_nonzero() == 0


def test_config_only_changes_goal_law_and_bank_and_keeps_screened_poses():
    root = Path('tokenhsi')
    source_name = NAME.replace('_straight', '')
    source, target = [yaml.safe_load((root/'data/cfg/multi_agent'/f'{n}.yaml').read_text())['env']
                      for n in (source_name, NAME)]
    allowed = deepcopy(source)
    allowed['pushGoalRandomization']['sampling'] = 'box_forward_uniform'
    allowed['openPushRSI']['bank_file'] = target['openPushRSI']['bank_file']
    assert allowed == target
    for env in (source, target):
        env['motion_file'] = 'tokenhsi/data/dataset_push_only_push180.yaml'
        validate_env(env)
    old, new = [json.loads(Path(env['openPushRSI']['bank_file']).read_text()) for env in (source, target)]
    assert old['candidates'] == new['candidates']
    assert new['contract'] == bank_contract(target)
    boxes, forward = sample_boxes(2048, 'cpu', target['pushSceneRandomization'], target['pushBox']['size'])
    offsets = sample_goal_offsets(boxes, target['pushGoalRandomization'])
    torch.testing.assert_close(offsets/offsets.norm(dim=-1, keepdim=True), forward, atol=1e-6, rtol=0)


@pytest.mark.parametrize('change', [{'distance_range':[0.,5.]}, {'distance_range':[5.,2.]},
    {'distance_range':[2.,float('nan')]}, {'distance_range':[True,5.]}, {'extra':1}])
def test_invalid_straight_goal_spec_rejected(change):
    spec = dict(sampling='box_forward_uniform', distance_range=[2.,5.])
    spec.update(change)
    with pytest.raises(ValueError): goal_travel_limit(spec)

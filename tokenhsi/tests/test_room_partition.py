from pathlib import Path

import isaacgym
import pytest
import torch
import yaml

from env.tasks.multi_agent.room_geometry import (
    room_walls, sample_partition_layout, compute_wall_cpa_collision_penalty)
from env.tasks.multi_agent.humanoid_ma_room import HumanoidMARoom
from utils.edge_scenario_spec import compose_graph


def test_partition_faces_and_passages_follow_physical_box():
    half = torch.tensor([[3., 3.], [4., 4.]])
    centers, sizes, starts, ends = room_walls(half, .4, 2.6, 1.8)
    assert centers.shape == (2, 5, 3) and starts.shape == (2, 8, 2)
    torch.testing.assert_close(half[:, 1] - sizes[:, 4, 1], torch.full((2,), 1.8))
    torch.testing.assert_close(starts[:, 4:].abs(), sizes[:, 4:5, :2].expand(-1, 4, -1))
    torch.testing.assert_close(ends[:, 4:].abs(), sizes[:, 4:5, :2].expand(-1, 4, -1))


def test_partition_risk_from_both_sides_and_safe_passage():
    _, _, starts, ends = room_walls(torch.tensor([[3., 3.]]), .4, 2.6, 1.8)
    position = torch.tensor([[[-1.2, 0., 1.], [1.2, 0., 1.], [-1.2, 2.2, 1.]]])
    velocity = torch.tensor([[[1., 0., 0.], [-1., 0., 0.], [1., 0., 0.]]])
    risk = compute_wall_cpa_collision_penalty(position, velocity, starts[:, 4:], ends[:, 4:], .7, 1., .9)
    torch.testing.assert_close(risk, torch.tensor([[.9, .9, 0.]]))


@pytest.mark.parametrize('tight', [False, True])
def test_partition_layout_requires_detour_and_keeps_entities_clear(tight):
    torch.manual_seed(29)
    n = 2048
    half = 3 + torch.randint(0, 5, (n, 2)).float() * .25
    boxes = .25 + torch.rand(n, 3, 3) * .4
    if tight:
        half.fill_(3.)
        boxes.fill_(.65)
    assigned = torch.rand(n, 3).argsort(-1)[:, :2]
    human, objects, goals = sample_partition_layout(half, boxes, assigned, .8, .2, .4, 1.8)
    own = objects.gather(1, assigned[..., None].expand(-1, -1, 2))
    radius = boxes[..., :2].norm(dim=-1) / 2
    assert (human[..., 0] * own[..., 0] > 0).all()
    assert (goals[..., 0] * own[..., 0] < 0).all()
    assert (objects.abs() + radius[..., None] + .2 <= half[:, None] + 1e-6).all()
    assert (objects[..., 0].abs() - radius >= .4 - 1e-6).all()
    assert (human.abs() + .8 <= half[:, None] + 1e-6).all()
    t = -own[..., 0] / (goals[..., 0] - own[..., 0])
    y_intersection = own[..., 1] + t * (goals[..., 1] - own[..., 1])
    assert (y_intersection.abs() < half[:, 1:] - 1.8).all()
    assert (own[..., 1].abs() + .2 <= half[:, 1:] - 1.8 + 1e-6).all()
    combined = torch.cat((objects, human, goals), 1)
    radii = torch.cat((radius, torch.full((n, 2), .4), radius.gather(1, assigned)), 1)
    distances = (combined[:, :, None] - combined[:, None]).norm(dim=-1)
    ignored = torch.eye(7, dtype=torch.bool)[None]
    ignored[:, 3:5, 5:] = True
    ignored[:, 5:, 3:5] = True
    assert (distances.masked_fill(ignored, torch.inf)
            >= radii[:, :, None] + radii[:, None] + .2 - 1e-6).all()


def test_goal_reset_uses_graph_slot_and_source_box_height():
    task = HumanoidMARoom.__new__(HumanoidMARoom)
    task._supports_holding_at = True
    task._enable_markers = False
    task._cliff = {'height': .1}
    task._passage_width = 1.8
    task._stage_half_size = torch.tensor([[3., 4.]])
    task._env_origins = torch.tensor([[20., 30., 0.]])
    task._tar_pos = torch.zeros(1, 2, 3)
    task._room_goal_xy = torch.tensor([[[1.5, .2], [-1.5, -.2]]])
    task._box_size = torch.tensor([[[.4, .4, .3], [.4, .4, .5], [.4, .4, .4]]])
    graph = compose_graph([('HOLDING_AT', 'HOLDING_AT')], [((2, 1), (0, 0))])
    task.relation_runtime = type('Runtime', (), {'graph': graph})()
    task._expand_slots = lambda ids: (torch.tensor([0, 0]), torch.tensor([0, 1]))
    task._scenario_physical_object = lambda env, agent: torch.tensor([1, 2])
    task._reset_task(torch.tensor([0]))
    torch.testing.assert_close(task._tar_pos, torch.tensor([[[18.5, 29.8, .3], [21.5, 30.2, .35]]]))


def test_partition_config_has_four_equal_tasks_and_shared_reward():
    root = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent'
    cfg = yaml.safe_load((root / 'approach_stage2_fan_room_partition.yaml').read_text())['env']
    old = yaml.safe_load((root / 'approach_stage2_fan_room.yaml').read_text())['env']
    assert cfg['relationGraph']['template_probabilities'] == dict(HOLDING=.25, SIT=.25, CLIMB=.25, HOLDING_AT=.25, HOLDING_ON_TOP=0.)
    assert cfg['relationReward'] == old['relationReward']


@pytest.mark.parametrize('tight', [False, True])
def test_pickup_detours_preserve_selected_sides_without_initial_collisions(tight):
    torch.manual_seed(71)
    n = 2048
    half = torch.full((n, 2), 3.) if tight else 3 + torch.randint(0, 5, (n, 2)) * .25
    boxes = torch.full((n, 3, 3), .65) if tight else .25 + torch.rand(n, 3, 3) * .4
    assigned = torch.rand(n, 3).argsort(-1)[:, :2]
    cross = torch.tensor([[False, False], [False, True], [True, False], [True, True]]).repeat(n // 4, 1)
    human, objects, goals = sample_partition_layout(half, boxes, assigned, .8, .2, .4, 1.8, cross)
    own = objects.gather(1, assigned[..., None].expand(-1, -1, 2))
    assert torch.equal(human[..., 0] * own[..., 0] < 0, cross)
    t = -human[..., 0] / (own[..., 0] - human[..., 0])
    y = human[..., 1] + t * (own[..., 1] - human[..., 1])
    assert (y.abs()[cross] < (half[:, 1:] - 1.8).expand(-1, 2)[cross]).all()
    assert (goals[..., 0] * own[..., 0] < 0).all()
    t = -own[..., 0] / (goals[..., 0] - own[..., 0])
    y = own[..., 1] + t * (goals[..., 1] - own[..., 1])
    assert (y.abs() < half[:, 1:] - 1.8).all()
    radius = boxes[..., :2].norm(dim=-1) / 2
    assert (objects.abs() + radius[..., None] + .2 <= half[:, None] + 1e-6).all()
    assert (human.abs() + .8 <= half[:, None] + 1e-6).all()
    physical = torch.cat((objects, human), 1)
    radii = torch.cat((radius, torch.full((n, 2), .4)), 1)
    separation = (physical[:, :, None] - physical[:, None]).norm(dim=-1)
    separation.masked_fill_(torch.eye(5, dtype=torch.bool)[None], torch.inf)
    assert (separation >= radii[:, :, None] + radii[:, None] + .2 - 1e-6).all()

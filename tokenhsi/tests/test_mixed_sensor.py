from pathlib import Path
from types import SimpleNamespace

import isaacgym
import torch
import yaml

from env.tasks.multi_agent.humanoid_ma import HumanoidMA
from env.tasks.multi_agent.humanoid_ma_mixed_sensor import HumanoidMAMixedSensor
from env.tasks.multi_agent.humanoid_ma_cliff import HumanoidMACliff
from env.tasks.multi_agent.cliff_geometry import sensor_grid
from env.tasks.multi_agent.mixed_sensor_geometry import sample_stage_goals, load_foot_vertices, lowest_foot_height
from env.tasks.multi_agent.room_geometry import room_walls
from utils.edge_scenario_spec import compose_graph, sample_graph, classify_templates


def test_stage_goals_fit_smallest_stage_and_do_not_overlap():
    torch.manual_seed(7)
    half = torch.full((2048, 2), 2.)
    size = torch.full((2048, 2, 3), .65)
    goals = sample_stage_goals(half, size, .8)
    assert (goals.abs() <= 1.2 + 1e-6).all()
    assert ((goals[:, 0] - goals[:, 1]).norm(dim=-1) >= .65 * 2 ** .5 + .2).all()


def test_configuration_samples_all_four_tasks_in_all_three_scenarios():
    cfg = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/approach_stage2_fan_mixed.yaml').read_text())['env']
    assert list(cfg['scenarioMixture'].values()) == [1 / 3] * 3
    assert list(cfg['relationGraph']['template_probabilities'].values()) == [.25] * 4 + [0.]
    assert cfg['cliff']['sensor_resolution'] == 32 and cfg['cliff']['sensor_occlusion'] == 'none'
    assert cfg['stateInit'] == 'Random'
    assert all(row == [1.] + [0.] * 7 for row in cfg['templateRsi'].values())
    torch.manual_seed(31)
    graph = sample_graph(3000, cfg['relationGraph'])
    ids = torch.arange(3000).repeat_interleave(2)
    agents = torch.arange(2).repeat(3000)
    kinds = classify_templates(graph, ids, agents, True)
    counts = torch.bincount(kinds, minlength=4)
    assert ((counts.float() / 6000 - .25).abs() < .03).all()


def test_mixture_resamples_partial_resets_in_train_and_eval(monkeypatch):
    task = HumanoidMAMixedSensor.__new__(HumanoidMAMixedSensor)
    task._scenario_ids = torch.full((12000,), -1)
    task._floor_height = torch.zeros(12000)
    task._scenario_probabilities = torch.tensor([1 / 3] * 3)
    task._scene_origins = torch.zeros(12000, 3, 3)
    task._scene_origins[:, :, 1] = torch.tensor([-32., 0., 32.])
    task._env_origins = torch.zeros(12000, 3)
    task._room, task._cliff = {'floor_height': .1}, {'height': 1.5}
    monkeypatch.setattr(HumanoidMACliff, '_reset_envs', lambda self, ids: None)
    ids = torch.arange(0, 12000, 2)
    for task._is_eval in (False, True):
        task._reset_envs(ids)
        counts = torch.bincount(task._scenario_ids[ids], minlength=3).float() / len(ids)
        assert ((counts - 1 / 3).abs() < .03).all()
        assert (task._scenario_ids[1::2] == -1).all()
        torch.testing.assert_close(task._env_origins[ids], task._scene_origins[ids, task._scenario_ids[ids]])
        assert (task._env_origins[1::2] == 0).all()
        torch.testing.assert_close(task._floor_height[ids], torch.where(task._scenario_ids[ids] == 2, .1, 1.5))
    assert torch.equal(task._sample_bridge_active(ids), task._scenario_ids[ids] == 1)


def test_rsi_foot_correction_preserves_pose_velocities_and_other_envs():
    task = HumanoidMAMixedSensor.__new__(HumanoidMAMixedSensor)
    task.device = 'cpu'
    task.num_bodies = 2
    task._cliff = {'height': 1.5}
    task._floor_height = torch.tensor([1.5, .1, 1.5])
    task._env_origins = torch.tensor([[0., -32., 0.], [96., 32., 0.], [192., 0., 0.]])
    task.physics_metadata = {'rsi_foot_clearance': .005}
    task._foot_body_ids = [0, 1]
    task._foot_vertices = [torch.tensor([[0., 0., -.08], [.2, 0., -.06]])] * 2
    root = torch.zeros(3, 2, 13); root[..., 2] = .94; root[..., 6] = 1.; root[..., 7:13] = .2
    body = root[:, :, None].expand(-1, -1, 2, -1).clone()
    body[..., 2] = .05
    task._humanoid_root_states = root.clone()
    task._kinematic_humanoid_rigid_body_states = body.clone()
    ids = torch.tensor([0, 1])
    task._place_humans(ids, task._env_origins[ids, None, :2].expand(-1, 2, -1), torch.zeros(2, 2))
    foot = lowest_foot_height(task._kinematic_humanoid_rigid_body_states[ids], task._foot_body_ids, task._foot_vertices)
    torch.testing.assert_close(foot, task._floor_height[ids, None].expand(-1, 2) + .005)
    torch.testing.assert_close(task._humanoid_root_states[ids, :, 7:13], root[ids, :, 7:13])
    torch.testing.assert_close(task._humanoid_root_states[2], root[2])
    torch.testing.assert_close(task._kinematic_humanoid_rigid_body_states[2], body[2])


def test_foot_geometry_uses_rotated_mesh_vertices():
    cfg = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/approach_stage2_fan_mixed.yaml').read_text())['env']
    vertices = load_foot_vertices(cfg['asset'], 'cpu')
    body = torch.zeros(1, 2, 2, 13)
    body[..., 2] = 1.; body[..., 6] = 1.
    height = lowest_foot_height(body, [0, 1], vertices)
    expected = 1. + min(float(v[:, 2].min()) for v in vertices)
    torch.testing.assert_close(height, torch.full((1, 2), expected))
    body[..., 4] = 2 ** -.5; body[..., 6] = 2 ** -.5
    expected = 1. - max(float(v[:, 0].max()) for v in vertices)
    torch.testing.assert_close(lowest_foot_height(body, [0, 1], vertices), torch.full((1, 2), expected))


def test_goal_reset_respects_shuffled_bindings_and_floor_height():
    task = HumanoidMAMixedSensor.__new__(HumanoidMAMixedSensor)
    task.device = 'cpu'
    task.num_agents = 2
    task._enable_markers = False
    task._env_origins = torch.tensor([[10., 20., 0.], [50., 60., 0.], [90., 100., 0.]])
    task._scenario_ids = torch.arange(3)
    task._floor_height = torch.tensor([1.5, 1.5, .1])
    task._room = {'partition': {'passage_width': 1.8}}
    task._room_half_size = torch.full((3, 2), 3.)
    task._goal_xy = torch.tensor([[[1., .3], [-1., -.3]]]).expand(3, -1, -1)
    task._tar_pos = torch.zeros(3, 2, 3)
    task._box_size = torch.tensor([[[.4, .4, .3], [.4, .4, .5], [.4, .4, .4]]]).expand(3, -1, -1)
    task.relation_runtime = SimpleNamespace(graph=compose_graph([('HOLDING_AT', 'HOLDING_AT')] * 3, [((2, 1), (0, 0))] * 3))
    task._scenario_physical_object = lambda env, agent: torch.where(agent == 0, 1, 2)
    task._reset_task(torch.arange(3))
    torch.testing.assert_close(task._tar_pos[..., :2] - task._env_origins[:, None, :2], task._goal_xy.flip(1))
    torch.testing.assert_close(task._tar_pos[..., 2], task._floor_height[:, None] + torch.tensor([[.2, .25]]))


def test_room_wall_risk_is_disabled_in_cliff_and_bridge():
    task = HumanoidMAMixedSensor.__new__(HumanoidMAMixedSensor)
    task._scenario_ids = torch.arange(3)
    task._agent_collision_ttc_discount = .99
    task._wall_collision_ttc_discount = .95
    task.dt = 1 / 30
    task._humanoid_root_states = torch.zeros(3, 2, 13)
    roots = torch.tensor([[[2.8, 0., 2.4], [2.8, 2., 2.4]]]).expand(3, -1, -1)
    task._humanoid_root_states[..., 7] = 1
    _, _, task._wall_starts, task._wall_ends = room_walls(torch.full((3, 2), 3.), .4, 2.6, 1.8)
    risk = task._cpa_collision_risk(roots, .7)
    assert (risk[:2] == 0).all() and (risk[2] > 0).all()


def test_partial_observation_uses_each_scenario_floor(monkeypatch):
    task = HumanoidMAMixedSensor.__new__(HumanoidMAMixedSensor)
    task._floor_height = torch.tensor([1.5, 1.5, .1])
    monkeypatch.setattr(HumanoidMA, '_compute_clean_humanoid_nodes', lambda self, p, *args: p)
    ids = torch.tensor([2, 0])
    origins = torch.tensor([[20., 30., 0.], [10., 40., 2.]])
    body = origins[:, None, None].expand(-1, 2, 3, -1).clone()
    body[..., 2] += task._floor_height[ids, None, None] + 1
    result = task._compute_clean_humanoid_nodes(body, None, None, None, origins, ids)
    torch.testing.assert_close(result[..., 2], torch.ones(2, 2, 3))


def test_square_sensor_is_centered_on_root_and_rotates_with_heading():
    cliff = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/approach_stage2_square_mixed.yaml').read_text())['env']['cliff']
    n, r = cliff['sensor_resolution'], cliff['sensor_range']
    pos = torch.zeros(1, 2, 15, 3)
    pos[0, :, 0] = torch.tensor([[-3., -2., .9], [0., 0., .9]])
    pos[0, :, 1:, :2] += 5  # other bodies, including the head, must not move the sensor
    yaw = torch.tensor([torch.pi / 2, 0.])
    rot = torch.zeros(1, 2, 15, 4)
    rot[..., 3] = 1
    rot[0, :, 0, 2], rot[0, :, 0, 3] = (yaw / 2).sin(), (yaw / 2).cos()
    behind = (torch.tensor([[[-5., 0., .5]]]), torch.tensor([[[0., 0., 0., 1.]]]), torch.full((1, 1, 3), .5))
    task = SimpleNamespace(_cliff=cliff, _square_sensor=True, num_agents=2, _env_origins=torch.zeros(1, 3),
                           _rigid_body_pos=pos, _rigid_body_rot=rot, _sensor_boxes=lambda ids: behind,
                           _sensor_grid=sensor_grid(n, r, 'cpu'))
    data = HumanoidMACliff._fan_sensor_data(task)
    assert data['observation'].shape == (1, 2, 2304)
    xy = data['xy'][0].view(2, n, n, 2)
    torch.testing.assert_close(xy.mean((1, 2)), torch.tensor([[-3., -2.], [0., 0.]]))
    # Local +x is the root heading: the front edge is r ahead, the back edge r behind.
    torch.testing.assert_close(xy[:, -1].mean(1), torch.tensor([[-3., -2. + r], [r, 0.]]))
    torch.testing.assert_close(xy[:, 0].mean(1), torch.tensor([[-3., -2. - r], [-r, 0.]]))
    torch.testing.assert_close((xy - xy.mean((1, 2), keepdim=True)).abs().amax((1, 2, 3)), torch.full((2,), r))
    # Both agents see the box top 1 m high behind agent 1, at 0.1 m above the 0.9 m root.
    top = data['observation'][0].amax(-1)
    torch.testing.assert_close(top, torch.full((2,), .1 / 3))
    hits = data['xy'][0, 1][data['observation'][0, 1] > 0]
    assert len(hits) and ((hits - torch.tensor([-5., 0.])).abs() <= .5).all()

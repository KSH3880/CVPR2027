"""Midpoint-only CONTACT with the imported AT and current-placement saturation."""
from isaacgym import gymapi
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import pytest
import torch
import yaml
from utils.push_point import local_points, world_points, midpoint_reward_phi
from utils.stage1_push import validate_env, check_checkpoint, sample_box_sizes, box_size_profiles, sample_boxes_and_goals, hand_box_overlap
from utils.edge_scenario_spec import compose_canonical_graph
from utils.edge_context_spec import HOLDING, AT
from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from env.tasks.multi_agent.humanoid_ma_stage1_push import PushRuntime


def config():
    return yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_midpoint_native_rsi.yaml').read_text())['env']


def geometry():
    boxes = torch.zeros(1, 1, 13); boxes[..., 2] = .65; boxes[..., 6] = 1
    local = local_points(torch.full((1, 1), 2), [1.3]*3, torch.full((1, 1), .65))
    target = world_points(boxes, local)
    hands = target[..., None, :].expand(-1, -1, 2, -1).clone()
    return boxes, local, hands


def test_config_and_checkpoint_keep_native_at_separate_from_old_contact():
    env = config(); validate_env(env)
    old = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_rect_native_rsi.yaml').read_text())['env']
    native = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml').read_text())['env']
    for key in ('relationReward', 'relationGraph', 'skillDiscProb'):
        assert env[key] == old[key]
    assert env['pushBox'] == dict(old['pushBox'], mass=20., tie_xy=True)
    assert env['pushGoalRandomization'] == dict(sampling='box_forward_uniform', distance_range=[1., 4.])
    assert env['relationReward'] == native['relationReward']
    check_checkpoint({'relation_experiment_config': {'env': deepcopy(env)}}, env)
    with pytest.raises(ValueError):
        check_checkpoint({'relation_experiment_config': {'env': old}}, env)
    wrong = deepcopy(env); wrong['pushTask']['hand_distance_m'] = .18
    with pytest.raises(ValueError): validate_env(wrong)


def test_tied_xy_grid_and_uniform_initial_goal_distance():
    env = config(); torch.manual_seed(42)
    sizes = sample_box_sizes(10000, 'cpu', env)
    assert torch.equal(sizes[..., 0], sizes[..., 1])
    assert (sizes[..., 2] == 1.3).all()
    assert len(torch.unique(sizes.reshape(-1, 3), dim=0)) == 7
    profiles = torch.tensor(box_size_profiles(env))
    torch.testing.assert_close(torch.unique(sizes.reshape(-1, 3), dim=0), profiles)
    assert len(box_size_profiles(yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_rect_native_rsi.yaml').read_text())['env'])) == 49
    boxes, goals, forward, distance = sample_boxes_and_goals(10000, 'cpu', torch.zeros(10000, 3), env, sizes)
    assert distance.min() >= 1. and distance.max() <= 4.
    assert distance.mean().item() == pytest.approx(2.5, abs=.02)
    for lo in (1., 2., 3.):
        assert ((distance >= lo) & (distance < lo+1)).float().mean().item() == pytest.approx(1/3, abs=.015)
    torch.testing.assert_close(goals-boxes[:, :2, :3], forward*distance[..., None], atol=1e-6, rtol=0)
    assert env['pushTask']['rsi_sampling'] == 'native_size_cache'
    for key in ('interaction_probability', 'late_probability', 'late_fraction', 'rsi_retarget', 'rsi_box_progress_scale', 'motion_hashes'):
        original = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_rect_native_rsi.yaml').read_text())['env']['pushTask']
        assert env['pushTask'][key] == original[key]
    previous = deepcopy(env); previous['pushBox'].pop('tie_xy')
    with pytest.raises(ValueError): check_checkpoint({'relation_experiment_config': {'env': previous}}, env)
    previous = deepcopy(env); previous['pushGoalRandomization']['distance_range'] = [1., 5.]
    with pytest.raises(ValueError): check_checkpoint({'relation_experiment_config': {'env': previous}}, env)


def test_hand_sphere_overlap_distinguishes_center_and_surface_penetration():
    box=torch.zeros(1,13); box[:,6]=1
    size=torch.ones(1,3)
    hands=torch.tensor([[[.52,0.,0.],[.49,0.,0.],[.6,0.,0.],[.52,.52,0.]]])
    depth,inside=hand_box_overlap(hands,box,size)
    torch.testing.assert_close(depth,torch.tensor([[.02,.05,0.,.04-math.sqrt(2)*.02]]),atol=1e-6,rtol=0)
    assert inside.tolist()==[[False,True,False,False]]
    box[:,3:7]=torch.tensor([0.,0.,2**-.5,2**-.5]);box[:,:3]=torch.tensor([2.,3.,1.])
    rotated=hands.clone();rotated[...,0]=-hands[...,1];rotated[...,1]=hands[...,0]
    rotated+=box[:,None,:3]
    other,other_inside=hand_box_overlap(rotated,box,size)
    torch.testing.assert_close(other,depth,atol=1e-6,rtol=0)
    assert torch.equal(other_inside,inside)


@pytest.mark.parametrize('offset', [[1., 0., 0.], [0., .5, 0.], [0., 0., .4]])
def test_individual_hand_separation_does_not_affect_midpoint_score(offset):
    boxes, local, hands = geometry()
    hands[..., 0, :] += torch.tensor(offset); hands[..., 1, :] -= torch.tensor(offset)
    torch.testing.assert_close(midpoint_reward_phi(hands, boxes, local, config()['pushPointReward']), torch.ones(1, 1))


@pytest.mark.parametrize('delta', [[.05, 0., 0.], [0., .2, 0.], [0., 0., .01], [.03, .04, .02]])
def test_continuous_midpoint_score_has_no_height_deadband(delta):
    boxes, local, hands = geometry(); hands += torch.tensor(delta)
    expected = math.exp(-10*(delta[0]**2 + delta[1]**2) - (delta[2]/.04)**2)
    assert midpoint_reward_phi(hands, boxes, local, config()['pushPointReward']).item() == pytest.approx(expected, abs=2e-6)


def test_strict_success_and_native_at_saturation_release_and_partial_reset():
    env = config(); graph = compose_canonical_graph(torch.full((2, 2), 3))
    task = SimpleNamespace(cfg={'env': env}, _push_hits=torch.ones(2, 2, dtype=torch.bool))
    runtime = PushRuntime(task, 2, graph, env['relationReward'], 'cpu')
    native = Stage1ContextRuntime(2, graph, env['relationReward'], 'cpu')
    hold = graph.edge_relation == HOLDING; at = graph.edge_relation == AT
    phi = torch.zeros(2, 4); progress = torch.full_like(phi, .3); z = torch.zeros_like(phi)
    phi[0, hold[0]] = .9; phi[1, hold[1]] = .90001
    result = runtime.step(phi, progress, z, z, z)
    assert not result['own_success'][0, hold[0]].any()
    assert result['own_success'][1, hold[1]].all()
    assert not result['reward_saturated'][0].any()
    phi[hold] = .1; phi[at] = math.exp(-10*.19**2)
    result = runtime.step(phi, progress, z, z, z)
    baseline = native.step(phi, progress, z, z, z)
    assert torch.equal(result['own_success'][at], baseline['own_success'][at])
    assert not result['own_success'][hold].any()
    assert result['reward_saturated'].all()
    torch.testing.assert_close(result['total'], torch.full_like(phi, .6))
    phi[at] = math.exp(-10*.21**2)
    assert not runtime.step(phi, progress, z, z, z)['reward_saturated'].any()
    phi[at] = 1.; z[at] = .002
    assert not runtime.step(phi, progress, z, z, z)['reward_saturated'].any()
    before = runtime.own_success[1].clone()
    runtime.reset(torch.tensor([0]), phi[:1], z[:1], z[:1], z[:1])
    assert torch.equal(runtime.own_success[1], before)


def test_target_tracks_box_translation_and_rotation():
    boxes, local, hands = geometry()
    boxes[..., :2] += torch.tensor([2., 3.])
    boxes[..., 3:7] = torch.tensor([0., 0., 2**-.5, 2**-.5])
    hands = world_points(boxes, local)[..., None, :].expand_as(hands)
    torch.testing.assert_close(midpoint_reward_phi(hands, boxes, local, config()['pushPointReward']), torch.ones(1, 1))

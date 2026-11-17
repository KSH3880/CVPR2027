from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from env.tasks.multi_agent.cliff_bridge_controller import CliffBridgeController
from env.tasks.multi_agent.humanoid_ma_coupled_push import HumanoidMACoupledPush
from env.tasks.multi_agent.stage2_reward import Stage2Runtime
from env.tasks.multi_agent.stage2_wait import Stage2Wait
from utils import stage2_spec
from utils.bridge_before_fixed import NAME, fixed_roles, validate
from utils.premature_push import motion_penalty
from utils.cliff_bridge_spec import COOPERATIVE, sample_roles_for_scene
from utils.experiments import experiment_horizon, native_config, validate_experiment
from utils.sensor_stage2_spec import BRIDGE


def test_config_matches_compact_except_scenario():
    cfg = native_config(NAME)
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    assert experiment_horizon(NAME) == 900
    compact = native_config('stage2_sensor_900_cliff_bridge_compact')
    assert cfg['experiment']['transfer'] == compact['experiment']['transfer']
    env = cfg['env']
    assert env['stateInit'] == 'Default'
    compact['env']['stateInit'] = 'Default'
    assert env['sensorScene']['bridge'].pop('before_training') == dict(
        same_platform_probability=.2, bridge_center_progress=dict(weight=.2, sigma=1.))
    assert env['stage2']['reward_source'] == 'v4_teachers'
    assert env['before']['reward_at_z_tolerance'] == .001
    assert env['before']['waiting_penalty'] == dict(radius_buffer=1.5, coefficient=1., ontop_distance_source='payload')
    assert env['stage2JointPush']['premature_motion_penalty'] == dict(weight=.5, speed_threshold=.05, force_epsilon=1e-6)
    env['stage2'].pop('reward_source')
    env['before'].pop('reward_at_z_tolerance')
    env['before']['waiting_penalty'] = compact['env']['before']['waiting_penalty']
    env['stage2JointPush'].pop('premature_motion_penalty')
    for scene in (cfg, compact):
        scene['env']['sensorScene'].pop('mixture')
        scene['env']['sensorScene']['bridge'].pop('branches')
        scene['env']['sensorScene'].pop('variant')
        scene.pop('experiment')
    assert cfg == compact


def test_validate_rejects_other_scenes_and_branches():
    scene = native_config(NAME)['env']['sensorScene']
    validate(scene)
    wrong = deepcopy(scene)
    wrong['mixture'] = dict(cliff=.4, bridge=.6, room=0.)
    with pytest.raises(ValueError):
        validate(wrong)
    wrong = deepcopy(scene)
    wrong['bridge']['branches'] = dict(controlled=.4, free=.3, cooperative=.3)
    with pytest.raises(ValueError):
        validate(wrong)


def test_fixed_roles():
    torch.manual_seed(0)
    env = native_config(NAME)['env']
    ranges = env['box']['build']['taskSizeRanges']
    n = 3000
    sizes = torch.empty(n, 4, 3)
    sizes[..., :2].uniform_(.4, .6)
    sizes[..., 2].uniform_(.35, .5)
    scenes = torch.full((n,), BRIDGE)
    branches = torch.full((n,), COOPERATIVE)
    counts = torch.full((n,), 2)
    _, groups, _, tasks, roles, push = sample_roles_for_scene(sizes, ranges,
        env['relationGraph']['template_probabilities'], scenes, branches, counts)
    rows = torch.arange(n)
    crosser, follower = roles[:, 1], roles[:, 3]
    groups, pre, tasks = fixed_roles(tasks, groups, roles, sizes[rows, crosser], ranges)
    assert (push.sum(-1) == 2).all() and not push[rows, crosser].any() and not push[rows, follower].any()
    assert (tasks[push] == 5).all()
    assert (tasks[rows, crosser] == 3).all()
    assert (pre[rows, follower] == crosser).all() and ((pre >= 0).sum(-1) == 1).all()
    assert (groups[rows, follower] == crosser).all() and (groups[rows, crosser] == crosser).all()
    pusher_groups = groups[push].view(n, 2)
    assert (pusher_groups[:, 0] == pusher_groups[:, 1]).all()
    share = torch.bincount(tasks[rows, follower], minlength=5)[[1, 2, 4]].float()/n
    assert torch.allclose(share, torch.full((3,), 1/3), atol=.04)


# V4 reward tests ported from the dev repo's test_sensor_v4_reward.py.
def mixed():
    tasks = torch.tensor([[3, 4, 5, 5], [3, 1, 3, 5]])
    groups = torch.tensor([[0, 0, 2, 2], [0, 0, 2, 3]])
    pre = torch.tensor([[-1, 0, -1, -1], [-1, 0, -1, -1]])
    return tasks, groups, pre, stage2_spec.compose_graph(tasks, groups, pre, 8, False)


@pytest.mark.parametrize('z_value', [0., .0009, .0011, .05, .199, .201])
def test_gate_and_precise_reward_have_separate_height_limits(z_value):
    env = native_config(NAME)['env']
    tasks, groups, pre, graph = mixed()
    r = Stage2Runtime(2, graph, env['relationReward'], 'cpu',
        contact_weights=env['jointPush']['contact_weights'], reward_at_z_tolerance=.001, relax_solo_at=True)
    members = torch.where(tasks == 5, (groups[:, :, None] == groups[:, None, :]).sum(-1), 1)
    r.before[:] = stage2_spec.before_members(pre)
    r.relaxed_at_owners[:] = members == 1
    r.contact.fill_(True)
    phi = torch.full_like(graph.edge_relation, .95, dtype=torch.float)
    z = torch.zeros_like(phi)
    at = graph.edge_relation == 7
    z[at] = z_value
    result = r.step(phi, torch.full_like(phi, .7), z, z*0, z*0)
    loose = at & (members == 1).gather(1, graph.edge_owner)
    assert result['raw_own_success'][loose].eq(z_value <= .2).all()
    assert result['raw_own_success'][at & ~loose].eq(z_value <= .001).all()
    assert result['reward_saturated'][at].eq(z_value <= .001).all()
    assert torch.equal(r.before, stage2_spec.before_members(pre))
    r.reset(torch.tensor([1]), phi[1:], z[1:], z[1:]*0, z[1:]*0)
    assert r.own_success[1][loose[1]].eq(z_value <= .2).all()


def test_wait_uses_payload_for_top_and_release_latches_across_partial_reset():
    graph = mixed()[-1]
    wait = Stage2Wait(graph, 2, native_config(NAME)['env']['before']['waiting_penalty'], 'cpu')
    raw = torch.zeros_like(graph.edge_valid)
    wait.reset(torch.arange(2), graph, raw)
    roots = torch.full((2, 4, 3), 20.)
    objects = torch.zeros(2, 8, 13)
    sizes = torch.ones(2, 8, 3)
    goals = torch.zeros(2, 4, 3)
    velocity = torch.ones_like(roots)
    penalty = wait.penalty(roots, velocity, objects, sizes, goals, raw)
    assert penalty[0, 1] > 0 and penalty[1].eq(0).all()
    raw[wait.rows, wait.predecessor] = True
    assert wait.penalty(roots, velocity, objects, sizes, goals, raw).eq(0).all()
    raw.zero_()
    assert wait.penalty(roots, velocity, objects, sizes, goals, raw).eq(0).all()
    wait.reset(torch.tensor([0]), graph, raw)
    assert wait.penalty(roots, velocity, objects, sizes, goals, raw)[0, 1] > 0


@pytest.mark.parametrize('enabled', [False, True])
def test_controller_substeps_run_feedback_only_in_v4(monkeypatch, enabled):
    from env.tasks.multi_agent import cliff_bridge_controller as module
    monkeypatch.setattr(module.gymtorch, 'unwrap_tensor', lambda value: value)
    events = []
    gym = SimpleNamespace(apply_rigid_body_force_tensors=lambda *args: events.append('force'),
        simulate=lambda *args: events.append('simulate'), fetch_results=lambda *args: events.append('fetch'))
    task = SimpleNamespace(_v4_rewards=enabled, num_envs=2, device='cpu', control_freq_inv=2,
        gym=gym, sim=None, _controlled_forces=torch.zeros(2, 3), render=lambda: events.append('render'),
        _refresh_sim_tensors=lambda: events.append('refresh'),
        _update_push_feedback=lambda ids: events.append('feedback'))
    CliffBridgeController._physics_step(task)
    expected = ['render', 'force', 'simulate']+(['fetch', 'refresh', 'feedback'] if enabled else [])
    assert events == expected*2


def test_feedback_contact_latches_and_premature_penalty_averages():
    env = native_config(NAME)['env']
    boxes = torch.zeros(1, 4, 13)
    boxes[..., 7] = .2
    hands = torch.zeros(1, 4, 2, 3)
    hands[:, 0, :, 0] = -2.
    normal = torch.zeros(1, 4, 2)
    normal[..., 0] = 1.
    near = torch.zeros(1, 4, 2, dtype=torch.bool)
    near[:, 0] = True
    task = SimpleNamespace(_joint=env['jointPush'], _push_hits=torch.zeros(1, 4, dtype=torch.bool),
        _premature_cfg=env['stage2JointPush']['premature_motion_penalty'],
        _premature_penalty=torch.zeros(1, 4), control_freq_inv=2,
        _rigid_body_pos=None, _contact_forces=hands, _key_body_ids=[0, 1],
        _ontop_box_contact=torch.ones(1, 4, 3), _assigned_box_values=lambda value: value,
        _members=torch.tensor([[2, 2, 1, 1]]), _templates_by_owner=torch.tensor([[5, 5, 3, 5]]),
        _groups=torch.tensor([[0, 0, 2, 3]]))
    task._contact_geometry = lambda *args: (torch.zeros(1, 4), normal, None, boxes, None, dict(near=near))
    HumanoidMACoupledPush._update_push_feedback(task, torch.tensor([0]))
    assert task._push_hits.tolist() == [[True, False, False, False]]
    torch.testing.assert_close(task._premature_penalty, torch.tensor([[-.0375, 0., 0., 0.]]))
    task._contact_forces.zero_()
    HumanoidMACoupledPush._update_push_feedback(task, torch.tensor([0]))
    assert task._push_hits[0, 0]
    torch.testing.assert_close(task._premature_penalty, torch.tensor([[-.0375, 0., 0., 0.]]))
    near[:, 1] = True
    force = hands.clone()
    force[:, 0, :, 0] = -2.
    assert motion_penalty(boxes[..., 7:9], force[..., :2], near, task._groups,
        task._members > 1, task._premature_cfg)[0].eq(0).all()

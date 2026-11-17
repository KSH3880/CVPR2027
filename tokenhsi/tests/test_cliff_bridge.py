from copy import deepcopy

import pytest
import torch

from utils import stage2_spec
from utils.cliff_bridge_layout import crossing_layout
from utils.cliff_bridge_spec import BRANCHES, NAME, bridge_control, bridge_schedule, bridge_trajectory, sample_branches, sample_roles_for_scene
from utils.coupled_push_spec import check_checkpoint, checkpoint_contract, group_sizes
from utils.experiments import experiment_horizon, native_config, validate_experiment
from utils.sensor_stage2_spec import paired_people
from env.tasks.multi_agent.stage2_reward import Stage2Runtime


def test_config_and_checkpoint_isolation():
    cfg = native_config(NAME)
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    assert experiment_horizon(NAME) == 900
    assert cfg['env']['sensorScene']['mixture'] == dict(cliff=.4, bridge=.6, room=0.)
    assert cfg['experiment']['transfer']['checkpoint'] == 'output/stage2/Stage2Distill900_00001500.pth'
    weights = dict(coupled_push_contract=checkpoint_contract(cfg['env']))
    check_checkpoint(weights, cfg['env'])
    with pytest.raises(ValueError):
        check_checkpoint(weights, native_config('stage2_sensor_900_room_before')['env'])
    original = native_config('stage2_sensor_900_room_before')
    expected = deepcopy(cfg)
    expected['experiment']['name'] = original['experiment']['name']
    scene = expected['env']['sensorScene']
    scene.pop('variant')
    scene['mixture'] = original['env']['sensorScene']['mixture']
    scene['bridge'].pop('controller')
    scene['bridge'].pop('branches')
    assert expected == original


def test_sampling_roles_exposure_and_reserved_bridge_token():
    torch.manual_seed(42)
    n = 24000
    env = native_config(NAME)['env']
    scenes = torch.multinomial(torch.tensor([.4, .6]), n, True)
    branches = sample_branches(n, env['sensorScene']['bridge']['branches'], 'cpu')
    counts = torch.randint(1, 4, (n,))
    sizes = torch.tensor([.5, .5, .4]).expand(n, 4, 3).clone()
    domain, groups, pre, tasks, roles, push = sample_roles_for_scene(sizes,
        env['box']['build']['taskSizeRanges'], env['relationGraph']['template_probabilities'], scenes, branches, counts)
    cliff, bridge = scenes == 0, scenes == 1
    assert (pre[cliff] == -1).all()
    assert ((tasks[cliff] >= 1) & (tasks[cliff] <= 4)).all()
    before = paired_people(pre)
    crossers = torch.where(branches == 0, 4, torch.where(branches == 1, 3, 1))*bridge
    coupled = (tasks == 5) & (group_sizes(groups) > 1)
    torch.testing.assert_close(torch.stack((before.float().mean(), crossers.float().mean()/4,
        coupled.float().mean())), torch.tensor([.24, .42, .075]), atol=.012, rtol=0)
    for branch, people in ((0, 0), (1, 1)):
        selected = bridge & (branches == branch)
        assert (push[selected].sum(-1) == people).all()
    free = bridge & (branches == 1)
    assert (before[free].sum(-1) == 2).all()
    assert not before[free].gather(1, roles[free, 3:4]).any()
    one = bridge & (branches == 2) & (counts == 1)
    assert (before[one].sum(-1) == 2).all()
    controlled = bridge & (branches == 0)
    packet = stage2_spec.compose_graph(tasks, groups, pre, 8)
    _, _, used = stage2_spec.physical_mapping(tasks, groups, pre, packet)
    assert (~used[controlled, 4:]).any(-1).all()
    assert (stage2_spec.required_tokens(groups, pre, tasks) <= 8).all()
    for name in BRANCHES:
        assert sample_branches(4, {}, 'cpu', name).eq(BRANCHES.index(name)).all()


def test_force_and_trajectory_follow_square_sensor_equations():
    spec = native_config(NAME)['env']['sensorScene']['bridge']['controller']
    start, goal = torch.zeros(3, 3), torch.tensor([[0., -3., 0.]]).expand(3, -1)
    arrival = torch.tensor([1., 7., 18.])
    duration, wait = bridge_schedule(torch.full((3,), 3.), arrival, spec['speed_range'], 1/30)
    position, velocity = bridge_trajectory(start, goal, arrival-wait, duration)
    torch.testing.assert_close(position, goal)
    torch.testing.assert_close(velocity, torch.zeros_like(velocity))
    state = torch.zeros(3, 13)
    mass = torch.tensor([10., 500., 3000.])
    force = bridge_control(goal, velocity, state, mass, spec)
    torch.testing.assert_close(force.norm(dim=-1)/mass, torch.full((3,), 6.))
    state[:, :3] = goal
    assert not bridge_control(goal, velocity, state, mass, spec).any()


@pytest.mark.parametrize('follower_task', [1, 2, 4])
def test_crossing_before_follower_waits_for_leader_at(follower_task):
    env = native_config(NAME)['env']
    tasks = torch.tensor([[3, follower_task, 3, follower_task]])
    pre = torch.tensor([[-1, 0, -1, 2]])
    groups = torch.tensor([[0, 0, 2, 2]])
    graph = stage2_spec.compose_graph(tasks, groups, pre, 8, shuffle=False)
    runtime = Stage2Runtime(1, graph, env['relationReward'], 'cpu',
        contact_weights=env['jointPush']['contact_weights'])
    runtime.before[:] = paired_people(pre)
    phi = torch.ones_like(graph.edge_relation, dtype=torch.float)
    phi[graph.edge_relation == 7] = 0.
    zero = torch.zeros_like(phi)
    dependent = graph.prereq_mask.any(-1)
    result = runtime.step(phi, zero, zero, zero, zero)
    assert not result['own_success'][dependent].any()
    phi[graph.edge_relation == 7] = 1.
    result = runtime.step(phi, zero, zero, zero, zero)
    assert result['own_success'][dependent].all()


@pytest.mark.parametrize('branch', [0, 1])
def test_before_crossing_layout_all_spans_and_directions(branch):
    torch.manual_seed(42)
    n = 72
    env = native_config(NAME)['env']
    sizes = torch.tensor([.5, .5, .4]).expand(n, 4, 3).clone()
    _, _, pre, tasks, roles, push = sample_roles_for_scene(sizes, env['box']['build']['taskSizeRanges'],
        env['relationGraph']['template_probabilities'], torch.ones(n, dtype=torch.long),
        torch.full((n,), branch), torch.ones(n, dtype=torch.long))
    phase = torch.where(push, -1, 0)
    progress = torch.zeros(n, 4)
    if branch == 1:
        phase[torch.arange(n), roles[:, 3]] = torch.arange(n)%2+1
        progress[torch.arange(n), roles[:, 3]] = torch.where(torch.arange(n)%2 == 0, .35, .75)
    span = torch.arange(n)%3+1
    bridge = torch.tensor([1.8, 1.8, 1.5]).expand(n, -1).clone()
    bridge[:, 0] *= span
    gap = bridge[:, 0]+.2
    source = torch.zeros(n, 2)
    destination = torch.stack((8+gap, torch.zeros(n)), -1)
    reverse = torch.arange(n)%2 == 0
    source[reverse], destination[reverse] = destination[reverse].clone(), source[reverse].clone()
    landing = torch.stack((4+gap/2, torch.zeros(n)), -1)
    half = torch.full((n, 2), 4.)
    radii = torch.full((n, 4, 2), .36)
    near = (pre >= 0) & (torch.rand(n, 4) < .5)
    pending = torch.arange(n)
    for _ in range(32):
        payload, target, humans, failed = crossing_layout(source[pending], half[pending], destination[pending],
            half[pending], landing[pending], bridge[pending], gap[pending], tasks[pending], pre[pending],
            radii[pending], phase[pending], progress[pending], near[pending],
            env['sensorScene']['bridge']['curriculum'], 2.5)
        assert torch.isfinite(humans).all()
        good = ~failed
        for person in range(4):
            pair = good & (pre[pending, person] >= 0)
            leader = pre[pending[pair], person]
            if pair.any():
                torch.testing.assert_close(target[pair, person], target[pair, leader])
                if (tasks[pending[pair], person] < 3).any():
                    sit = pair & (tasks[pending, person] < 3)
                    torch.testing.assert_close(payload[sit, person], payload[sit, pre[pending[sit], person]])
        pending = pending[failed]
        if not len(pending):
            break
    assert not len(pending), (branch, pending.tolist())

from copy import deepcopy

import pytest
import torch

from utils.experiments import native_config, validate_experiment, experiment_horizon
from utils.unified_training import validate_unified_env
from utils import stage2_spec as spec
from utils.premature_push import motion_penalty
from env.tasks.multi_agent.stage2_reward import Stage2Runtime
from env.tasks.multi_agent.stage2_wait import Stage2Wait


NAME = 'stage2_900_v4_reward'
ENV = native_config(NAME)['env']


def mixed_graph():
    templates = torch.tensor([[3, 4, 5, 5], [3, 1, 3, 5]])
    groups = torch.tensor([[0, 0, 2, 2], [0, 0, 2, 3]])
    predecessor = torch.tensor([[-1, 0, -1, -1], [-1, 0, -1, -1]])
    return spec.compose_graph(templates, groups, predecessor, 8, False)


def test_new_ppo_contract_and_reject_changed_teacher_rewards():
    train = native_config(NAME, True)
    validate_unified_env(ENV)
    validate_experiment(NAME, ENV, train)
    assert experiment_horizon(NAME) == ENV['episodeLength'] == 900
    assert train['params']['algo']['name'] == 'ma'
    assert 'distillation' not in train['params']['config']
    for key in ('reward_at_z_tolerance', 'waiting_penalty'):
        bad = deepcopy(ENV)
        bad['before'].pop(key)
        with pytest.raises(ValueError):
            validate_unified_env(bad)
    bad = deepcopy(ENV)
    bad['stage2JointPush']['premature_motion_penalty']['weight'] = 0.
    with pytest.raises(ValueError):
        validate_unified_env(bad)


def test_loose_enable_and_solo_success_keep_strict_reward_and_coupled_success():
    graph = mixed_graph()
    runtime = Stage2Runtime(2, graph, ENV['relationReward'], 'cpu',
        contact_weights=ENV['jointPush']['contact_weights'], reward_at_z_tolerance=.001)
    runtime.before[:] = torch.tensor([[True, True, False, False], [True]*4])
    runtime.contact.fill_(True)
    phi = torch.full_like(graph.edge_relation, .95, dtype=torch.float)
    progress = torch.full_like(phi, .7)
    z = torch.zeros_like(phi)
    at = graph.edge_relation == 7
    z[at] = .05
    result = runtime.step(phi, progress, z, z*0, z*0)
    loose = at & runtime.before.gather(1, graph.edge_owner)
    strict = at & ~runtime.before.gather(1, graph.edge_owner)
    assert result['raw_own_success'][loose].all()
    assert not result['raw_own_success'][strict].any()
    assert not result['reward_saturated'][at].any()
    assert result['success_component'][at].eq(0).all()
    assert result['own_success'][graph.prereq_mask.any(-1)].all()
    ids = torch.tensor([1])
    runtime.reset(ids, phi[ids], z[ids], z[ids]*0, z[ids]*0)
    assert runtime.own_success[1, loose[1]].all()
    result = runtime.step(phi, progress, z*0, z*0, z*0)
    assert result['reward_saturated'][at].all()


def test_waiting_ontop_arms_from_payload_and_release_survives_success_loss():
    graph = mixed_graph()
    wait = Stage2Wait(graph, 2, ENV['before']['waiting_penalty'], 'cpu')
    raw = torch.zeros_like(graph.edge_valid)
    wait.reset(torch.arange(2), graph, raw)
    roots = torch.full((2, 4, 3), 20.)
    objects = torch.zeros(2, 8, 13)
    sizes = torch.ones(2, 8, 3)
    goals = torch.zeros(2, 4, 3)
    velocity = torch.ones_like(roots)
    penalty = wait.penalty(roots, velocity, objects, sizes, goals, raw)
    assert penalty[0, 1] > 0
    assert penalty[1].eq(0).all()
    raw[wait.rows, wait.predecessor] = True
    assert wait.penalty(roots, velocity, objects, sizes, goals, raw).eq(0).all()
    raw.zero_()
    assert wait.penalty(roots, velocity, objects, sizes, goals, raw).eq(0).all()
    wait.reset(torch.tensor([0]), graph, raw)
    assert wait.penalty(roots, velocity, objects, sizes, goals, raw)[0, 1] > 0


def test_premature_penalty_charges_only_pusher_until_entire_group_is_ready():
    config = ENV['stage2JointPush']['premature_motion_penalty']
    velocity = torch.tensor([[[.2, 0.]]*4])
    forces = torch.tensor([[[[-2., 0.]]*2, [[0., 0.]]*2, [[-2., 0.]]*2, [[0., 0.]]*2]])
    near = torch.tensor([[[True]*2, [False]*2, [True]*2, [False]*2]])
    groups = torch.tensor([[0, 0, 2, 3]])
    eligible = torch.tensor([[True, True, False, False]])
    penalty, _, _ = motion_penalty(velocity, forces, near, groups, eligible, config)
    torch.testing.assert_close(penalty, torch.tensor([[-.075, 0., 0., 0.]]))
    near[:, 1] = True
    assert motion_penalty(velocity, forces, near, groups, eligible, config)[0].eq(0).all()

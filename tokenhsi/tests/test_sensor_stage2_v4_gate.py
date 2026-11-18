from types import SimpleNamespace

import isaacgym
import pytest
import torch

from env.tasks.multi_agent.humanoid_ma_sensor_stage2 import HumanoidMASensorStage2
from env.tasks.multi_agent.stage2_reward import Stage2Runtime
from utils import stage2_spec
from utils.experiments import native_config
from utils.sensor_stage2_spec import WAREHOUSE


def make_task(v4, family):
    env = native_config('stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress')['env']
    tasks = torch.tensor([[5, 5, 3, family], [5, 3, family, 5], [3, family, 3, family]])
    groups = torch.tensor([[0, 0, 2, 2], [0, 1, 1, 0], [0, 0, 2, 2]])
    pre = torch.tensor([[-1, -1, -1, 2], [-1, -1, 1, -1], [-1, 0, -1, 2]])
    domain = torch.tensor([2, 2, 0])
    graph = stage2_spec.compose_graph(tasks, groups, pre, 8, False)
    runtime = Stage2Runtime(3, graph, env['relationReward'], 'cpu',
        contact_weights=env['jointPush']['contact_weights'],
        before_at_z_tolerance=env['stage2']['before_at_z_tolerance'],
        reward_at_z_tolerance=env['before']['reward_at_z_tolerance'], relax_solo_at=v4)
    if v4:
        runtime.relaxed_at_owners[1] = True
    task = SimpleNamespace(num_agents=4, num_objects=8, device='cpu', relation_runtime=runtime,
        _draw_episode_graph=lambda ids: (domain[ids], groups[ids], pre[ids], tasks[ids]),
        _relation_graph_spec=env['relationGraph'], _leader_bank=24,
        _bridge_pushers=torch.zeros(3, 4, dtype=torch.bool),
        _scene_ids=torch.full((3,), WAREHOUSE), _roles=torch.arange(4).repeat(3, 1),
        _bridge_slot=torch.full((3,), 20), _bridge_training_init=None,
        _unreachable=torch.zeros(3, 4, dtype=torch.bool), _sampling_counts=torch.zeros(18),
        _stage2_domain_counts=torch.zeros(3), _update_box_assignment_colors=lambda ids: None,
        _count_sampling=lambda *args: None, _before_near_pairs=torch.zeros(3, 4, dtype=torch.bool),
        _logical_box_order=torch.zeros(3, 8, dtype=torch.long),
        _stage2_task_slots=torch.zeros(3, 8, dtype=torch.bool),
        _scene_domains=torch.zeros(3, dtype=torch.long),
        _edge_goal_owners=torch.zeros(3, 4, dtype=torch.bool),
        _ontop_maintain_steps=torch.zeros_like(graph.edge_relation),
        _push_hits=torch.zeros(3, 4, dtype=torch.bool))
    for name in ('_groups', '_templates', '_before_predecessor', '_members',
                 '_regions', '_agent_box_assignment'):
        setattr(task, name, torch.zeros(3, 4, dtype=torch.long))
    return task


@pytest.mark.parametrize('v4', [False, True])
@pytest.mark.parametrize('family', [1, 2, 4])
@pytest.mark.parametrize('z_error', [.05, .201])
def test_scene_sampler_connects_at_height_gate_without_relaxing_precise_reward(v4, family, z_error):
    task = make_task(v4, family)
    runtime = task.relation_runtime
    untouched = runtime.relaxed_at_owners[1].clone() if v4 else None
    ids = torch.tensor([2, 0])
    HumanoidMASensorStage2._sample_episode_graph(task, ids)
    if v4:
        torch.testing.assert_close(runtime.relaxed_at_owners[ids], task._members[ids] == 1)
        torch.testing.assert_close(runtime.relaxed_at_owners[1], untouched)
    else:
        assert runtime.relaxed_at_owners is None

    graph = runtime.graph
    selected = torch.zeros(3, 1, dtype=torch.bool)
    selected[ids] = True
    at = (graph.edge_relation == 7) & graph.edge_valid & selected
    solo = at & (task._members == 1).gather(1, graph.edge_owner)
    dependent = graph.prereq_mask.any(-1) & graph.edge_valid & selected
    phi = torch.full_like(graph.edge_relation, .95, dtype=torch.float)
    z = torch.where(at, z_error, 0.)
    zero = torch.zeros_like(phi)
    runtime.reset(ids, phi[ids], z[ids], zero[ids], zero[ids])
    assert runtime.own_success[solo].eq(z_error <= .2).all()
    result = runtime.step(phi, phi * .7, z, zero, zero)
    assert result['raw_own_success'][solo].eq(z_error <= .2).all()
    assert not result['raw_own_success'][at & ~solo].any()
    assert result['prerequisite_gate'][dependent].eq(z_error <= .2).all()
    assert not result['reward_saturated'][at].any()

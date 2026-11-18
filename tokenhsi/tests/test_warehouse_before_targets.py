from types import SimpleNamespace

import isaacgym
import pytest
import torch

from env.tasks.multi_agent import warehouse_before, humanoid_ma_stage2
from env.tasks.multi_agent.humanoid_ma_coupled_push import HumanoidMACoupledPush
from env.tasks.multi_agent.before_reward import predecessor_at_progress
from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges
from utils import stage2_spec
from utils.evaluation_protocol import spatial_success
from utils.experiments import native_config
from utils.geometry import rotate
from env.tasks.multi_agent.square_sensor_scene import yaw_quat


@pytest.mark.parametrize('yaw', [0., 1.3, -2.4])
@pytest.mark.parametrize('default', [True, False])
def test_apron_normalizes_only_default_unused_targets_after_group_transform(monkeypatch, yaw, default):
    n, m = 3, 4
    origins = torch.tensor([[500., 100., 0.], [900., -200., 0.], [1500., 600., 0.]])
    roots = torch.zeros(n, m, 13)
    roots[..., 0] = torch.arange(m)*1.5
    roots[..., 2], roots[..., 6] = 1., 1.
    if not default:
        roots[..., :2] += origins[:, None, :2]
    goals = origins[:, None].expand(-1, m, -1).clone()
    goals[:, ::2] = roots[:, ::2, :3]+torch.tensor([2., 1., -.6])
    pre = torch.tensor([[-1, 0, -1, 2]]).repeat(n, 1)
    task = SimpleNamespace(num_agents=m, device='cpu', _before_predecessor=pre,
        _humanoid_root_states=roots, _tar_pos=goals, _env_origins=origins,
        _place_failed=torch.zeros(n, dtype=torch.bool),
        _surface_half=torch.ones(n, 4, 2, 2)*4,
        _warehouse={'wall_thickness': .4}, _warehouse_coop={'apron_length': 6.},
        StateInit=SimpleNamespace(Default=0), _state_init=0 if default else 1)
    task._group_entities = lambda ids: None
    transformed_goals = goals.clone()

    def place_group(ids, rows, leader, head, entities, occupied, region):
        q = yaw_quat(torch.tensor([yaw]))[0]
        for row in rows.tolist():
            e = ids[row]
            member = head[row] == leader
            shift = origins[e]-rotate(q, roots[e, leader, :3])+torch.tensor([-7., 0., 1.])
            roots[e, member, :3] = rotate(q, roots[e, member, :3])+shift
            goals[e, member] = rotate(q, goals[e, member])+shift
            transformed_goals[e, member] = goals[e, member]

    task._place_group = place_group
    monkeypatch.setattr(warehouse_before, '_park_unused', lambda *args: None)
    old_roots, old_goals = roots.clone(), goals.clone()
    ids = torch.tensor([2, 0])
    warehouse_before.place(task, ids)
    follower = pre[ids] >= 0
    if default:
        torch.testing.assert_close(goals[ids][follower], roots[ids, :, :3][follower])
    else:
        torch.testing.assert_close(goals, transformed_goals)
    before = (old_goals[ids]-old_roots[ids, :, :3]).norm(dim=-1)
    after = (goals[ids]-roots[ids, :, :3]).norm(dim=-1)
    torch.testing.assert_close(after[~follower], before[~follower], atol=2e-4, rtol=1e-4)
    assert (after[~follower] > 1.).all()
    torch.testing.assert_close(roots[1], old_roots[1])
    torch.testing.assert_close(goals[1], old_goals[1])
    torch.testing.assert_close(roots[..., 3:], old_roots[..., 3:])


@pytest.mark.parametrize('family', [1, 2, 4])
def test_unused_targets_do_not_change_rewards_gates_or_success(family):
    torch.manual_seed(5)
    templates = torch.tensor([[3, family, 3, family]])
    pre = torch.tensor([[-1, 0, -1, 2]])
    graph = stage2_spec.compose_graph(templates, torch.tensor([[0, 0, 2, 2]]), pre, 8, shuffle=True)
    roots, hands, feet = torch.rand(1, 4, 3), torch.rand(1, 4, 2, 3), torch.rand(1, 4, 2, 3)
    objects = torch.zeros(1, 8, 13)
    objects[..., :3], objects[..., 6] = torch.rand(1, 8, 3), 1.
    sizes = torch.rand(1, 8, 3)+.2
    goals = torch.rand(1, 4, 3)
    goals[pre >= 0] = 1200.
    corrected = torch.where((pre >= 0)[..., None], roots, goals)
    cfg = native_config('stage2_sensor_900_warehouse_fixedroles_v4_beforepairs_doorprogress')['env']['relationReward']
    results = [evaluate_interaction_edges(hands, feet, roots, objects, sizes, target, graph, cfg, .94)
               for target in (goals, corrected)]
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for name in results[0][1]:
        torch.testing.assert_close(results[0][1][name], results[1][1][name], rtol=0, atol=0)
    progress = [predecessor_at_progress(result[1]['progress'], roots, objects, sizes, target, graph, cfg)
                for result, target in zip(results, (goals, corrected))]
    torch.testing.assert_close(*progress, rtol=0, atol=0)
    packet = stage2_spec.task_graph_packet(graph, 1).reshape(1, 4, 5)
    torch.testing.assert_close(spatial_success(packet, roots, objects, sizes, goals),
                               spatial_success(packet, roots, objects, sizes, corrected))


@pytest.mark.parametrize('method,owners', [('_reset_boxes', [0, 2, 3, 0, 1, 2, 3]),
                                        ('_configure_ontop_targets', [0, 2, 0, 1, 2, 3])])
@pytest.mark.parametrize('has_default', [False, True])
@pytest.mark.parametrize('has_loco', [False, True])
def test_shared_object_sampling_filters_default_like_loco_and_restores_metadata(monkeypatch, method, owners, has_default, has_loco):
    task = humanoid_ma_stage2.HumanoidMAStage2.__new__(humanoid_ma_stage2.HumanoidMAStage2)
    slots = (torch.arange(2).repeat_interleave(4), torch.arange(4).repeat(2))
    task._reset_default_slots = slots if has_default else None
    task._reset_ref_slots = {'loco': slots} if has_loco else {}
    task._before_predecessor = torch.tensor([[-1, 0, -1, 2], [-1, -1, -1, -1]])
    task._templates = torch.tensor([[3, 1, 3, 4], [5, 1, 2, 3]])
    task.num_agents = 4
    seen = []

    def parent(self, ids):
        for selected in (self._reset_default_slots, self._reset_ref_slots.get('loco')):
            if selected is not None:
                assert selected[1].tolist() == owners
        seen.append(True)

    monkeypatch.setattr(HumanoidMACoupledPush, method, parent)
    monkeypatch.setattr(humanoid_ma_stage2, 'apply_before_layout', lambda *args: None)
    getattr(task, method)(torch.arange(2))
    assert seen == [True]
    assert task._reset_default_slots is (slots if has_default else None)
    if has_loco:
        assert task._reset_ref_slots['loco'] is slots
    else:
        assert task._reset_ref_slots == {}

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from utils.box_cleanup_spec import BOX_SIZE, cleanup_graph_spec, cleanup_layout, cleanup_reward_config
from utils.edge_stage1_spec import compile_stage1_graph, semantic_graph_packet
from utils.edge_scenario_spec import agent_object_indices, agent_goal_indices
from utils.edge_ontop_spec import expand_graph
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata


def test_two_rounds_assign_eight_distinct_boxes_and_keep_owner_goals():
    assigned = []
    for round_index in (0, 1):
        graph = expand_graph(compile_stage1_graph(cleanup_graph_spec(round_index), 4, 16), 1)
        env = torch.zeros(4, dtype=torch.long)
        agents = torch.arange(4)
        boxes = agent_object_indices(graph, env, agents)
        assigned.extend(boxes.tolist())
        assert torch.equal(boxes, agents + 4 * round_index)
        assert torch.equal(agent_goal_indices(graph, env, agents), agents)
        assert semantic_graph_packet(graph, 1).shape == (1, 40)
        assert graph.required_goal.all()
        assert not graph.prereq_mask.any()
        assert (graph.term_index == -1).all()
    assert len(set(assigned)) == 8


def test_layout_has_only_pile_upper_targets_and_separate_destinations():
    for seed in range(100):
        torch.manual_seed(seed)
        sizes = torch.randint(3, (16, 3)) * .05 + torch.tensor([.4, .4, .3])
        humans, boxes, rotations, goals = cleanup_layout(sizes)
        assert torch.allclose(boxes[8:, 2], sizes[8:, 2] / 2)
        assert torch.allclose(boxes[:8, 2] - sizes[:8, 2] / 2, sizes[8:, 2] + .001)
        assert (boxes[:8, :2] - boxes[8:, :2]).abs().max() <= .015
        assert boxes[:, :2].abs().max() < 1.2
        assert torch.allclose(rotations.norm(dim=-1), torch.ones(16))
        assert (rotations[:, :2] == 0).all()
        yaw = 2 * torch.atan2(rotations[:, 2], rotations[:, 3])
        assert yaw.abs().max() <= torch.pi / 12
        assert yaw.std() > 0
        extent = sizes.clone() / 2
        extent[:, 0] = (yaw.cos().abs() * sizes[:, 0] + yaw.sin().abs() * sizes[:, 1]) / 2
        extent[:, 1] = (yaw.sin().abs() * sizes[:, 0] + yaw.cos().abs() * sizes[:, 1]) / 2
        for a in range(16):
            for b in range(a):
                assert ((boxes[a] - boxes[b]).abs() >= extent[a] + extent[b] - 1e-6).any()
        assert torch.allclose(goals[..., 2], sizes[:8, 2].reshape(2, 4) / 2)
    directions = humans / humans.norm(dim=-1, keepdim=True)
    assert (((goals[..., :2] - humans[None]) * directions).sum(-1) > 0).all()
    destinations = goals[..., :2].flatten(0, 1)
    distances = torch.cdist(destinations, destinations)
    distances.fill_diagonal_(torch.inf)
    assert distances.min() > .7


def test_box_colors_are_uniform_across_both_rounds():
    from types import SimpleNamespace
    from env.tasks.multi_agent.box_cleanup_demo import BoxCleanupDemo
    calls = []
    task = SimpleNamespace(viewer=True, _video_enabled=False, num_objects=16,
                           _box_handles=list(range(16)), envs=['scene'],
                           gym=SimpleNamespace(set_rigid_body_color=lambda *args: calls.append(args)))
    for _ in range(2):
        BoxCleanupDemo._update_box_assignment_colors(task, torch.tensor([0]))
    assert len(calls) == 32
    assert {args[1] for args in calls} == set(range(16))
    assert all((args[-1].x, args[-1].y, args[-1].z) == pytest.approx((.45, .45, .45))
               for args in calls)


def test_demo_alias_is_explicit_preserves_checkpoint_and_still_checks_contract():
    path = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent/approach_scenario_stage1_self_sum.yaml'
    current = yaml.safe_load(path.read_text())['env']['relationReward']
    saved = deepcopy(current)
    saved['stage1_variant'] += '_rescue'
    saved['hard_skill_training']['at_placement'] = {'xy_radius': .5, 'z_scale': .25, 'reward_weight': .6}
    saved['hard_skill_training']['climb_state'] = {'xy_scale': 2., 'height_scale': 4.}
    before = deepcopy(saved)
    canonical = cleanup_reward_config(saved)
    assert canonical == current
    assert saved == before
    metadata = checkpoint_metadata(canonical)
    check_checkpoint_metadata({'relation_metadata': metadata}, checkpoint_metadata(current))
    metadata['graph_record_width'] = 6
    with pytest.raises(ValueError, match='graph packet mismatch'):
        check_checkpoint_metadata({'relation_metadata': metadata}, checkpoint_metadata(current))
    with pytest.raises(ValueError, match='rescue checkpoint'):
        cleanup_reward_config(current)


def test_round_barrier_requires_all_four_settled_and_preserves_physical_scene(monkeypatch):
    from env.tasks.multi_agent.box_cleanup_demo import BoxCleanupDemo
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    monkeypatch.setattr(HumanoidMACarry, 'post_physics_step', lambda self: None)
    task = BoxCleanupDemo.__new__(BoxCleanupDemo)
    task.device = 'cpu'
    task.cleanup_finished = False
    task.cleanup_round = 0
    task.cleanup_stable = torch.zeros(4, dtype=torch.long)
    task.cleanup_waiting = torch.zeros(4, dtype=torch.bool)
    task.cleanup_completed = torch.zeros(8, dtype=torch.bool)
    task.cleanup_events = []
    task.reset_buf = torch.zeros(1)
    task.progress_buf = torch.ones(1, dtype=torch.long)
    task._box_states = torch.zeros(1, 16, 13)
    task._box_size = torch.tensor(BOX_SIZE).repeat(1, 16, 1)
    task._box_size[0, :, 2] = torch.tensor([.3, .35, .4, .3] * 4)
    task.cleanup_goals = cleanup_layout(task._box_size[0])[3]
    task._tar_pos = task.cleanup_goals[0:1].clone()
    task._box_states[0, :4, :3] = task._tar_pos[0]
    task._rigid_body_pos = torch.full((1, 4, 2, 3), 10.)
    task._key_body_ids = torch.arange(2)
    task._agent_box_assignment = torch.arange(4)[None]
    task._env_origins = torch.zeros(1, 3)
    task._enable_markers = False
    task.obs_buf = torch.zeros(1, 1)
    task.extras = {}
    calls = []
    task._sample_episode_graph = lambda ids: calls.append('graph')
    task._reset_relation_history = lambda ids: calls.append('history')
    task._compute_observations = lambda: calls.append('obs')
    task._update_box_assignment_colors = lambda ids: None
    initial = task._box_states.clone()
    task._box_states[0, 3, 2] = .8
    for _ in range(20):
        task.post_physics_step()
    assert task.cleanup_round == 0
    assert task.cleanup_waiting.tolist() == [True, True, True, False]
    task._box_states.copy_(initial)
    for _ in range(14):
        task.post_physics_step()
    assert task.cleanup_round == 0
    task.post_physics_step()
    assert task.cleanup_round == 1
    assert torch.equal(task._box_states, initial)
    assert calls == ['graph', 'history', 'obs']
    assert torch.equal(task._agent_box_assignment[0], torch.arange(4, 8))
    task._box_states[0, 4:8, :3] = task._tar_pos[0]
    for _ in range(15):
        task.post_physics_step()
    assert task.cleanup_finished
    assert task.cleanup_completed.all()
    assert len(task.cleanup_events) == 8

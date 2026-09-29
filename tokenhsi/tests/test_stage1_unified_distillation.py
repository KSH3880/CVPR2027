from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import isaacgym
import torch
import yaml
from torch.distributions import Independent, Normal, kl_divergence

from learning.amp_datasets import AMPDataset
from learning.common_agent import CommonAgent
from learning.multi_agent.distillation import gaussian_forward_kl
from learning.multi_agent.ma_agent import MAAgent
from learning.multi_agent.stage1_unified_teacher import Stage1UnifiedTeacher
from env.tasks.multi_task.humanoid_traj_sit_carry_climb import compute_location_observations
from utils.edge_scenario_spec import sample_graph, agent_goal_indices, agent_object_indices
from utils.edge_ontop_spec import ON_TOP


CONFIG = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent'
BASE = yaml.safe_load((CONFIG / 'approach_scenario_stage1_self_sum.yaml').read_text())['env']
DISTILL = yaml.safe_load((CONFIG / 'approach_scenario_stage1_self_sum_distill.yaml').read_text())['env']


def test_distill_config_keeps_34_environment_contract():
    assert DISTILL == BASE


def _task_with_all_templates(role_swap=False):
    presets = ('holding', 'sit', 'climb', 'holding_at', 'holding_ontop')
    graphs = [sample_graph(1, BASE['relationGraph'], preset=preset,
                           role_swap=role_swap, generator=torch.Generator().manual_seed(i))
              for i, preset in enumerate(presets)]
    graph = type(graphs[0])(**{
        field.name: (torch.cat([getattr(g, field.name) for g in graphs])
                     if isinstance(getattr(graphs[0], field.name), torch.Tensor)
                     else getattr(graphs[0], field.name))
        for field in fields(graphs[0])})
    n, m, o, bodies = 5, 2, 3, 15
    root = torch.zeros(n, m, 3)
    root[..., 0] = torch.arange(n).float().unsqueeze(1) * 12
    root[:, 1, 1] = 2
    root[..., 2] = .94
    body_pos = root[:, :, None, :].expand(n, m, bodies, 3).clone()
    body_rot = torch.zeros(n, m, bodies, 4)
    body_rot[..., 3] = 1
    zero = torch.zeros(n, m, bodies, 3)
    body = torch.cat([body_pos, body_rot, zero, zero], -1)

    box = torch.zeros(n, o, 13)
    box[..., 0] = root[:, :1, 0] + torch.tensor([1., 2., 3.])
    box[..., 1] = torch.tensor([.5, 1., 1.5])
    box[..., 2] = torch.tensor([.20, .25, .30])
    box[..., 6] = 1
    size = torch.full((n, o, 3), .4)
    signs = torch.tensor([[x, y, z] for x in (-1., 1.)
                          for y in (-1., 1.) for z in (-1., 1.)])
    bps = signs[None, None] * size[:, :, None] / 2
    order = torch.tensor([[2, 0, 1], [1, 2, 0], [0, 2, 1], [2, 1, 0], [1, 0, 2]])
    goals = torch.zeros(n, m, 3)
    goals[..., 0] = root[..., 0] + torch.tensor([4., 5.])
    goals[..., 1] = torch.tensor([2., 3.])
    goals[..., 2] = .2
    task = SimpleNamespace(
        num_envs=n, num_agents=m, num_objects=o, device='cpu',
        relation_runtime=SimpleNamespace(graph=graph), _relation_cfg=BASE['relationReward'],
        _rigid_body_pos=body_pos, _rigid_body_rot=body_rot,
        _rigid_body_vel=zero, _rigid_body_ang_vel=zero,
        _kinematic_humanoid_rigid_body_states=body.clone(),
        _box_states=box, _box_bps=bps, _box_size=size,
        _tar_pos=goals, _char_h=.94,
        _logical_box_values=lambda values: values[torch.arange(n)[:, None], order])
    return task


def test_all_templates_route_to_correct_teacher_blocks_and_targets():
    task = _task_with_all_templates()
    teacher = Stage1UnifiedTeacher.__new__(Stage1UnifiedTeacher)
    teacher.env_cfg = {'localRootObsPolicy': False, 'rootHeightObsPolicy': False}
    teacher.counts = torch.zeros(5, dtype=torch.long)
    obs = teacher.observation(task)
    assert obs.shape == (10, 354)
    torch.testing.assert_close(teacher.counts, torch.tensor([6, 1, 1, 1, 1]))
    graph = task.relation_runtime.graph
    logical_box = task._logical_box_values(task._box_states)
    logical_size = task._logical_box_values(task._box_size)
    for scene, onehot in enumerate((2, 1, 3, 2, 2)):
        row = 2 * scene
        assert obs[row, 350:354].argmax().item() == onehot
        assert obs[row, 350:354].sum().item() == 1
        slot_env = torch.tensor([scene])
        slot_agent = torch.tensor([0])
        primary = agent_object_indices(graph, slot_env, slot_agent).item()
        box = logical_box[scene, primary]
        size = logical_size[scene, primary]
        top = box[2] + size[2] / 2
        if scene == 0:
            target = box[:3]
            local = obs[row, 223 + 58 + 39:223 + 100]
        elif scene == 1:
            target = box[:3].clone()
            target[2] = top + .12
            local = obs[row, 223 + 20:223 + 23]
        elif scene == 2:
            target = box[:3].clone()
            target[2] = top + .94
            local = obs[row, 223 + 100:223 + 103]
        elif scene == 3:
            goal = agent_goal_indices(graph, slot_env, slot_agent).item()
            target = task._tar_pos[scene, goal]
            local = obs[row, 223 + 58 + 39:223 + 100]
        else:
            top_edge = graph.edge_valid[scene] & (graph.edge_relation[scene] == ON_TOP)
            support = logical_box[scene, int(graph.edge_dst[scene, top_edge][0] - 2)]
            support_size = logical_size[scene, int(graph.edge_dst[scene, top_edge][0] - 2)]
            target = support[:3].clone()
            target[2] += (size[2] + support_size[2]) / 2
            local = obs[row, 223 + 58 + 39:223 + 100]
        torch.testing.assert_close(local, target - task._rigid_body_pos[scene, 0, 0])


def test_reset_body_state_is_used_for_teacher_observation():
    task = _task_with_all_templates()
    teacher = Stage1UnifiedTeacher.__new__(Stage1UnifiedTeacher)
    teacher.env_cfg = {'localRootObsPolicy': False, 'rootHeightObsPolicy': False}
    teacher.counts = torch.zeros(5, dtype=torch.long)
    before = teacher.observation(task)
    task._kinematic_humanoid_rigid_body_states[2, 0, :, 0] += .5
    after = teacher.observation(task, torch.tensor([2]))
    assert not torch.equal(before[4], after[4])
    torch.testing.assert_close(before[[0, 1, 2, 3, 6, 7, 8, 9]],
                               after[[0, 1, 2, 3, 6, 7, 8, 9]])


def test_role_swap_routes_nonzero_agent_to_all_teacher_tasks():
    task = _task_with_all_templates(role_swap=True)
    teacher = Stage1UnifiedTeacher.__new__(Stage1UnifiedTeacher)
    teacher.env_cfg = {'localRootObsPolicy': False, 'rootHeightObsPolicy': False}
    teacher.counts = torch.zeros(5, dtype=torch.long)
    obs = teacher.observation(task)
    assert obs.shape == (10, 354)
    torch.testing.assert_close(obs[1::2, 350:354].argmax(-1),
                               torch.tensor([2, 1, 3, 2, 2]))
    torch.testing.assert_close(obs[::2, 350:354].argmax(-1), torch.full((5,), 2))


def test_teacher_blocks_match_original_unified_observation():
    task = _task_with_all_templates()
    teacher = Stage1UnifiedTeacher.__new__(Stage1UnifiedTeacher)
    teacher.env_cfg = {'localRootObsPolicy': False, 'rootHeightObsPolicy': False}
    teacher.counts = torch.zeros(5, dtype=torch.long)
    obs = teacher.observation(task)
    graph = task.relation_runtime.graph
    boxes = task._logical_box_values(task._box_states)
    corners = task._logical_box_values(task._box_bps)
    sizes = task._logical_box_values(task._box_size)
    block_mask = torch.zeros(4, 127, dtype=torch.bool)
    for i, (start, end) in enumerate(zip((0, 20, 58, 100), (20, 58, 100, 127))):
        block_mask[i, start:end] = True
    for scene, task_id in enumerate((2, 1, 3, 2, 2)):
        slot_env, slot_agent = torch.tensor([scene]), torch.tensor([0])
        source = agent_object_indices(graph, slot_env, slot_agent).item()
        box, bps, size = boxes[scene, source], corners[scene, source], sizes[scene, source]
        target = box[:3].clone()
        if scene == 1:
            target[2] = box[2] + size[2] / 2 + .12
        elif scene == 2:
            target[2] = box[2] + size[2] / 2 + .94
        elif scene == 3:
            goal = agent_goal_indices(graph, slot_env, slot_agent).item()
            target = task._tar_pos[scene, goal]
        elif scene == 4:
            edge = graph.edge_valid[scene] & (graph.edge_relation[scene] == ON_TOP)
            support_id = int(graph.edge_dst[scene, edge][0] - 2)
            support, support_size = boxes[scene, support_id], sizes[scene, support_id]
            target = support[:3].clone()
            target[2] += (size[2] + support_size[2]) / 2
        root = task._kinematic_humanoid_rigid_body_states[scene, 0, 0][None]
        task_mask = torch.zeros(1, 4, dtype=torch.bool)
        task_mask[0, task_id] = True
        facing = torch.tensor([[1., 0., 0.]])
        reference = compute_location_observations(
            root, root[:, None, :3].expand(1, 10, 3),
            target[None], box[None], bps[None], facing,
            box[None], bps[None], target[None],
            box[None], bps[None], target[None],
            task_mask, block_mask, True)
        torch.testing.assert_close(obs[2 * scene, 223:350], reference[0], rtol=0, atol=1e-6)


def test_forward_kl_matches_distribution_and_freezes_teacher_labels():
    student_mu = torch.randn(6, 32, requires_grad=True)
    student_sigma = torch.full((6, 32), .055, requires_grad=True)
    teacher_mu = torch.randn(6, 32, requires_grad=True)
    teacher_sigma = torch.full((6, 32), .055, requires_grad=True)
    actual = gaussian_forward_kl(student_mu, student_sigma, teacher_mu, teacher_sigma)
    expected = kl_divergence(Independent(Normal(teacher_mu, teacher_sigma), 1),
                             Independent(Normal(student_mu, student_sigma), 1)).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert teacher_mu.grad is None and teacher_sigma.grad is None
    assert student_mu.grad.norm() > 0


def test_teacher_labels_follow_scene_shuffle_and_partial_reset():
    agent = SimpleNamespace(num_actors=3, num_agents=2)
    scenes = torch.arange(12).reshape(4, 3, 1)
    labels = scenes.unsqueeze(2) * 10 + torch.arange(2).view(1, 1, 2, 1)
    grouped = MAAgent._group_agent_tensor(agent, labels.reshape(4, 6, 1))
    dataset = AMPDataset(12, 4, False, False, 'cpu', 1)
    dataset.update_values_dict(dict(obs=MAAgent._flatten_scene_tensor(agent, scenes),
                                    teacher_mu=grouped))
    for i in range(3):
        batch = dataset[i]
        torch.testing.assert_close(batch['teacher_mu'],
                                   batch['obs'].unsqueeze(1) * 10 + torch.arange(2).view(1, 2, 1))

    agent = MAAgent.__new__(MAAgent)
    agent._teacher = object()
    agent.vec_env = SimpleNamespace(env=SimpleNamespace(task=SimpleNamespace(
        num_envs=4, num_agents=2, device='cpu')))
    with patch.object(CommonAgent, 'env_reset', return_value={'obs': torch.zeros(4, 1)}):
        agent.env_reset()
        agent.env_reset([])
        torch.testing.assert_close(agent._teacher_reset_ids, torch.arange(4))
        agent._teacher_reset_ids = None
        agent.env_reset(torch.tensor([[2], [6]]))
        torch.testing.assert_close(agent._teacher_reset_ids, torch.tensor([1, 3]))

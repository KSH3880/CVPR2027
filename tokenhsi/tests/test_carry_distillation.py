from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from copy import deepcopy

import isaacgym
import pytest
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
DISTILL = yaml.safe_load((CONFIG / 'approach_scenario_stage1_unified_size_rsi_task_embedding_carry_distill.yaml').read_text())['env']


def test_carry_config_geometry_rsi_network_and_checkpoint_isolation():
    from utils.unified_training import validate_unified_env, validate_typed_bias_config
    from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata, validate_relation_config
    original = yaml.safe_load((CONFIG / 'approach_scenario_stage1_unified_size_rsi_task_embedding.yaml').read_text())['env']
    train_dir = CONFIG.parent / 'train/rlg'
    train = yaml.safe_load((train_dir / 'amp_ma_carry_relation_unified_size_rsi_task_embedding_carry_distill.yaml').read_text())
    old_train = yaml.safe_load((train_dir / 'amp_ma_carry_relation_unified_size_rsi_task_embedding.yaml').read_text())
    expected = deepcopy(original)
    expected['relationReward']['stage1_variant'] += '_carry_distill'
    expected['relationGraph']['template_probabilities'] = dict(zip(
        original['relationGraph']['template_probabilities'], (0., 0., 0., .5, .5)))
    expected['skillDiscProb'] = [1/3, 0., 0., 0., 1/3, 1/6, 0., 1/6]
    assert DISTILL == expected
    assert train['params']['network'] == old_train['params']['network']
    validate_relation_config(DISTILL['relationReward'])
    validate_unified_env(DISTILL)
    validate_typed_bias_config(DISTILL, train)
    with pytest.raises(ValueError, match='differs'):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(original['relationReward'])},
                                  checkpoint_metadata(DISTILL['relationReward']))


def test_carry_size_sampler_has_only_independent_placement_tasks():
    from utils.size_rsi import sample_sizes
    from utils.task_role_spec import task_size_probabilities, task_graph_packet
    from utils.edge_scenario_spec import sample_size_conditioned_graph
    torch.manual_seed(42)
    sizes = sample_sizes(2048, 'cpu', 'holding_at')
    assert ((sizes[:, :2] >= .2) & (sizes[:, :2] <= .6)).all()
    probabilities = task_size_probabilities(sizes[:, :2], carry_only=True)
    torch.testing.assert_close(probabilities, torch.tensor([0., 0., 0., .5, .5]).expand(2048, 2, 5))
    graph = sample_size_conditioned_graph(probabilities, DISTILL['relationGraph'], 'random_scenario')
    tasks = task_graph_packet(graph, 2048).reshape(2048, 2, 5)[..., 1].long()
    assert set(map(tuple, tasks.tolist())) == {(2, 2), (2, 3), (3, 2), (3, 3)}
    assert .45 < (tasks == 2).float().mean() < .55
    with pytest.raises(ValueError, match='Carry-only'):
        task_size_probabilities(torch.tensor([[[.8, .8, .5]]]), carry_only=True)


def test_carry_amp_samples_real_carry_family_experts_only():
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    from utils.unified_training import SKILLS
    calls = []
    libs = {}
    for index, name in enumerate(SKILLS):
        libs[name] = SimpleNamespace(index=index,
            sample_motions=lambda n: torch.zeros(n, dtype=torch.long),
            sample_time=lambda motions, truncate_time: torch.zeros(len(motions)))
    def build(motions, times, lib):
        calls.extend([SKILLS[lib.index]] * len(motions))
        return torch.full((len(motions), 20), float(lib.index))
    task = SimpleNamespace(_carry_only=True, _size_aware_rsi=True, device='cpu',
        _skill=SKILLS, _motion_lib=libs, dt=1/30, _num_amp_obs_steps=10,
        get_num_amp_obs=lambda: 50, build_amp_obs_demo=build)
    obs = HumanoidMACarry._fetch_conditioned_amp_demo(task, 4096).reshape(4096, 10, 5)
    assert set(calls) == {'loco', 'omomo', 'pickUp', 'putDown'}
    torch.testing.assert_close(obs[..., -3:], torch.tensor([1., 0., 0.]).expand(4096, 10, 3))
    for name, probability in [('loco', 1/3), ('omomo', 1/3), ('pickUp', 1/6), ('putDown', 1/6)]:
        assert abs(calls.count(name)/len(calls) - probability) < .04


def test_two_agent_four_object_teacher_targets_and_student_gradient(monkeypatch):
    from utils.edge_scenario_spec import compose_canonical_graph
    from learning.multi_agent.distillation import gradient_report, teacher_digest
    import test_task_role_message as reference
    task = _task_with_all_templates()
    task.num_envs, task.num_objects = 4, 4
    task._relation_cfg = DISTILL['relationReward']
    for key, value in vars(task).copy().items():
        if isinstance(value, torch.Tensor):
            setattr(task, key, value[:4].clone())
    for name in ('_box_states', '_box_bps', '_box_size'):
        value = getattr(task, name)
        setattr(task, name, torch.cat((value, value[:, 2:3].clone()), 1))
    task._box_states[:, 3, 0] += 1
    task._box_states[..., 3:7] = torch.tensor([.2, 0., 0., .98])
    task._box_states[..., 3:7] /= task._box_states[..., 3:7].norm(dim=-1, keepdim=True)
    task._logical_box_values = lambda value: value[:, [3, 1, 0, 2]]
    graph = compose_canonical_graph(torch.tensor([[3, 3], [3, 4], [4, 3], [4, 4]]))
    # Exchange goal IDs while keeping each owner's source/support binding.
    graph = replace(graph, edge_dst=torch.where(graph.edge_dst >= 6, 13 - graph.edge_dst, graph.edge_dst))
    task.relation_runtime.graph = graph
    checkpoint = Path('/home/hwanhee/CVPR2027/TokenHSI/output/tokenhsi/ckpt_stage1.pth')
    if not checkpoint.exists():
        pytest.skip('Original unified teacher checkpoint is not installed')
    teacher = Stage1UnifiedTeacher(checkpoint, 'cpu')
    before = teacher_digest(teacher)
    obs = teacher.observation(task)
    torch.testing.assert_close(obs[:, -4:], torch.tensor([0., 0., 1., 0.]).expand(8, 4))
    boxes = task._logical_box_values(task._box_states)
    sizes = task._logical_box_values(task._box_size)
    for scene, templates in enumerate(((3, 3), (3, 4), (4, 3), (4, 4))):
        for owner, template in enumerate(templates):
            if template == 3:
                target = task._tar_pos[scene, 1-owner]
            else:
                target = boxes[scene, 2+owner, :3].clone()
                for obj in (owner, 2+owner):
                    q = boxes[scene, obj, 3:7]
                    target[2] += (2*q[0]*q[3]).abs()*sizes[scene, obj, 1]/2
                    target[2] += (1-2*q[0]**2).abs()*sizes[scene, obj, 2]/2
            torch.testing.assert_close(obs[2*scene+owner, 320:323],
                                       target-task._rigid_body_pos[scene, owner, 0])
    mu_t, sigma_t = teacher.distribution(obs)
    assert mu_t.shape == sigma_t.shape == (8, 32)
    assert torch.isfinite(mu_t).all() and torch.isfinite(sigma_t).all()
    assert not mu_t.requires_grad
    train = yaml.safe_load((CONFIG.parent / 'train/rlg/amp_ma_carry_relation_unified_size_rsi_task_embedding_carry_distill.yaml').read_text())
    monkeypatch.setattr(reference, 'CFG', DISTILL)
    monkeypatch.setattr(reference, 'TRAIN', train)
    student = reference.network()
    model = SimpleNamespace(a2c_network=student)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    initial = student.actor_encoder.edge_encoder.category_embed.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        mu, log_sigma = student.eval_actor(reference.observations(graph))
        mu = mu.reshape(8, 32)
        loss = .001 * gaussian_forward_kl(mu, log_sigma.exp().reshape(8, 32), mu_t, sigma_t)
        report = gradient_report(model, loss, mu.square().mean())
        assert report['kl_critic_grad_absent']
        assert report['edge_encoder'] > 0
        loss.backward()
        optimizer.step()
    updated = student.actor_encoder.edge_encoder.category_embed.weight.detach()
    assert ((initial[2:4]-updated[2:4]).abs().sum(-1) > 0).all()
    assert teacher_digest(teacher) == before


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

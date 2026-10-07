from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import isaacgym
import pytest
import torch
import yaml

import test_task_role_message as reference
from test_carry_distillation import _task_with_all_templates
from learning.multi_agent.stage1_unified_teacher import Stage1UnifiedTeacher
from learning.multi_agent.distillation import gaussian_forward_kl, gradient_report, teacher_digest
from utils.relation_task_spec import checkpoint_metadata, check_checkpoint_metadata, validate_relation_config
from utils.unified_training import validate_unified_env, validate_typed_bias_config, SKILLS
from utils.edge_scenario_spec import compose_canonical_graph, sample_size_conditioned_graph
from utils.task_role_spec import task_size_probabilities, task_graph_packet
from utils.size_rsi import sample_sizes

ROOT = Path(__file__).resolve().parents[1] / 'data/cfg'
NAME = 'approach_scenario_stage1_unified_size_rsi_task_embedding'
CFG = yaml.safe_load((ROOT / 'multi_agent' / (NAME + '_distill.yaml')).read_text())['env']
TRAIN = yaml.safe_load((ROOT / 'train/rlg/amp_ma_carry_relation_unified_size_rsi_task_embedding_distill.yaml').read_text())


def test_original_environment_and_network_preserved_with_isolated_checkpoint():
    original = yaml.safe_load((ROOT / 'multi_agent' / (NAME + '.yaml')).read_text())['env']
    expected = deepcopy(original)
    expected['relationReward']['stage1_variant'] += '_distill'
    assert CFG == expected
    base_train = yaml.safe_load((ROOT / 'train/rlg/amp_ma_carry_relation_unified_size_rsi_task_embedding.yaml').read_text())
    assert TRAIN['params']['network'] == base_train['params']['network']
    validate_relation_config(CFG['relationReward'])
    validate_unified_env(CFG)
    validate_typed_bias_config(CFG, TRAIN)
    for suffix in ('', '_carry_distill'):
        old = yaml.safe_load((ROOT / 'multi_agent' / (NAME + suffix + '.yaml')).read_text())['env']
        with pytest.raises(ValueError, match='differs'):
            check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(old['relationReward'])},
                                      checkpoint_metadata(CFG['relationReward']))


def test_four_task_size_sampling_and_amp_family_distribution():
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    torch.manual_seed(42)
    sizes = sample_sizes(2048, 'cpu')
    graph = sample_size_conditioned_graph(task_size_probabilities(sizes[:, :2]),
                                         CFG['relationGraph'], 'random_scenario')
    tasks = task_graph_packet(graph, 2048).reshape(2048, 2, 5)[..., 1].long()
    assert len(set(map(tuple, tasks.tolist()))) == 16
    frequencies = torch.bincount(tasks.flatten(), minlength=4)/4096
    torch.testing.assert_close(frequencies, torch.tensor([.1, .25, .325, .325]), atol=.03, rtol=0)
    libs = {name: SimpleNamespace(index=i,
        sample_motions=lambda n: torch.zeros(n, dtype=torch.long),
        sample_time=lambda ids, truncate_time: torch.zeros(len(ids))) for i, name in enumerate(SKILLS)}
    task = SimpleNamespace(_carry_only=False, _size_aware_rsi=True, device='cpu',
        _skill=SKILLS, _motion_lib=libs, dt=1/30, _num_amp_obs_steps=10,
        get_num_amp_obs=lambda: 50,
        build_amp_obs_demo=lambda ids, times, lib: torch.full((len(ids), 20), float(lib.index)))
    obs = HumanoidMACarry._fetch_conditioned_amp_demo(task, 4096).reshape(4096, 10, 5)
    family = obs[:, 0, -3:].argmax(-1)
    torch.testing.assert_close(torch.bincount(family, minlength=3)/4096,
                               torch.tensor([.65, .1, .25]), atol=.03, rtol=0)
    skill = obs[:, 0, 0].long()
    for f, allowed in enumerate(({0, 4, 5, 7}, {0, 1}, {0, 2, 3})):
        assert set(skill[family == f].tolist()) == allowed


def test_all_sixteen_pairs_teacher_routing_targets_and_embedding_gradients(monkeypatch):
    task = _task_with_all_templates()
    task.num_envs, task.num_objects = 16, 4
    for key, value in vars(task).copy().items():
        if isinstance(value, torch.Tensor):
            setattr(task, key, value[torch.arange(16) % 5].clone())
    for name in ('_box_states', '_box_bps', '_box_size'):
        value = getattr(task, name)
        setattr(task, name, torch.cat((value, value[:, 2:3].clone()), 1))
    task._box_states[:, 3, 0] += 1
    # A 90-degree box yaw checks that SIT facing follows the box's local +X.
    task._box_states[..., 3:7] = torch.tensor([0., 0., .5**.5, .5**.5])
    task._logical_box_values = lambda value: value[:, [3, 1, 0, 2]]
    task._relation_cfg = CFG['relationReward']
    templates = torch.cartesian_prod(torch.arange(1, 5), torch.arange(1, 5))
    graph = compose_canonical_graph(templates)
    graph = replace(graph, edge_dst=torch.where(graph.edge_dst >= 6, 13-graph.edge_dst, graph.edge_dst))
    task.relation_runtime.graph = graph
    checkpoint = Path(TRAIN['params']['config']['teacher_distillation']['checkpoint'])
    if not checkpoint.exists():
        pytest.skip('Original unified checkpoint is not installed')
    teacher = Stage1UnifiedTeacher(checkpoint, 'cpu')
    before = teacher_digest(teacher)
    obs = teacher.observation(task)
    mapping = torch.tensor([0, 1, 3, 2, 2])
    torch.testing.assert_close(obs[:, -4:].argmax(-1), mapping[templates.flatten()])
    torch.testing.assert_close(teacher.counts, torch.tensor([0, 8, 8, 8, 8]))
    torch.testing.assert_close(obs[templates.flatten() == 1, 270:272],
                               torch.tensor([0., 1.]).expand(8, 2), atol=1e-6, rtol=0)
    boxes, sizes = task._logical_box_values(task._box_states), task._logical_box_values(task._box_size)
    for scene in range(16):
        for owner in range(2):
            kind = int(templates[scene, owner])
            target = boxes[scene, owner, :3].clone()
            if kind in (1, 2):
                target[2] += sizes[scene, owner, 2]/2 + (.12 if kind == 1 else task._char_h)
                start = 243 if kind == 1 else 323
            elif kind == 3:
                target = task._tar_pos[scene, 1-owner]
                start = 320
            else:
                target = boxes[scene, owner+2, :3].clone()
                target[2] += (sizes[scene, owner, 2]+sizes[scene, owner+2, 2])/2
                start = 320
            torch.testing.assert_close(obs[2*scene+owner, start:start+3],
                                       target-task._rigid_body_pos[scene, owner, 0])
    mu_t, sigma_t = teacher.distribution(obs)
    assert torch.isfinite(mu_t).all() and torch.isfinite(sigma_t).all()
    monkeypatch.setattr(reference, 'CFG', CFG)
    monkeypatch.setattr(reference, 'TRAIN', TRAIN)
    net = reference.network()
    initial = net.actor_encoder.edge_encoder.category_embed.weight.detach().clone()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        mu, log_sigma = net.eval_actor(reference.observations(graph))
        loss = .001*gaussian_forward_kl(mu.reshape(32, 32), log_sigma.exp().reshape(32, 32), mu_t, sigma_t)
        report = gradient_report(SimpleNamespace(a2c_network=net), loss, mu.square().mean())
        assert report['kl_critic_grad_absent'] and report['edge_encoder'] > 0
        loss.backward()
        optimizer.step()
    assert ((initial-net.actor_encoder.edge_encoder.category_embed.weight).abs().sum(-1) > 0).all()
    assert teacher_digest(teacher) == before

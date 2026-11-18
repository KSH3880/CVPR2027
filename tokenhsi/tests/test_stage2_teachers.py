from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils import stage2_spec as spec, coupled_push_spec as coupled
from utils.experiments import native_config, validate_experiment
from utils.task_role_spec import task_graph_packet
from learning.multi_agent.human_coordination import stage2_human_relations
from learning.multi_agent.stage2_teachers import Stage2Teachers
from env.tasks.multi_agent.stage2_group_layout import translate_people, recenter_before
from test_stage2_push_common import _common_observation

ENV = native_config('stage2')['env']


@pytest.mark.parametrize('people', [5, 6, 8, 16])
@pytest.mark.parametrize('preset', ['random_scenario', 'mixed', 'before', 'coupled', 'all_solo', 'all_coupled'])
def test_sequential_groups_preserve_relation_and_physical_constraints(people, preset):
    torch.manual_seed(13)
    _, groups, predecessor, templates = spec.sample_episode(torch.full((128, people, 3), .4),
        ENV['box']['build']['taskSizeRanges'], ENV['relationGraph']['template_probabilities'], preset)
    graph = spec.compose_graph(templates, groups, predecessor, 2*people)
    mapping, primary, valid = spec.physical_mapping(templates, groups, predecessor, graph)
    packet = task_graph_packet(graph, len(templates))
    relations = stage2_human_relations(packet, groups)
    before = spec.before_members(predecessor)
    shared = (relations == 3).any(-1)
    assert not (before & shared).any()
    assert (coupled.group_sizes(groups) <= 4).all()
    assert torch.equal((relations == 2).sum((1, 2)), (predecessor >= 0).sum(-1))
    assert mapping.max() < coupled.physical_box_count(people, 2*people)
    assert mapping.shape == valid.shape == (128, 2*people)
    shared_physical = primary[:, :, None] == primary[:, None, :]
    assert shared_physical[relations == 3].all()
    if preset == 'mixed':
        assert before.any(-1).all() and shared.any(-1).all()
    if preset in ('coupled', 'all_coupled', 'all_solo'):
        assert not before.any()
    if preset in ('before', 'all_solo'):
        assert not shared.any()


def test_larger_scene_is_inference_only():
    env = deepcopy(ENV)
    env.update(numAgents=8, numObjects=16)
    env['stage2JointPush']['physical_boxes'] = coupled.physical_box_count(8, 16)
    validate_experiment('stage2', env, native_config('stage2', True), evaluation=True)
    with pytest.raises(ValueError, match='inference-only'):
        validate_experiment('stage2', env, native_config('stage2', True))


def test_group_translation_and_before_recentering_preserve_internal_geometry():
    task = SimpleNamespace(num_agents=6, num_envs=1, device='cpu', num_physical_objects=6)
    task._groups = torch.tensor([[0, 0, 2, 2, 2, 5]])
    task._before_predecessor = torch.tensor([[-1, 0, -1, -1, -1, -1]])
    task._humanoid_root_states = torch.randn(1, 6, 13)
    task._kinematic_humanoid_rigid_body_states = torch.randn(1, 6, 15, 13)
    task._env_origins = torch.zeros(1, 3)
    task._group_spawn_targets = torch.zeros(1, 6, 2)
    task._tar_pos = torch.randn(1, 6, 3)
    task._box_states = torch.randn(1, 6, 13)
    task._agent_box_assignment = torch.tensor([[0, 0, 2, 2, 2, 5]])
    ids = torch.tensor([0])
    roots = task._humanoid_root_states.clone()
    translate_people(task, ids)
    assert torch.equal(roots[..., 2:], task._humanoid_root_states[..., 2:])
    centers = torch.stack([task._humanoid_root_states[0, task._groups[0] == g, :2].mean(0)
                           for g in (0, 2, 5)])
    torch.testing.assert_close(torch.pdist(centers), torch.full((3,), 5.), atol=1e-5, rtol=0)
    task._humanoid_root_states[0, 1, :2] += 3.
    distance = task._humanoid_root_states[0, 0, :2]-task._box_states[0, 0, :2]
    recenter_before(task, ids)
    torch.testing.assert_close(task._humanoid_root_states[0, :2, :2].mean(0),
                               task._group_spawn_targets[0, 0], atol=1e-5, rtol=0)
    torch.testing.assert_close(task._humanoid_root_states[0, 0, :2]-task._box_states[0, 0, :2], distance)


def test_real_teachers_load_exact_weights_and_route_only_shared_push(tmp_path):
    torch.set_num_threads(1)
    task = SimpleNamespace(device='cpu', num_agents=4, num_objects=8,
        cfg={'args': SimpleNamespace(output_path=str(tmp_path))})
    for name, result in dict(get_obs_size=1272, get_num_amp_obs=1330, get_action_size=32,
            get_policy_obs_mode='clean_scene', get_scene_entity_sizes=[223, 30, 1],
            get_scene_kinematic_size=7, get_scene_arena_scale=16.,
            get_scene_normalized_entity_sizes=[223, 30, 0], get_relation_suffix_size=24).items():
        setattr(task, name, lambda value=result: value)
    paths = [Path('pth/stage2_teacher_enable_13500.pth'), Path('pth/stage2_teacher_coupled_25500.pth')]
    teachers = Stage2Teachers(task, *paths)
    for teacher, path in zip((teachers.before, teachers.coupled), paths):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        actual = {'a2c_network.'+k: v for k, v in teacher.a2c_network.state_dict().items()}
        assert actual.keys() == saved['model'].keys()
        assert all(torch.equal(v, saved['model'][k]) for k, v in actual.items())
        assert not any(p.requires_grad for p in teacher.parameters())
        assert all(torch.equal(v, saved['running_mean_std'][k]) for k, v in teacher.normalizer.state_dict().items())
    obs = _common_observation('before_sit')
    mask = torch.tensor([[False, False, True, True]])
    state = torch.random.get_rng_state()
    with torch.no_grad():
        expected = torch.where(mask.reshape(-1, 1), teachers.coupled.means(obs), teachers.before.means(obs))
        torch.random.set_rng_state(state)
        actual = teachers.means(obs, mask)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
    assert torch.isfinite(actual).all()
    obs = _common_observation('coupled')
    valid = torch.tensor([[True]*4+[False]*4])
    changed = obs.clone()
    changed[:, 892:1132].view(1, 8, 30)[:, 4:] += 17.
    changed[:, 1136:1248].view(1, 16, 7)[:, 8:12, :3] += 29.
    for teacher in (teachers.before, teachers.coupled):
        teacher.a2c_network.actor_encoder.relation_graph_spec['shuffle_token_order'] = False
        teacher.a2c_network.coordination.shuffle_human_order = False
    with torch.no_grad():
        expected = teachers.means(obs, mask, valid)
        actual = teachers.means(changed, mask, valid)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)

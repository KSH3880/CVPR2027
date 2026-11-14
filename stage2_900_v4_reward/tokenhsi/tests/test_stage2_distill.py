from copy import deepcopy

import pytest
import torch

from learning.multi_agent.distill_replay import DistillReplay
from utils import stage2_spec as spec, coupled_push_spec as coupled
from utils.before_object_alias import alias_layout
from utils.experiments import native_config, validate_experiment
from utils.stage2_distill_spec import teacher_probability
from utils.task_role_spec import task_graph_packet

ENV = native_config('stage2')['env']


@pytest.mark.parametrize('people', [4, 5, 8, 16])
@pytest.mark.parametrize('extra', [0, 1, 4])
@pytest.mark.parametrize('preset', ['random_scenario', 'mixed', 'before', 'coupled'])
def test_small_token_budget_includes_supports_and_aliases(people, extra, preset):
    torch.manual_seed(52)
    objects = people+extra
    _, groups, predecessor, tasks = spec.sample_episode(torch.full((64, people, 3), .4),
        ENV['box']['build']['taskSizeRanges'], ENV['relationGraph']['template_probabilities'],
        preset, num_objects=objects)
    graph = spec.compose_graph(tasks, groups, predecessor, objects)
    mapping, primary, slots = spec.physical_mapping(tasks, groups, predecessor, graph)
    packet = task_graph_packet(graph, len(tasks))
    dependent, target, alias, used = alias_layout(packet, people, objects)
    demand = used+dependent.sum(-1)
    assert (demand <= objects).all()
    assert torch.equal(demand, spec.required_tokens(groups, predecessor, tasks))
    assert torch.equal(demand, people+(tasks == 4).sum(-1))
    assert mapping.shape == slots.shape == (64, objects)
    assert mapping.max() < coupled.physical_box_count(people, objects)
    row, record = dependent.nonzero(as_tuple=True)
    assert torch.equal(mapping[row, alias[row, record]-people], mapping[row, target[row, record]-people])
    if not extra:
        assert not (tasks == 4).any()
    if preset == 'mixed':
        assert (predecessor >= 0).any(-1).all()
        assert ((coupled.group_sizes(groups) > 1) & (tasks == 5)).any(-1).all()


def test_full_budget_preserves_existing_scene_sampling_and_rng():
    sizes = torch.full((256, 4, 3), .4)
    arguments = (sizes, ENV['box']['build']['taskSizeRanges'], ENV['relationGraph']['template_probabilities'],
                 'random_scenario', (1., 1., 1.))
    torch.manual_seed(18)
    expected = spec._sample_episode(*arguments)
    rng = torch.random.get_rng_state()
    torch.manual_seed(18)
    actual = spec.sample_episode(*arguments, num_objects=8)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize('preset', ['holding_ontop', 'before_stack'])
def test_impossible_fixed_presets_report_token_requirement(preset):
    with pytest.raises(ValueError, match='requires at least'):
        spec.sample_episode(torch.full((8, 4, 3), .4), ENV['box']['build']['taskSizeRanges'],
            ENV['relationGraph']['template_probabilities'], preset, num_objects=4)


def test_mixed_minimum_budget_reports_incompatible_fixed_assets():
    with pytest.raises(ValueError, match='compatible enable leader'):
        spec.sample_episode(torch.full((8, 4, 3), .3), ENV['box']['build']['taskSizeRanges'],
            ENV['relationGraph']['template_probabilities'], 'mixed', num_objects=4)


def test_distillation_config_and_teacher_schedule():
    train = native_config('stage2_distill', True)
    env = native_config('stage2_distill')['env']
    assert env == ENV
    validate_experiment('stage2_distill', env, train)
    cfg = train['params']['config']['distillation']
    assert teacher_probability(1, cfg) == teacher_probability(20, cfg) == 1.
    assert 0 < teacher_probability(21, cfg) < 1.
    assert teacher_probability(200, cfg) == teacher_probability(300, cfg) == 0.
    direct = dict(cfg, teacher_warmup_iterations=0, teacher_decay_iterations=0)
    assert teacher_probability(1, direct) == 0.
    bad = deepcopy(train)
    bad['params']['config']['distillation']['rollout_steps'] = 0
    with pytest.raises(ValueError):
        validate_experiment('stage2_distill', env, bad)


def test_replay_wrap_and_resume_preserve_observation_label_pairs():
    replay = DistillReplay(5, 3, 2, 'cpu')
    for index in range(3):
        values = torch.arange(index*4, (index+1)*4).float()[:, None]
        replay.add(values.expand(-1, 3), values.expand(-1, 2)*7)
    assert replay.size == 5 and replay.cursor == 2
    restored = DistillReplay(5, 3, 2, 'cpu')
    restored.load_state_dict(replay.state_dict())
    assert restored.size == replay.size and restored.cursor == replay.cursor
    assert torch.equal(restored.obs, replay.obs) and torch.equal(restored.actions, replay.actions)
    obs, action = restored.sample(100)
    assert torch.equal(action[:, 0], obs[:, 0]*7)
    assert set(obs[:, 0].tolist()) <= set(range(7, 12))

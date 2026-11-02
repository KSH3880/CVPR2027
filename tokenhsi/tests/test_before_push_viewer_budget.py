from copy import deepcopy

import pytest
import torch

from utils.before_object_alias import alias_layout
from utils.before_push_spec import sample_graph
from utils.before_spec import checkpoint_contract
from utils.experiments import native_config
from utils.task_role_spec import task_graph_packet
from utils.unified_training import validate_unified_env


NAME = 'stage2_before_push_near50_pair50_source5_virtual_asym_goal5_goalroute'
ENV = native_config(NAME)['env']


@pytest.mark.parametrize('people,objects', [(1, 1), (3, 3), (6, 8), (8, 8), (8, 12)])
def test_small_viewer_budget_preserves_checkpoint_and_training_contract(people, objects):
    env = deepcopy(ENV)
    env.update(numAgents=people, numObjects=objects)
    validate_unified_env(env, evaluation=True)
    assert checkpoint_contract(env) == checkpoint_contract(ENV)
    with pytest.raises(ValueError, match='two for training'):
        validate_unified_env(env)


@pytest.mark.parametrize('people,objects', [(0, 0), (4, 3), (8, 7)])
def test_viewer_rejects_less_than_one_token_per_person(people, objects):
    env = deepcopy(ENV)
    env.update(numAgents=people, numObjects=objects)
    with pytest.raises(ValueError, match='one object token'):
        validate_unified_env(env, evaluation=True)


@pytest.mark.parametrize('people,objects,preset', [
    (1, 1, 'random_scenario'), (3, 3, 'random_scenario'),
    (6, 8, 'random_scenario'), (8, 8, 'random_scenario'),
    (8, 12, 'random_scenario'), (8, 12, 'all_solo'),
    (8, 8, 'push'), (8, 8, 'before_sit'),
    (8, 12, 'before_stack'), (8, 9, 'one_pair_stack'),
])
def test_sampled_tasks_fit_including_follower_aliases(people, objects, preset):
    torch.set_num_threads(1)
    torch.manual_seed(2)
    graph, predecessor, templates = sample_graph(torch.full((256, people, 3), .4),
        ENV['box']['build']['taskSizeRanges'], ENV['relationGraph']['template_probabilities'],
        preset, objects, pairing_probability=.5)
    packet = task_graph_packet(graph, len(templates))
    dependent, _, _, count = alias_layout(packet, people, 2*people)
    required = count+dependent.sum(-1)
    assert torch.equal(required, people+(templates == 4).sum(-1))
    assert (required <= objects).all()
    assert torch.equal(dependent, predecessor >= 0)
    rows, followers = (predecessor >= 0).nonzero(as_tuple=True)
    assert (templates[rows, predecessor[rows, followers]] == 3).all()
    assert not (templates[rows, followers] == 5).any()
    if preset == 'push':
        assert (templates == 5).all()
    if objects == people:
        assert not (templates == 4).any()


@pytest.mark.parametrize('preset,objects', [
    ('holding_ontop', 15), ('before_stack', 11), ('one_pair_stack', 8),
])
def test_impossible_fixed_presets_fail_without_resampling(preset, objects):
    with pytest.raises(ValueError, match='preset exceeds'):
        sample_graph(torch.full((1, 8, 3), .4), ENV['box']['build']['taskSizeRanges'],
            ENV['relationGraph']['template_probabilities'], preset, objects)

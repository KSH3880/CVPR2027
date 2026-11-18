from copy import deepcopy

import pytest
import torch

from utils.coupled_push_spec import (asset_bank, assign_regions, check_checkpoint,
    checkpoint_contract, group_sizes, physical_assignment, physical_box_count,
    region_states, sample_groups, sample_episode, spare_assignment, compose_graph)
from utils.experiments import native_config, validate_experiment


NAME = 'stage2_coupled_push_individual_goal_solo5'


def environment(people, objects=None):
    env = native_config(NAME)['env']
    objects = 2*people if objects is None else objects
    env.update(numAgents=people, numObjects=objects)
    env['stage2JointPush']['physical_boxes'] = physical_box_count(people, objects)
    return env


@pytest.mark.parametrize('people,physical', [(4, 24), (6, 38), (8, 52)])
def test_group_assets_and_regions_cover_each_physical_box(people, physical):
    torch.manual_seed(42)
    env = environment(people)
    sizes, mass = asset_bank(128, env, 'cpu')
    assert sizes.shape == (128, physical, 3)
    torch.testing.assert_close(mass[:,4*people:], sizes[:,4*people:].prod(-1)*3)
    groups = sample_groups(128, people, .5, 'cpu')
    assignment = physical_assignment(torch.full_like(groups, 5), groups)
    same = groups[:,:,None] == groups[:,None]
    assert torch.equal(assignment[:,:,None] == assignment[:,None], same)
    assert ((assignment >= 4*people) & (assignment < physical)).all()
    whole = sizes.gather(1, assignment[...,None].expand(-1,-1,3))
    states = torch.zeros(128, people, 13)
    states[...,6] = 1
    tokens, cells = region_states(states, whole, group_sizes(groups), assign_regions(groups))
    torch.testing.assert_close(cells[...,0]*group_sizes(groups), whole[...,0])
    centers = (tokens[:,None,:,:3]*same[...,None]).sum(2)/group_sizes(groups)[...,None]
    torch.testing.assert_close(centers, states[...,:3], atol=2e-6, rtol=0)
    shared = sample_groups(128, people, .5, 'cpu', 'all_coupled')
    shared_assignment = physical_assignment(torch.full_like(shared, 5), shared)
    assert (shared_assignment == physical-1).all()
    torch.testing.assert_close(sizes[:,-1,0], sizes[:,4*people,0]*people)
    torch.testing.assert_close(sizes[:,-1,1:], sizes[:,4*people,1:])
    torch.testing.assert_close(mass[:,-1], mass[:,4*people]*people)


@pytest.mark.parametrize('people,objects', [(1,1),(3,3),(5,7),(7,14),(9,20),(12,12)])
def test_inference_reuses_weights_and_resume_keeps_entity_counts(people, objects):
    env = environment(people, objects)
    original = native_config(NAME)['env']
    checkpoint = {'coupled_push_contract': checkpoint_contract(original)}
    validate_experiment(NAME, env, native_config(NAME, True), evaluation=True)
    check_checkpoint(checkpoint, env, inference=True)
    assert checkpoint_contract(env) != checkpoint_contract(original)
    validate_experiment(NAME, env, native_config(NAME, True))
    with pytest.raises(ValueError, match='checkpoint settings differ'):
        check_checkpoint(checkpoint, env)
    custom = {'coupled_push_contract': checkpoint_contract(env)}
    check_checkpoint(custom, env)
    check_checkpoint(custom, original, inference=True)
    modified = deepcopy(env)
    modified['stage2JointPush']['goal_distance'] = [2., 10.]
    with pytest.raises(ValueError, match='checkpoint settings differ'):
        check_checkpoint(checkpoint, modified, inference=True)
    modified = deepcopy(env)
    modified['stage2JointPush']['physical_boxes'] = 24
    with pytest.raises(ValueError, match='physical bank'):
        validate_experiment(NAME, modified, native_config(NAME, True), evaluation=True)


@pytest.mark.parametrize('people', [6, 8])
def test_larger_groups_are_included_in_spatial_scores(people):
    from utils.evaluation_protocol import coupled_scores, shared_center_scores
    groups = torch.zeros(2, people, dtype=torch.long)
    individual = torch.ones(2, people, 2, dtype=torch.bool)
    simultaneous = individual.clone()
    simultaneous[1] = False
    result = coupled_scores(groups, individual, individual, simultaneous, simultaneous)
    row = result['point'][str(people)]
    assert row['groups'] == 2 and row['people'] == 2*people
    assert row['simultaneous_ever'] == 1 and row['individual_ever'] == 2*people
    centers = shared_center_scores(groups, simultaneous[...,0], simultaneous[...,0])
    assert centers[str(people)]['groups'] == 2
    assert centers[str(people)]['ever_rate'] == .5


@pytest.mark.parametrize('people,objects', [(1,1),(3,3),(5,6),(7,10),(9,20)])
@pytest.mark.parametrize('preset', ['random_scenario','all_solo','all_coupled'])
def test_sampling_respects_available_supports_and_object_binding(people, objects, preset):
    env = environment(people, objects)
    weights = torch.tensor(list(env['relationGraph']['template_probabilities'].values()))
    groups, tasks = sample_episode(128, people, objects, .5, weights, 'cpu', preset)
    assert ((tasks == 4).sum(-1) <= objects-people).all()
    if preset == 'all_solo':
        assert group_sizes(groups).eq(1).all()
    if preset == 'all_coupled':
        assert group_sizes(groups).eq(people).all() and tasks.eq(5).all()
    graph = compose_graph(tasks, groups, False, individual_goals=True, num_objects=objects)
    assert graph.num_objects == objects
    mapping = torch.cat((physical_assignment(tasks,groups),spare_assignment(tasks,objects)),-1)
    assert mapping.shape == (128, objects)
    onto = (graph.edge_relation == 8) & graph.edge_valid
    rows, edges = onto.nonzero(as_tuple=True)
    owners = graph.edge_owner[rows,edges]
    support = graph.edge_dst[rows,edges]-people
    assert ((support >= people) & (support < objects)).all()
    assert torch.equal(mapping[rows,support],3*people+owners)
    at = (graph.edge_relation == 7) & graph.edge_valid
    assert torch.equal(graph.edge_dst[at], (people+objects+graph.edge_owner)[at])
    sizes, _ = asset_bank(128, env, 'cpu')
    assert sizes.shape[1] == physical_box_count(people, objects)
    assert (mapping < sizes.shape[1]).all()


def test_impossible_forced_ontop_does_not_retry_forever():
    with pytest.raises(ValueError, match='distinct spare support'):
        sample_episode(1, 3, 3, .5, torch.ones(6)/6, 'cpu', 'holding_ontop')


def test_only_scenes_with_insufficient_objects_are_redrawn(monkeypatch):
    from utils import coupled_push_spec as spec
    draws = [torch.tensor([[4,4,4],[3,3,3]]), torch.tensor([[4,3,3]])]
    sizes = []

    def groups(count, people, *args):
        sizes.append(count)
        return torch.arange(people)[None].expand(count,-1)

    monkeypatch.setattr(spec, 'sample_groups', groups)
    monkeypatch.setattr(spec, 'sample_templates', lambda *args: draws.pop(0))
    _, tasks = sample_episode(2, 3, 4, .5, torch.ones(6)/6, 'cpu', 'random_scenario')
    assert sizes == [2,1]
    assert tasks.tolist() == [[4,3,3],[3,3,3]]


def test_sufficient_objects_preserve_sampling_and_rng():
    from utils.coupled_push_spec import sample_templates
    env = environment(4)
    weights = torch.tensor(list(env['relationGraph']['template_probabilities'].values()))
    torch.manual_seed(42)
    expected_groups = sample_groups(128, 4, .5, 'cpu')
    expected_tasks = sample_templates(expected_groups, weights)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(42)
    groups, tasks = sample_episode(128, 4, 8, .5, weights, 'cpu', 'random_scenario')
    assert torch.equal(groups, expected_groups) and torch.equal(tasks, expected_tasks)
    assert torch.equal(torch.get_rng_state(), expected_rng)

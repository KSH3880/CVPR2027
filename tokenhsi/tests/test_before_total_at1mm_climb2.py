from copy import deepcopy

import pytest
import torch

from env.tasks.multi_agent.before_wait import BeforeWait
from test_before_wait import setup
from utils.before_push_spec import sample_graph
from utils.before_spec import check_checkpoint, checkpoint_contract
from utils.edge_ontop_spec import permute_graph
from utils.experiments import experiment_horizon, native_config, validate_experiment
from utils.unified_training import validate_unified_env

NAME = 'stage2_before_total_at1mm_climb2'
ENV = native_config(NAME)['env']
OPTION = ENV['before']['waiting_penalty']


def test_only_climb_buffer_and_names_change():
    cfg = native_config(NAME)
    base = native_config('stage2_before_total_at1mm')
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    assert experiment_horizon(NAME) == cfg['env']['episodeLength'] == 900
    assert cfg['env']['before']['waiting_penalty'].pop('climb_radius_buffer') == 2.
    cfg['experiment']['name'] = base['experiment']['name']
    assert cfg == base
    train = native_config(NAME, True)
    train['params']['config']['name'] = 'Stage2BeforeTotalAt1mm'
    assert train == native_config('stage2_before_total_at1mm', True)
    with pytest.raises(ValueError, match='checkpoint contract'):
        check_checkpoint({'before_contract': checkpoint_contract(base['env'])}, ENV)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_only_climb_arms_in_added_half_meter_and_release_stays_ever(family):
    state, args, graph, rows, edges = setup(family, config=OPTION)
    roots, velocity, boxes, sizes, goals, raw = args
    old = BeforeWait(graph, len(roots), .1,
        native_config('stage2_before_total_at1mm')['env']['before']['waiting_penalty'], 'cpu')
    old.reset(torch.arange(len(roots)), graph, raw)
    entity = boxes if family == 'stack' else roots
    index = state.source-4 if family == 'stack' else state.owners
    center = goals[rows, state.goal, :2].clone()
    rb = sizes[rows, state.support, :2].norm(dim=-1)/2
    entity[rows, index, :2] = center
    entity[rows, index, 0] += rb+1.75
    assert not old.penalty(*args).any()
    result = state.penalty(*args)
    assert bool(result.any()) == (family == 'climb')
    state.reset(torch.arange(len(roots)), graph, raw)
    radius = rb+(2. if family == 'climb' else 1.5)
    entity[rows, index, 0] = center[:, 0]+radius+.001
    assert not state.penalty(*args).any()
    entity[rows, index, 0] -= .002
    velocity[:] = torch.tensor([0., 0., .7])
    expected = torch.zeros(roots.shape[:2])
    expected[rows, state.owners] = .7
    torch.testing.assert_close(state.penalty(*args), expected)
    entity[rows, index, 0] += 100
    torch.testing.assert_close(state.penalty(*args), expected)
    raw[rows, state.predecessor] = True
    assert not state.penalty(*args).any()
    raw.zero_()
    assert not state.penalty(*args).any()
    state.reset(torch.tensor([1]), graph, raw)
    entity[rows, index, :2] = center
    expected.zero_();expected[1, state.owners[rows == 1]] = .7
    torch.testing.assert_close(state.penalty(*args), expected)


@pytest.mark.parametrize('preset', ['before_sit', 'before_climb', 'before_stack', 'all_solo', 'push'])
def test_shuffled_multiple_pairs_and_independent_tasks(preset):
    torch.manual_seed(41)
    graph, _, _ = sample_graph(torch.full((3, 4, 3), .4), ENV['box']['build']['taskSizeRanges'],
        ENV['relationGraph']['template_probabilities'], preset)
    graph = permute_graph(graph, torch.rand(3, 8).argsort(-1))
    state = BeforeWait(graph, 3, .1, OPTION, 'cpu')
    raw = torch.zeros_like(graph.edge_valid)
    state.reset(torch.arange(3), graph, raw)
    rows = state.rows
    roots, velocity = torch.zeros(3, 4, 3), torch.ones(3, 4, 3)
    boxes, sizes, goals = torch.zeros(3, 8, 13), torch.full((3, 8, 3), .4), torch.zeros(3, 4, 3)
    goals[..., 0] = torch.arange(4)*10.
    entity = boxes if preset=='before_stack' else roots
    index = state.source-4 if preset=='before_stack' else state.owners
    entity[rows, index, :2] = goals[rows, state.goal, :2]
    entity[rows, index, 0] += sizes[rows, state.support, :2].norm(dim=-1)/2+1.75
    raw[rows[::2], state.predecessor[::2]] = True
    expected = torch.zeros(3, 4)
    if preset=='before_climb':
        expected[rows[1::2], state.owners[1::2]] = 3**.5
    torch.testing.assert_close(state.penalty(roots, velocity, boxes, sizes, goals, raw), expected)


@pytest.mark.parametrize('value', [0., -1., True, None, '2', float('nan'), float('inf')])
def test_invalid_climb_override_rejected(value):
    env = deepcopy(ENV)
    env['before']['waiting_penalty']['climb_radius_buffer'] = value
    with pytest.raises(ValueError, match='waiting penalty'):
        validate_unified_env(env)


def test_resume_preserves_training_state_and_rejects_wrong_source(tmp_path):
    from utils.before_total_at1mm_climb2_resume import prepare
    import yaml

    base = native_config('stage2_before_total')
    base['train'] = native_config('stage2_before_total', True)
    original = dict(epoch=12000, frame=1572864000,
        model={'actor': torch.tensor([1., 2.])}, optimizer={'state': torch.tensor([3.])},
        running_mean_std={'mean': torch.tensor([4.])}, env_state={'state': torch.tensor([5.])},
        before_contract=checkpoint_contract(base['env']), relation_experiment_config=base)
    source = tmp_path/'source.pth'
    torch.save(original, source)
    before = source.read_bytes()
    path = prepare(source, tmp_path/'converted')
    actual = torch.load(path, map_location='cpu', weights_only=False)
    assert path.name == 'initial_00012000.pth'
    for key in ('model', 'optimizer', 'running_mean_std', 'env_state'):
        for k, v in original[key].items():
            assert torch.equal(v, actual[key][k])
    assert actual['epoch'] == original['epoch'] and actual['frame'] == original['frame']
    assert actual['relation_experiment_config']['env'] == ENV
    sidecar = yaml.safe_load((path.parent/'relation_config.yaml').read_text())
    assert sidecar == actual['relation_experiment_config']
    assert source.read_bytes() == before
    assert prepare(source, tmp_path/'converted') == path
    assert prepare(path, tmp_path/'unused') == path
    original['epoch'] = 12500
    torch.save(original, source)
    with pytest.raises(ValueError, match='epoch 12000'):
        prepare(source, tmp_path/'bad')

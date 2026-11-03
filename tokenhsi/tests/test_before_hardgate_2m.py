from copy import deepcopy

import pytest
import torch

from env.tasks.multi_agent.before_reward import predecessor_at_progress
from utils.before_spec import checkpoint_contract, check_checkpoint
from utils.experiments import native_config, validate_experiment
from utils.unified_training import validate_unified_env
from utils.edge_ontop_spec import select_graph, permute_graph
from test_before_asymmetric_progress import geometry

BASE = 'stage2_before_push_near50_pair50_source5_virtual_goal5_goalroute_hardgate'
NAME = BASE + '_2m'


def test_config_separates_only_gate_radius():
    cfg = native_config(NAME)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], native_config(NAME, True))
    old = native_config(BASE)
    assert cfg['env']['before'].pop('hard_gate_radius') == 2.
    assert cfg['env'] == old['env']
    cfg = native_config(NAME)
    with pytest.raises(ValueError, match='checkpoint contract'):
        check_checkpoint({'before_contract': checkpoint_contract(old['env'])}, cfg['env'])


@pytest.mark.parametrize('value', [None, False, 0., -1., float('inf'), float('nan'), '2', {}])
def test_invalid_gate_radius(value):
    env = deepcopy(native_config(NAME)['env'])
    env['before']['hard_gate_radius'] = value
    with pytest.raises(ValueError, match='Hard gate radius'):
        validate_unified_env(env)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
@pytest.mark.parametrize('size_scale', [.5, 1., 2.])
def test_two_meter_boundary_and_reopening(family, size_scale):
    args, rows, edges, _ = geometry(family, [0., 0., 0., 0.])
    roots, boxes, sizes, goals, graph, config = args
    sizes.mul_(size_scale)
    distance = torch.tensor([2.0001, 2., 1.9999, 0.])
    source = graph.edge_src[rows, edges]
    if family == 'stack':
        boxes[rows, source-graph.num_agents, 0] = distance
    else:
        roots[rows, source, 0] = distance
    original = torch.rand(4, len(graph.ids))
    ordinary = predecessor_at_progress(original, *args)
    predecessor = graph.prereq_mask[rows, edges].long().argmax(-1)
    raw = torch.zeros_like(original, dtype=torch.bool)
    for success in [False, True, False]:
        raw[rows, predecessor] = success
        result = predecessor_at_progress(original, *args, raw_success=raw,
            hard_gate=True, hard_gate_radius=2.)
        expected = ordinary.clone()
        if not success:
            expected[rows[1:], edges[1:]] = 0.
        assert torch.equal(result, expected)
        # Reopening uses the original size-dependent proximity reward.
        ids = torch.tensor([3, 1])
        partial = predecessor_at_progress(original[ids], roots[ids], boxes[ids], sizes[ids],
            goals[ids], select_graph(graph, ids), config, raw_success=raw[ids],
            hard_gate=True, hard_gate_radius=2.)
        assert torch.equal(partial, result[ids])
        order = torch.rand_like(original).argsort(-1)
        shuffled = predecessor_at_progress(original.gather(1, order), roots, boxes, sizes,
            goals, permute_graph(graph, order), config, raw_success=raw.gather(1, order),
            hard_gate=True, hard_gate_radius=2.)
        assert torch.equal(shuffled, result.gather(1, order))

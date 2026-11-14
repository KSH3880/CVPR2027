from copy import deepcopy

import pytest
import torch

from env.tasks.multi_agent.before_push_reward import BeforePushRuntime
from test_before_wait import setup
from utils.before_spec import check_checkpoint, checkpoint_contract
from utils.edge_context_spec import AT, HOLDING
from utils.edge_ontop_spec import permute_graph
from utils.experiments import experiment_horizon, native_config, validate_experiment
from utils.stage1_jointpush import compose_graph
from utils.unified_training import validate_unified_env

NAME = 'stage2_before_total_at1mm'
ENV = native_config(NAME)['env']


def runtime(graph, strict=True):
    return BeforePushRuntime(len(graph.edge_valid), graph, ENV['relationReward'], 'cpu',
        contact_weights=ENV['jointPush']['contact_weights'],
        reward_at_z_tolerance=.001 if strict else None)


def test_new_training_changes_only_reward_height_and_names():
    cfg, base = native_config(NAME), native_config('stage2_before_total')
    train = native_config(NAME, True)
    validate_unified_env(cfg['env'])
    validate_experiment(NAME, cfg['env'], deepcopy(train))
    assert experiment_horizon(NAME) == cfg['env']['episodeLength'] == 900
    assert not train['params']['load_checkpoint']
    assert cfg['experiment']['transfer']['checkpoint'].endswith('_00026000.pth')
    assert cfg['env']['before'].pop('reward_at_z_tolerance') == .001
    cfg['experiment']['name'] = base['experiment']['name']
    assert cfg == base
    train['params']['config']['name'] = 'Stage2BeforeTotal'
    assert train == native_config('stage2_before_total', True)
    for source, target in ((ENV, base['env']), (base['env'], ENV)):
        with pytest.raises(ValueError, match='checkpoint contract'):
            check_checkpoint({'before_contract': checkpoint_contract(source)}, target)


@pytest.mark.parametrize('value', [None, True, -.001, .2, float('nan')])
def test_invalid_reward_height_rejected(value):
    env = deepcopy(ENV)
    env['before']['reward_at_z_tolerance'] = value
    with pytest.raises(ValueError, match='AT reward height'):
        validate_unified_env(env)


@pytest.mark.parametrize('family', ['sit', 'climb', 'stack'])
def test_reward_height_does_not_change_current_gate_or_ever_release(family):
    state, inputs, graph, rows, edges = setup(family, dt=1/30,
        config=ENV['before']['waiting_penalty'])
    phi = torch.full(graph.edge_relation.shape, .8)
    phi[graph.edge_relation == HOLDING] = .1
    progress = torch.full_like(phi, .4)
    zero = torch.zeros_like(phi)
    z = zero.clone()
    z[rows, state.predecessor] = torch.tensor([.0009, .0011, -.0011, .1])
    old = runtime(graph, False).components(phi, progress, z, zero, zero, graph)
    new = runtime(graph).components(phi, progress, z, zero, zero, graph)
    for key in ('raw_own_success', 'own_success', 'prerequisite_gate'):
        assert torch.equal(old[key], new[key])
    assert new['prerequisite_gate'][rows, edges].all()
    torch.testing.assert_close(new['total'][rows, edges], old['total'][rows, edges])
    torch.testing.assert_close(new['success_component'][rows, state.predecessor],
        torch.tensor([.2, 0., 0., 0.]))
    torch.testing.assert_close(new['state_component'][rows, state.predecessor],
        torch.tensor([.2, .16, .16, .16]))
    torch.testing.assert_close(new['progress_component'][rows, state.predecessor],
        torch.tensor([.2, .08, .08, .08]))
    allowed = graph.edge_relation == AT
    allowed |= (graph.edge_relation == HOLDING) & (
        graph.edge_owner == graph.edge_owner[rows, state.predecessor, None])
    assert torch.equal(new['total'][~allowed], old['total'][~allowed])
    assert state.penalty(*inputs).any()
    inputs[-1].copy_(new['raw_own_success'])
    assert not state.penalty(*inputs).any()
    phi[graph.edge_relation == AT] = .1
    off = runtime(graph).components(phi, progress, z, zero, zero, graph)
    assert not off['prerequisite_gate'][rows, edges].any()
    inputs[-1].copy_(off['raw_own_success'])
    assert not state.penalty(*inputs).any()
    state.reset(torch.tensor([1]), graph, off['raw_own_success'])
    assert state.penalty(*inputs)[1].any()
    order = torch.rand_like(phi).argsort(-1)
    shuffled = permute_graph(graph, order)
    actual = runtime(shuffled).components(phi.gather(1, order), progress.gather(1, order),
        z.gather(1, order), zero, zero, shuffled)
    for key in new:
        if key in ('local_task_reward', 'agent_task_reward', 'teammate_task_reward'):
            torch.testing.assert_close(actual[key], off[key])
        else:
            torch.testing.assert_close(actual[key], off[key].gather(1, order))


@pytest.mark.parametrize('height,passes', [(0., True), (.000999, True),
    (.001001, False), (-.001001, False), (.05, False)])
def test_solo_at_and_push_placement_saturation(height, passes):
    graph = compose_graph(torch.tensor([[3, 5, 1, 4]]), num_objects=8)
    rt = runtime(graph)
    phi = torch.full(graph.edge_relation.shape, .8)
    holding = graph.edge_relation == HOLDING
    phi[holding] = .1
    progress = torch.full_like(phi, .4)
    z = torch.zeros_like(phi)
    at = graph.edge_relation == AT
    z[at] = height
    rt.contact.zero_()
    result = rt.components(phi, progress, z, torch.zeros_like(z), torch.zeros_like(z), graph)
    assert result['raw_own_success'][at].all()
    assert (result['success_component'][at] == (.2 if passes else 0.)).all()
    connected = (holding | (graph.edge_relation == 12)) & (graph.edge_owner < 2)
    assert (result['paired_placement_success'][connected] == passes).all()
    assert (result['reward_saturated'][at] == passes).all()
    if not passes:
        rt.contact[:] = True
        own_contact = rt.components(phi, progress, z, torch.zeros_like(z), torch.zeros_like(z), graph)
        assert own_contact['reward_saturated'][graph.edge_relation == 12].all()

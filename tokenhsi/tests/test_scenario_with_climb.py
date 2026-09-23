from pathlib import Path

import pytest
import torch
import yaml

from env.tasks.multi_agent.edge_stage1_reward import Stage1ContextRuntime
from utils.edge_context_spec import HOLDING, AT
from utils.edge_interaction_spec import SIT, CLIMB
from utils.edge_ontop_spec import ON_TOP
from utils.edge_scenario_spec import (CLIMB_TEMPLATES, classify_templates,
    compose_graph, sample_graph, valid_binding_pair)
from utils.edge_stage1_spec import validate_sampler
from utils.relation_task_spec import (checkpoint_metadata,
    check_checkpoint_metadata, validate_relation_config)


ROOT = Path(__file__).resolve().parents[1]
ENV = yaml.safe_load(
    (ROOT / 'data/cfg/multi_agent/approach_scenario_with_climb.yaml').read_text())['env']


def test_config_climb_reward_rsi_amp_and_checkpoint_contract():
    validate_relation_config(ENV['relationReward'])
    validate_sampler(ENV['relationGraph'])
    assert ENV['relationReward']['schema_version'] == 9
    assert tuple(ENV['relationGraph']['template_probabilities']) == CLIMB_TEMPLATES
    assert ENV['templateRsi']['CLIMB'] == [.5, 0, .5, 0, 0, 0, 0, 0]
    assert ENV['skillDiscProb'] == pytest.approx([.26, .2, .08, .06, .1, .1, .1, .1])
    assert sum(ENV['skillDiscProb']) == pytest.approx(1.)
    climb = ENV['relationReward']['climb']
    assert climb['success_phi_threshold'] == .6
    assert climb['feet_height_tolerance'] == .07
    assert ENV['relationReward']['progress']['climb_pinning'] == 'bbox_valid_radius'
    assert ENV['box']['reset']['groundStandaloneSitClimb'] is True
    old = yaml.safe_load(
        (ROOT / 'data/cfg/multi_agent/approach_scenario_no_climb.yaml').read_text())['env']
    with pytest.raises(ValueError, match='schema mismatch'):
        check_checkpoint_metadata(
            {'relation_metadata': checkpoint_metadata(old['relationReward'])},
            checkpoint_metadata(ENV['relationReward']))


def test_sampler_has_five_templates_and_random_bindings():
    graph = sample_graph(8192, ENV['relationGraph'],
        generator=torch.Generator().manual_seed(17))
    assert set(graph.edge_relation[graph.edge_valid].tolist()) == {
        HOLDING, AT, ON_TOP, SIT, CLIMB}
    slots = torch.arange(8192).repeat_interleave(2)
    agents = torch.arange(2).repeat(8192)
    template = classify_templates(graph, slots, agents, with_climb=True)
    assert set(template.tolist()) == set(range(5))
    frequencies = torch.bincount(template, minlength=5).float() / len(template)
    assert torch.all((frequencies - .2).abs() < .02)


def test_climb_invalid_and_shared_object_rules():
    assert not valid_binding_pair(('CLIMB', 'CLIMB'), ((0,), (0,)))
    assert not valid_binding_pair(('SIT', 'CLIMB'), ((0,), (0,)))
    assert not valid_binding_pair(
        ('CLIMB', 'HOLDING_ON_TOP'), ((0,), (1, 0)))
    assert valid_binding_pair(('HOLDING_AT', 'CLIMB'), ((0, 0), (0,)))
    assert valid_binding_pair(
        ('HOLDING_ON_TOP', 'CLIMB'), ((0, 1), (0,)))


def test_climb_success_saturates_only_its_edge():
    graph = compose_graph([('CLIMB', 'HOLDING')], [((0,), (1,))])
    climb = ((graph.edge_relation == CLIMB) & graph.edge_valid).nonzero()[0, 1]
    holding = ((graph.edge_relation == HOLDING) & graph.edge_valid).nonzero()[0, 1]
    phi = torch.zeros(1, 4)
    progress = torch.full((1, 4), .3)
    feet_error = torch.ones(1, 4)
    phi[0, climb] = .6
    feet_error[0, climb] = .07
    runtime = Stage1ContextRuntime(1, graph, ENV['relationReward'], 'cpu')
    out = runtime.step(phi, progress, torch.zeros_like(phi), feet_error)
    assert out['own_success'][0, climb]
    assert out['total'][0, climb] == pytest.approx(.6)
    assert not out['reward_saturated'][0, holding]

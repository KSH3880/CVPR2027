from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges
from env.tasks.multi_agent.size_rsi_cache import SizeRsiCache
from env.tasks.multi_agent.edge_ontop_task import SampledOnTopTaskMixin
from utils.edge_scenario_spec import (classify_templates, agent_object_indices,
    compose_graph, compose_canonical_graph, sample_size_conditioned_graph, validate_graph)
from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata
from utils.size_rsi import (TASKS, TASK_PROBS, SIZE_RANGES, sample_sizes, sample_size_graph,
                            task_probabilities, resolve_skills, screen_states, size_key)
from utils.unified_training import validate_unified_env, family_from_templates

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi.yaml').read_text())['env']


def test_config_contract_and_checkpoint_isolation():
    validate_relation_config(CFG['relationReward'])
    validate_unified_env(CFG)
    base = yaml.safe_load((ROOT / 'data/cfg/multi_agent/approach_scenario_stage1_unified.yaml').read_text())['env']
    with pytest.raises(ValueError):
        check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(base['relationReward'])},
                                  checkpoint_metadata(CFG['relationReward']))
    bad = deepcopy(CFG)
    bad['templateRsi']['HOLDING_ON_TOP'] = [.4, 0., 0., 0., 0., .1, .5, 0.]
    with pytest.raises(ValueError, match='40/10/40/10'):
        validate_unified_env(bad)
    bad = deepcopy(CFG)
    bad['box']['reset']['randomAssignment'] = True
    with pytest.raises(ValueError, match='asset / RSI contract'):
        validate_unified_env(bad)


def test_size_conditioned_tasks_keep_distribution_and_owner_bindings():
    torch.manual_seed(172)
    sizes = sample_sizes(12000, 'cpu')
    weights = task_probabilities(sizes[:, :2])
    assert torch.allclose(weights.mean((0, 1)), torch.tensor(TASK_PROBS), atol=.008)
    graph = sample_size_graph(sizes[:300], CFG['relationGraph'], 'random_scenario')
    rows = torch.arange(300).repeat_interleave(2)
    owners = torch.arange(2).repeat(300)
    tasks = classify_templates(graph, rows, owners, True)
    assert torch.equal(agent_object_indices(graph, rows, owners), owners)
    assert (weights[rows, owners, tasks] > 0).all()
    assert torch.equal(family_from_templates(tasks), torch.where(tasks == 1, 1,
                       torch.where(tasks == 2, 2, 0)))
    for preset, task in (('sit', 'SIT'), ('climb', 'CLIMB'), ('holding_ontop', 'HOLDING_ON_TOP')):
        values = sample_sizes(128, 'cpu', preset)
        bounds = torch.tensor(SIZE_RANGES[task])
        assert ((values[:, :2] >= bounds[:, 0]-1e-6) &
                (values[:, :2] <= bounds[:, 1]+1e-6)).all()
        assert ((values*20 - (values*20).round()).abs() < 1e-5).all()
        assert (values[:, :2, 0] != values[:, :2, 1]).any()


def test_batched_canonical_graph_matches_all_task_pairs():
    ids = torch.cartesian_prod(torch.arange(5), torch.arange(5))
    templates = [[TASKS[i] for i in pair] for pair in ids.tolist()]
    bindings = [[(a, a) if task == 'HOLDING_AT' else (a, a+2)
                 if task == 'HOLDING_ON_TOP' else (a,)
                 for a, task in enumerate(pair)] for pair in templates]
    expected = compose_graph(templates, bindings, num_objects=4)
    actual = compose_canonical_graph(ids)
    validate_graph(actual)
    for name, value in vars(expected).items():
        if torch.is_tensor(value):
            assert torch.equal(getattr(actual, name), value), name
        else:
            assert getattr(actual, name) == value


@pytest.mark.parametrize('preset', ['random_scenario', 'holding', 'sit', 'climb',
                                    'holding_at', 'holding_ontop'])
@pytest.mark.parametrize('shuffle', [False, True])
def test_cached_probabilities_and_partial_reset_match_original_sampler(preset, shuffle):
    sizes = sample_sizes(128, 'cpu', preset)
    probabilities = task_probabilities(sizes[:, :2])
    rows = torch.tensor([127, 4, 18, 0, 65])
    spec = dict(CFG['relationGraph'], shuffle_edge_order=shuffle)
    torch.manual_seed(42)
    expected = sample_size_graph(sizes[rows], spec, preset)
    torch.manual_seed(42)
    actual = sample_size_conditioned_graph(probabilities[rows], spec, preset)
    validate_graph(actual)
    for name, value in vars(expected).items():
        if torch.is_tensor(value):
            assert torch.equal(getattr(actual, name), value), name


@pytest.mark.parametrize('preset', ['holding', 'sit', 'climb', 'holding_at', 'holding_ontop'])
def test_progress_uses_correct_target_dimensions_without_changing_phi(preset):
    sizes = sample_sizes(1, 'cpu', preset)
    graph = sample_size_graph(sizes, CFG['relationGraph'], preset)
    hands = torch.zeros(1, 2, 2, 3)
    feet = torch.zeros_like(hands)
    roots = torch.zeros(1, 2, 3)
    roots[..., 0] = 2.
    boxes = torch.zeros(1, 4, 13)
    boxes[..., 6] = 1.
    boxes[:, :2, 0] = 1.
    goals = torch.zeros(1, 2, 3)
    phi, diag = evaluate_interaction_edges(hands, feet, roots, boxes, sizes, goals, graph,
                                           CFG['relationReward'], .94)
    base = deepcopy(CFG['relationReward'])
    base['progress'] = {'kind': 'distance', 'delta': .5, 'sigma': 1., 'climb_pinning': 'bbox_valid_radius'}
    previous, _ = evaluate_interaction_edges(hands, feet, roots, boxes, sizes, goals, graph, base, .94)
    assert torch.equal(phi, previous)
    buffers = CFG['relationReward']['progress']['bbox_buffers']
    relations = {6: 'HOLDING', 7: 'AT', 8: 'ON_TOP', 9: 'SIT', 10: 'CLIMB'}
    from utils.edge_interaction_spec import SIT, CLIMB
    from utils.edge_ontop_spec import ON_TOP
    relations.update({SIT: 'SIT', CLIMB: 'CLIMB', ON_TOP: 'ON_TOP'})
    for edge in range(graph.edge_valid.shape[-1]):
        if not graph.edge_valid[0, edge]:
            continue
        name = relations[int(graph.edge_relation[0, edge])]
        dst = int(graph.edge_dst[0, edge]) - 2
        radius = 0. if name == 'AT' else float(sizes[0, dst, :2].norm())/2
        expected = 1/(1 + max(float(diag['distance_xy'][0, edge])-radius-buffers[name], 0.))
        assert float(diag['progress'][0, edge]) == pytest.approx(expected)


def test_physics_screen_allows_contacts_but_rejects_kicks_and_nonfinite():
    root = torch.zeros(4, 13)
    boxes = torch.zeros(4, 2, 13)
    initial_root, initial_boxes = root.clone(), boxes.clone()
    root[0, 7] = 2.9
    root[1, 7] = 3.1
    boxes[2, 1, 7] = 5.1
    boxes[3, 0, 0] = float('nan')
    assert screen_states(root, boxes, initial_root, initial_boxes).tolist() == [True, False, False, False]


def test_cache_preserves_holes_late_phase_clip_time_and_fallback():
    skills = CFG['skill']
    sizes = torch.tensor([[[.4, .4, .4], [.4, .4, .4], [.6, .6, .3], [.6, .6, .3]]])
    lib = SimpleNamespace(num_motions=lambda: 1, _motion_weights=torch.ones(1), _motion_dt=torch.tensor([.05]))
    env = SimpleNamespace(_skill=skills, num_envs=1, device='cpu', _motion_lib={s: lib for s in skills})
    cache = SizeRsiCache.__new__(SizeRsiCache)
    cache.env, cache.sizes, cache.spec = env, sizes.tolist(), CFG['sizeAwareRsi']
    valid = np.array([1, 0, 1, 0, 1, 1, 0, 1], dtype=bool)
    phase = np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=bool)
    cache.entries = {(kind, size_key(sizes[0, 0])): [(valid, phase)]
                     for kind in ('sit', 'climb', 'pickUp', 'carryWith', 'putDown')}
    cache.entries[('putDownOnTop', size_key(sizes[0, 0]), size_key(sizes[0, 2]))] = [(np.zeros(8, dtype=bool), phase)]
    cache._prepare_sampling()
    rows, owners = torch.zeros(4000, dtype=torch.long), torch.zeros(4000, dtype=torch.long)
    clips, times = cache.sample('sit', rows, owners, torch.ones_like(rows))
    frames = (times/.05).round().long()
    assert set(frames.tolist()) <= {0, 2, 4, 5, 7}
    assert (frames == 5).float().mean() > .70
    assert clips.eq(0).all()
    templates = torch.tensor([4, 3])
    availability = cache.availability(torch.zeros(2, dtype=torch.long), torch.tensor([0, 1]), templates)
    requested = torch.full((2,), skills.index('putDown'))
    weights = torch.tensor([CFG['templateRsi']['HOLDING_ON_TOP']]*2)
    chosen, missing = resolve_skills(requested, weights, availability, skills.index('loco'))
    assert missing.tolist() == [True, False]
    assert chosen[0] != skills.index('putDown') and chosen[1] == skills.index('putDown')
    assert cache._key('putDownOnTop', 0, 0) != ('putDownOnTop', size_key(sizes[0, 0]), (10, 10, 5))


def test_persistent_cache_reuses_frames_and_invalidates_physics(tmp_path, monkeypatch):
    cfg = deepcopy(CFG)
    cfg['asset'] = {'assetRoot': str(tmp_path), 'assetFileName': 'humanoid.xml'}
    (tmp_path / 'humanoid.xml').write_text('<asset/>')
    cfg['sizeAwareRsi']['cacheDirectory'] = str(tmp_path / 'cache')
    tensor = torch.zeros(1)
    lib = SimpleNamespace(num_motions=lambda: 1, _motion_weights=torch.ones(1), _motion_dt=torch.tensor([.05]),
                          **{name: tensor for name in ('gts', 'grs', 'lrs', 'grvs', 'gravs', 'dvs')})
    env = SimpleNamespace(cfg={'env': cfg, 'sim': {'substeps': 2}}, sim_params=SimpleNamespace(dt=1/60),
        _skill=CFG['skill'], num_envs=1, device='cpu', _root_states=torch.zeros(6, 13),
        _dof_state=torch.zeros(1, 2), _motion_lib={s: lib for s in CFG['skill']},
        _box_size=torch.tensor([[[.4, .4, .4], [.4, .4, .4], [.6, .6, .3], [.6, .6, .3]]]))
    calls = []
    def screen(cache, kind, owner, profiles, *_):
        calls.append((kind, owner))
        cache.entries.update({key: [(np.ones(3, dtype=bool), np.ones(3, dtype=bool))] for key in profiles})
    monkeypatch.setattr(SizeRsiCache, '_screen_profiles', screen)
    first = SizeRsiCache(env)
    assert first.path.exists() and calls
    calls.clear()
    second = SizeRsiCache(env)
    assert not calls and second.path == first.path
    env.cfg['sim']['substeps'] = 3
    third = SizeRsiCache(env)
    assert calls and third.path != first.path


def test_screened_own_support_contact_allowed_foreign_body_obstruction_rejected():
    sizes = torch.full((1, 4, 3), .4)
    graph = sample_size_graph(sizes, CFG['relationGraph'], 'holding_ontop')
    boxes = torch.zeros(1, 4, 13)
    boxes[..., 6] = 1.
    boxes[0, :, :3] = torch.tensor([[0., 0., .6], [5., 0., .2],
                                   [0., 0., .2], [2., 0., .2]])
    bodies = torch.full((1, 2, 4, 13), 20.)
    bodies[0, 0, 0, :3] = boxes[0, 0, :3]
    bodies[0, 0, 1, :3] = boxes[0, 2, :3]
    task = SimpleNamespace(_size_aware_rsi=True, _scenario_no_climb=True,
        _scenario_with_climb=True, device='cpu', num_agents=2, num_objects=4,
        _relation_cfg=CFG['relationReward'], relation_runtime=SimpleNamespace(graph=graph),
        _logical_box_order=torch.arange(4)[None], _box_states=boxes, _box_size=sizes,
        _humanoid_root_states=torch.zeros(1, 2, 13), _kinematic_humanoid_rigid_body_states=bodies,
        _tar_pos=torch.zeros(1, 2, 3),
        _reset_ref_slots={'putDown': (torch.tensor([0]), torch.tensor([0]))},
        _reset_ref_motion_ids={'putDown': torch.tensor([0])})
    task._logical_box_values = lambda values, ids=None: values if ids is None else values[ids]
    task._putdown_ontop_supports = lambda: SampledOnTopTaskMixin._putdown_ontop_supports(task)
    assert not SampledOnTopTaskMixin._ontop_scene_infeasible(task, torch.tensor([0])).item()
    bodies[0, 0, 2, :3] = boxes[0, 3, :3]
    assert SampledOnTopTaskMixin._ontop_scene_infeasible(task, torch.tensor([0])).item()

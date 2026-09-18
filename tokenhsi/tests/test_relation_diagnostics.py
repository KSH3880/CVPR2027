"""Completed-episode denominators, terminal timing and bounded trace contracts."""
import csv
from pathlib import Path

import pytest
import torch

from env.tasks.multi_agent.relation_diagnostics import (
    PlacementEpisodeMetrics, RelationTimeline, placement_valid)
from utils.relation_task_spec import validate_relation_config, checkpoint_metadata, check_checkpoint_metadata


def test_common_placement_boundaries():
    xy = torch.tensor([.1, .10001, .1, .1])
    z = torch.tensor([.001, 0., .00101, -.001])
    assert placement_valid(xy, z).tolist() == [True, False, False, True]


def test_completed_episode_metrics_exclude_initial_and_condition_correctly():
    m = PlacementEpisodeMetrics(2, 2, .1, 'cpu')
    # Agent episodes: delayed placement then loss; never placed; starts placed;
    # placement at terminal step. First-reset geometry is not a simulated step.
    m.reset(torch.arange(2), torch.tensor([[False, False], [True, False]]))
    for placed in ([[False, False], [True, False]],
                   [[True, False], [False, False]],
                   [[True, False], [False, False]],
                   [[False, False], [True, True]]):
        m.step(torch.tensor(placed))
    m.finish(torch.ones(2, dtype=torch.bool))
    out = m.consume()
    assert out['placement/completed_count'] == 4
    assert out['placement/initially_placed_count'] == 1
    assert out['placement/eligible_completed_count'] == 3
    assert out['placement/reached_count'] == 2
    assert out['placement/episode_ever_rate'].item() == pytest.approx(2 / 3)
    assert out['placement/episode_final_rate'].item() == pytest.approx(1 / 3)
    # Unweighted mean of each reached episode's own post-first fraction.
    assert out['placement/post_first_retention'].item() == pytest.approx((2 / 3 + 1) / 2)
    assert out['placement/first_seconds'].item() == pytest.approx((.2 + .4) / 2)
    assert out['placement/longest_hold_seconds'].item() == pytest.approx((.2 + .1) / 2)
    assert out['placement/post_first_seconds'].item() == pytest.approx((.3 + .1) / 2)
    assert out['placement/initially_placed_final_rate'] == 1
    # Repeated termination cannot double-count completed agent episodes.
    m.finish(torch.ones(2, dtype=torch.bool))
    empty = m.consume()
    assert empty['placement/completed_count'] == 0
    assert 'placement/episode_ever_rate' not in empty
    assert 'placement/first_seconds' not in empty


def test_active_episodes_cross_rollouts_and_partial_resets_are_independent():
    m = PlacementEpisodeMetrics(2, 1, 1., 'cpu')
    m.reset(torch.arange(2), torch.zeros(2, 1, dtype=torch.bool))
    m.step(torch.tensor([[True], [False]]))
    assert m.consume()['placement/completed_count'] == 0
    m.finish(torch.tensor([True, False]))
    assert m.consume()['placement/reached_count'] == 1
    m.reset(torch.tensor([0]), torch.zeros(1, 1, dtype=torch.bool))
    m.step(torch.tensor([[False], [True]]))
    m.finish(torch.tensor([False, True]))
    out = m.consume()
    assert out['placement/first_seconds'] == 2
    assert out['placement/eligible_completed_count'] == 1
    assert m.active.tolist() == [[True], [False]]


def test_never_reached_has_no_fake_zero_conditional_metrics():
    m = PlacementEpisodeMetrics(1, 1, 1., 'cpu')
    m.reset(torch.tensor([0]), torch.zeros(1, 1, dtype=torch.bool))
    m.step(torch.zeros(1, 1, dtype=torch.bool))
    m.finish(torch.tensor([True]))
    out = m.consume()
    assert out['placement/episode_ever_rate'] == 0
    assert out['placement/episode_final_rate'] == 0
    assert out['placement/reached_count'] == 0
    assert 'placement/post_first_retention' not in out
    assert 'placement/first_seconds' not in out


def test_reset_resampling_is_not_counted_as_completion():
    m = PlacementEpisodeMetrics(1, 1, 1., 'cpu')
    ids = torch.tensor([0])
    m.reset(ids, torch.ones(1, 1, dtype=torch.bool))
    m.reset(ids, torch.zeros(1, 1, dtype=torch.bool))
    m.step(torch.ones(1, 1, dtype=torch.bool))
    m.finish(torch.tensor([True]))
    out = m.consume()
    assert out['placement/initially_placed_count'] == 0
    assert out['placement/completed_count'] == 1
    assert out['placement/first_seconds'] == 1


def test_initially_placed_only_has_separate_metrics():
    m = PlacementEpisodeMetrics(1, 1, 1., 'cpu')
    m.reset(torch.tensor([0]), torch.ones(1, 1, dtype=torch.bool))
    m.step(torch.zeros(1, 1, dtype=torch.bool))
    m.finish(torch.tensor([True]))
    out = m.consume()
    assert out['placement/initially_placed_fraction'] == 1
    assert out['placement/initially_placed_final_rate'] == 0
    assert out['placement/eligible_completed_count'] == 0
    assert 'placement/episode_ever_rate' not in out


def test_timeline_selection_every_step_cap_and_terminal_flush(tmp_path):
    trace = RelationTimeline(str(tmp_path), 2, {'timeline_every_episodes': 2, 'timeline_max_steps': 3})
    ids = torch.arange(2)
    trace.reset(ids)
    diag = {'holding/phi': torch.tensor([[.8, .9], [.7, .6]]),
            'saturation_active': torch.zeros(2, 2)}
    for step in range(1, 5):
        trace.record(step, torch.ones(2), torch.full((2,), step), .1, diag)
    trace.flush()
    assert not trace.selected
    with open(trace.path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6  # 3 steps, 2 agents, only env 0
    assert {r['env'] for r in rows} == {'0'}
    assert [float(r['episode_step']) for r in rows[::2]] == [1, 2, 3]
    trace.reset(ids)  # second episode is not traced
    assert not trace.selected
    trace.finish(torch.zeros(2, dtype=torch.bool))  # one actual step in episode 2
    trace.reset(ids)  # third is traced
    assert trace.selected == {0}
    trace.record(5, torch.full((2,), 3), torch.ones(2), .1, diag)
    trace.finish(torch.tensor([True, False]))
    assert not trace.rows and not trace.selected
    with open(trace.path) as f:
        assert len(list(csv.DictReader(f))) == 8


def test_initialization_resets_do_not_skip_first_timeline(tmp_path):
    trace = RelationTimeline(str(tmp_path), 1, {})
    for _ in range(3):
        trace.reset(torch.tensor([0]))
    assert trace.selected == {0}
    assert trace.episode_counts == [1]
    trace.record(1, torch.tensor([3]), torch.ones(1), .1, {'phi': torch.ones(1, 1)})
    trace.finish(torch.tensor([True]))
    assert trace.path is not None
    with open(trace.path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and float(rows[0]['episode']) == 3


def test_timeline_can_be_disabled_and_sessions_have_distinct_files(tmp_path):
    for cfg in ({'enabled': False}, {'timeline_enabled': False}, {'timeline_sample_envs': 0}):
        trace = RelationTimeline(str(tmp_path), 1, cfg)
        trace.reset(torch.tensor([0]))
        assert not trace.selected
    paths = []
    for _ in range(2):
        trace = RelationTimeline(str(tmp_path), 1, {})
        trace.reset(torch.tensor([0]))
        trace.record(1, torch.ones(1), torch.ones(1), .1, {'phi': torch.ones(1, 1)})
        trace.flush()
        paths.append(trace.path)
    assert paths[0] != paths[1]


@pytest.mark.parametrize('key,value', [
    ('timeline_enabled', 1), ('timeline_sample_envs', -1), ('timeline_sample_envs', True),
    ('timeline_every_episodes', 0), ('timeline_max_steps', 0), ('timeline_max_steps', 2.5)])
def test_invalid_timeline_config_is_rejected(key, value):
    with pytest.raises(ValueError, match='timeline'):
        validate_relation_config({'mode': 'state_relation_v0', 'diagnostics': {key: value}})


def test_diagnostic_only_options_preserve_checkpoint_compatibility():
    cfg = {'mode': 'state_relation_v0'}
    new = dict(cfg, diagnostics={'timeline_every_episodes': 20, 'timeline_max_steps': 100})
    validate_relation_config(new)
    check_checkpoint_metadata({'relation_metadata': checkpoint_metadata(cfg)}, checkpoint_metadata(new))


def test_terminal_hook_occurs_after_reset_calculation_before_auto_reset():
    # Integration ordering matters: placement must be sampled before terminal
    # detection, then finalized before any episode reset overwrites box states.
    source = (Path(__file__).resolve().parents[1] / 'env/tasks/multi_agent/humanoid_ma_carry.py').read_text()
    section = source.split('    def post_physics_step(self):', 1)[1].split('    def _compute_metrics_evaluation', 1)[0]
    assert section.index('self._compute_reward') < section.index('self._compute_reset()')
    assert section.index('self._compute_reset()') < section.index('self._finish_relation_diagnostics()')

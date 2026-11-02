from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import numpy as np
import pytest
from ksh_jointpush.ksh_terminal_success_eval import (
    success_history, terminal_success_records, success_stats,
)


def test_reentry_is_terminal_success_but_not_uninterrupted_hold():
    # Re-entry, stable hold, never reached, first reached at the last frame.
    current = np.array([[0, 0, 0, 0], [1, 1, 0, 0],
                        [0, 1, 0, 0], [1, 1, 0, 1]], dtype=bool)[:, None, :, None]
    result = success_history(current, np.array([4]), np.ones((1, 4), dtype=bool))
    assert result['ever'][0, :, 0].tolist() == [True, True, False, True]
    assert result['terminal'][0, :, 0].tolist() == [True, True, False, True]
    assert result['continuous'][0, :, 0].tolist() == [False, True, False, True]
    assert result['first_step'][0, :, 0].tolist() == [2, 2, 0, 4]


def test_separate_scene_end_and_criterion_specific_history():
    current = np.ones((4, 2, 1, 2), dtype=bool)
    current[2:, 0] = False  # These frames are after this scene's end.
    current[2, 1, 0, 0] = False  # Point leaves and returns; region keeps holding.
    result = success_history(current, np.array([2, 4]), np.ones((2, 1), dtype=bool))
    assert result['terminal'].all()
    assert result['continuous'][:, 0].tolist() == [[True, True], [False, True]]


def trace():
    frames, n, m, o = 4, 1, 4, 8
    packet = np.zeros((n, m, 5), dtype=np.int64)
    packet[..., 0] = 1
    packet[..., 1] = 4
    packet[..., 2] = np.arange(m)
    packet[..., 3] = m + np.arange(m)
    packet[..., 4] = m + o + np.arange(m)
    boxes = np.zeros((frames, n, o, 7), dtype=np.float32)
    boxes[..., 6] = 1
    boxes[:, 0, :m, 0] = [[1, 1, 1, 1], [.1, 1, .1, 1],
                          [.3, .1, .1, 1], [.1, .1, .1, .4]]
    valid = np.ones((frames, n, m), dtype=bool)
    valid[-1, 0, 2] = False
    return dict(packet=packet, roots=np.zeros((frames, n, m, 3), dtype=np.float32),
                boxes=boxes, sizes=np.ones((n, o, 3), dtype=np.float32),
                goals=np.zeros((n, m, 3), dtype=np.float32), groups=np.array([[0, 1, 1, 1]]),
                valid=valid, fall_step=np.array([[-1, -1, 3, -1]]),
                ever=np.array([[[1, 1], [1, 1], [1, 1], [0, 0]]], dtype=bool))


def test_fallen_member_passive_box_at_goal_does_not_count_terminal_success():
    agents, groups = terminal_success_records(trace(), 0)
    assert [r['ever_point'] for r in agents] == [True, True, True, False]
    assert [r['terminal_point'] for r in agents] == [True, True, False, False]
    assert [r['continuous_point'] for r in agents] == [False, True, False, False]
    assert agents[2]['geometric_terminal_point'] and not agents[2]['alive_at_terminal']
    assert len(groups) == 2 and groups[0]['members'] == 1 and groups[1]['members'] == 3
    assert not groups[1]['ever_point'] and not groups[1]['terminal_point']
    stats = success_stats(agents, 'point')
    assert stats['trials'] == 4
    assert stats['terminal']['pooled_rate'] == .5
    assert stats['terminal']['pooled_given_ever'] == 2 / 3


def test_group_each_ever_is_distinct_from_same_end_success():
    data = trace()
    data['boxes'][1, 0, 3, 0] = .1
    data['ever'][0, 3] = True
    agents, groups = terminal_success_records(data, 1)
    assert groups[1]['ever_point']
    assert not groups[1]['terminal_point']
    data['fall_step'][:] = -1
    data['valid'][:] = True
    data['boxes'][-1, 0, 3, 0] = .1
    agents, groups = terminal_success_records(data, 1)
    assert groups[1]['terminal_point']
    assert not groups[1]['continuous_point']  # Agent3 left and re-entered.


def test_all_fallen_end_uses_fall_frame_ever_and_excludes_later_passive_motion():
    data = trace()
    data['fall_step'][:] = 2
    data['valid'][2:] = False
    data['boxes'][2:, 0, :4, 0] = 0
    data['ever'][:] = False
    data['ever'][0, [0, 2]] = True
    agents, groups = terminal_success_records(data, 0)
    assert all(r['terminal_step'] == 2 for r in agents)
    assert all(r['terminal_reason'] == 'all_agents_fallen' for r in agents)
    assert [r['ever_point'] for r in agents] == [True, False, True, False]
    assert not any(r['terminal_point'] for r in agents)
    assert not any(r['continuous_point'] for r in agents)
    assert [r['geometric_terminal_point'] for r in agents] == [True, False, True, False]


def test_packet_permutation_maps_falls_to_physical_actor_and_ever_to_task():
    data = trace()
    original = terminal_success_records(data, 2)
    order = [2, 0, 3, 1]
    data['packet'] = data['packet'][:, order]
    data['ever'] = data['ever'][:, order]
    agents, groups = terminal_success_records(data, 2)
    expected = [{k: v for k, v in r.items() if k != 'task_index'} for r in original[0]]
    actual = [{k: v for k, v in r.items() if k != 'task_index'} for r in agents]
    assert sorted(actual, key=lambda r: r['agent']) == sorted(expected, key=lambda r: r['agent'])
    assert groups == original[1]


def test_reactivated_or_inconsistent_ever_trace_is_rejected():
    data = trace()
    data['valid'][-1, 0, 2] = True
    with pytest.raises(AssertionError, match='Incomplete/reactivated'):
        terminal_success_records(data, 0)
    data = trace()
    data['ever'][0, 3] = True
    with pytest.raises(AssertionError, match='Original ever'):
        terminal_success_records(data, 0)

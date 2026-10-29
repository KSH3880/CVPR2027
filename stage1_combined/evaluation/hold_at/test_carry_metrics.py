import numpy as np
import pytest

from evaluation.hold_at.carry_metrics import ACTION_LIMIT, PROTOCOL, score
from evaluation.hold_at.metrics import aggregate_tokenhsi, PROTOCOL as COMMON_PROTOCOL


def test_cutoff_upright_terminal_frame_and_later_episode_exclusion():
    objects = np.zeros((605, 5, 13))
    objects[..., 0] = 1
    objects[..., 6] = 1
    active = np.ones((605, 5), bool)
    objects[:, 0, 0] = .125
    objects[:, 0, 3] = .5
    objects[:, 0, 6] = np.sqrt(.75)  # Position reached but tilted60degrees.
    objects[598:, 1, 0] = .125  # Final native20s action, also terminal, must count.
    active[599:, 1] = False
    objects[599:, 2, 0] = .125  # One action too late for the native20s cutoff.
    objects[1:, 3, 0] = .125  # Reached only after the first episode ended.
    active[1:, 3] = False
    objects[:, 4, 0] = .125
    up = np.tile([0., 0., 1.], (5, 1))
    result = score(dict(objects=objects, goals=np.zeros((605, 5, 3)), active=active), up)
    assert ACTION_LIMIT == result['horizon_action_steps'] == 599
    assert result['success_by_env'] == [False, True, False, False, True]
    assert result['first_success_step'] == [0, 599, 0, 0, 1]
    assert result['min_success_error_cm'] == [None, 12.5, None, None, 12.5]
    assert result['success_count'] == 2 and result['success_precision_cm'] == 12.5


def test_original_per_object_up_axis_is_used_instead_of_assuming_world_z():
    objects = np.zeros((1, 2, 13)); objects[..., 6] = 1
    objects[0, 1, 5:7] = np.sqrt(.5)  # Rotates localX toY; worldZ remains unchanged.
    up = np.tile([1., 0., 0.], (2, 1))
    result = score(dict(objects=objects, goals=np.zeros((1, 2, 3)), active=np.ones((1, 2), bool)), up)
    assert result['success_by_env'] == [True, False]


def test_position_is_full3d_and_includes_twenty_cm_boundary():
    objects = np.zeros((1, 3, 13)); objects[..., 6] = 1
    objects[0, :, :3] = [[.2, 0, 0], [.15, .15, 0], [0, 0, .201]]
    result = score(dict(objects=objects, goals=np.zeros((1, 3, 3)), active=np.ones((1, 3), bool)),
                   np.tile([0., 0., 1.], (3, 1)))
    assert result['success_by_env'] == [True, False, False]


def test_secondary_stats_use_independent_seed_sample_std_without_mutating_primary():
    records = [dict(protocol=COMMON_PROTOCOL, seed=i, num_envs=512,
                    point_ever_count=512, region_ever_count=512,
                    tokenhsi_criteria=dict(protocol=PROTOCOL, success_count=k, success_precision_cm=10.))
               for i, k in enumerate([0, 256, 512])]
    result = aggregate_tokenhsi(records)
    assert result['mean_percent'] == result['sample_std_percentage_points'] == 50
    assert result['pooled_success_precision_cm'] == 10
    assert all(r['point_ever_count'] == 512 for r in records)
    with pytest.raises(ValueError):
        aggregate_tokenhsi(records[:2])

import numpy as np
import pytest
from learning.cvpr2027_scoring import spatial_success, ever_from_trace, group_scores, seed_statistics
from learning.cvpr2027_scoring import spatial_metrics, trial_minimum_distances, error_statistics


def test_region_yaw_and_no_contact_gate():
    box = np.array([0, 0, .25, 0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
    # Long local X maps to world Y. Inside Region, outside Point.
    point, region = spatial_success(0, [0, .3, .6], box, [1, .5, .5], [0, 0, 0])
    assert not point and region
    assert not spatial_success(0, [.21, 0, .6], box, [1, .5, .5], [0, 0, 0])[1]


def test_push_3d_boundary_and_no_one_mm_gate():
    box = [0, 0, .5, 0, 0, 0, 1]
    assert spatial_success(4, [0, 0, .51], box, [1, 1, 1], [0, 0, .5]).all()
    assert spatial_success(4, [.2, 0, 0], box, [1, 1, 1], [0, 0, 0]).all()
    assert not spatial_success(4, [.2000000001, 0, 0], box, [1, 1, 1], [0, 0, 0]).any()


def test_fall_frame_counts_but_later_frames_do_not():
    success = np.zeros((3, 2, 2), bool)
    success[1, 0] = True
    success[2, 1] = True
    fallen = np.array([[False, False], [True, True], [False, False]])
    ever, valid = ever_from_trace(success, fallen)
    assert ever[0].all() and not ever[1].any()
    assert valid[1].all() and not valid[2].any()


def test_group_denominators_and_null():
    # Two independent people, one enabling pair, one coupled triple.
    score = group_scores([[1, 0, 1, 1, 0, 0, 0]], [[0, 0, 1, 1, 2, 2, 2]],
                         [[0, 1, 2, 2, 3, 3, 3]], [[0, 0, 0, 0, 1, 1, 1]],
                         frame_success=[[[1, 0, 1, 1, 0, 0, 0]]],
                         frame_valid=np.ones((1, 1, 7), bool))
    assert score['independent']['rate'] == .5
    assert score['enabling']['trials'] == score['coupled']['trials'] == 1
    assert score['coupled']['rate'] == 1 and score['coupled_sizes']['3'] == 1
    assert np.isclose(score['mixed'], 2.5/3)
    assert group_scores([[1]], [[0]], [[0]], [[0]])['mixed'] is None
    assert seed_statistics([0, .5, 1])['std'] == .5


def test_enable_requires_simultaneous_valid_success():
    args = ([[1, 1]], [[1, 1]], [[0, 0]], [[0, 0]])
    with pytest.raises(ValueError, match='frame_success'):
        group_scores(*args)
    frames = np.array([[[1, 0]], [[0, 1]]], dtype=bool)
    valid = np.ones_like(frames)
    assert group_scores(*args, frame_success=frames, frame_valid=valid)['enabling']['rate'] == 0
    frames[1, 0, 0] = True
    assert group_scores(*args, frame_success=frames, frame_valid=valid)['enabling']['rate'] == 1
    # A previous fall invalidates one member; cached individual ever cannot help.
    valid[1, 0, 0] = False
    assert group_scores(*args, frame_success=frames, frame_valid=valid)['enabling']['rate'] == 0


def test_xy100_boundary_and_unshrunk_plane_error():
    box = [0, 0, .25, 0, 0, 0, 1]
    # XY inside full support but outside old 80%; Point may fail while Region succeeds.
    ok, distance = spatial_metrics(0, [.49, 0, .7], box, [1, 1, .5], [0, 0, 0], xy_fraction=1.)
    assert ok.tolist() == [False, True]
    np.testing.assert_allclose(distance, [np.hypot(.49,.1), .1], atol=1e-12)
    assert not spatial_success(0, [.49,0,.7], box, [1,1,.5], [0,0,0])[1]
    assert spatial_success(0, [.5,0,.6], box, [1,1,.5], [0,0,0], xy_fraction=1.)[1]
    assert not spatial_success(0, [.500001,0,.6], box, [1,1,.5], [0,0,0], xy_fraction=1.)[1]
    _, distance = spatial_metrics(0, [.6,.7,.9], box, [1,1,.5], [0,0,0], xy_fraction=1.)
    np.testing.assert_allclose(distance[1], np.sqrt(.1**2+.2**2+.3**2))


def test_xy100_yaw_payload_and_transport_errors():
    box = [0,0,.25,0,0,np.sin(np.pi/4),np.cos(np.pi/4)]
    ok, d = spatial_metrics(3, [0,.49,.75], box, [1,.4,.5], [0,0,0], .4, 1.)
    assert ok.tolist() == [False,True]
    np.testing.assert_allclose(d[1], .05)
    for task in (2,4):
        ok, d = spatial_metrics(task, [.1,0,.1], box, [1,.4,.5], [0,0,0], xy_fraction=1.)
        assert ok.all() and d[0] == d[1]
        np.testing.assert_allclose(d[0], np.sqrt(.02))


def test_native_chair_uses_original_target_not_backrest_top():
    from learning.cvpr2027_eval import metrics_frame
    roots = np.zeros((2,1,13)); roots[:,0,2] = [.65,1.1]
    boxes = np.zeros((2,2,13)); boxes[:,:,6] = 1
    boxes[:,0,2] = .5
    sizes = np.ones((2,2,3))
    goals = np.zeros((2,1,3)); goals[:,:,2] = .65
    ok, d = metrics_frame('sit', roots, boxes, sizes, goals, 1., 'native_point')
    assert ok.tolist() == [[True,True],[False,False]]
    np.testing.assert_allclose(d, [[0,0],[.45,.45]])


def test_error_uses_all_valid_frames_but_only_region_success_trials():
    success = np.array([[[0,1],[0,0]], [[0,0],[0,0]], [[1,1],[1,1]]], bool)
    fallen = np.array([[0,0],[1,1],[0,0]], bool)
    ever, valid = ever_from_trace(success, fallen)
    # Point minimum can occur after Region success on the fall frame; later zero is invalid.
    distances = np.array([[[.6,.1],[.9,.9]], [[.3,.4],[.5,.5]], [[0,0],[0,0]]])
    minimum = trial_minimum_distances(distances, valid)
    np.testing.assert_allclose(minimum, [[.3,.1],[.5,.5]])
    errors = error_statistics(minimum, ever[:,1])
    assert errors == dict(point=dict(mean_cm=30.,samples=1), region=dict(mean_cm=10.,samples=1))
    assert error_statistics(minimum, [False,False])['point'] == dict(mean_cm=None,samples=0)
    assert seed_statistics([1.,None,3.])['mean'] is None


def test_single_scene_ranges_and_goal_direction():
    from env.cvpr2027_scene import sample_scene
    n = 256
    roots = np.zeros((n, 3))
    sizes = np.tile([[1.8, 1.8, 1.5], [.5, .5, .4]], (n, 1, 1))
    boxes, goals = sample_scene(np.random.RandomState(0), roots, sizes, 'push')
    source = np.linalg.norm(boxes[:, 0, :2], axis=-1)
    delta = goals[:, :2]-boxes[:, 0, :2]
    distance = np.linalg.norm(delta, axis=-1)
    assert np.all((source >= 2) & (source <= 5))
    assert np.all((distance >= 2) & (distance <= 5))
    yaw = 2*np.arctan2(boxes[:, 0, 5], boxes[:, 0, 6])
    forward = np.stack((-np.sin(yaw), np.cos(yaw)), -1)
    assert np.all((delta*forward).sum(-1)/distance >= np.cos(np.pi/4)-1e-6)
    assert np.allclose(boxes[:, :, 2], sizes[:, :, 2]/2)
    assert np.all(boxes[:, :, 7:] == 0)
    for task in ('sit', 'climb', 'at', 'on_top'):
        sizes = np.tile([[.5, .5, .4], [.6, .6, .4]], (n, 1, 1))
        boxes, goals = sample_scene(np.random.RandomState(1), roots, sizes, task)
        approach = np.linalg.norm(boxes[:, 0, :2], axis=-1)
        assert np.all((approach >= 1) & (approach <= 5))
        if task in ('at', 'on_top'):
            d = np.linalg.norm(goals[:, :2]-boxes[:, 0, :2], axis=-1)
            assert np.all((d >= 1) & (d <= 5))
        if task == 'on_top':
            assert np.allclose(boxes[:, 1, :2], goals[:, :2])

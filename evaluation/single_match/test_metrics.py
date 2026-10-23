import numpy as np
import pytest
from metrics import classify, geometry, score_trace, aggregate


def scene(size=(1., 1., .4)):
    boxes = np.zeros((1, 2, 13), dtype=np.float32)
    boxes[..., 6] = 1
    boxes[..., 2] = .2
    sizes = np.array([[size, size]], dtype=np.float32)
    return boxes, sizes


def test_point_and_region_are_not_nested():
    boxes, sizes = scene((.25, .25, .4))
    root = np.array([[.15, 0, .5]])
    point, region = classify('sit', root, boxes, sizes, np.zeros((1, 3)))
    assert point.item() and not region.item()
    sizes[:] = [1.6, 1.6, .4]
    root[0, 0] = .5
    point, region = classify('sit', root, boxes, sizes, np.zeros((1, 3)))
    assert not point.item() and region.item()


def test_yaw_region_and_independent_z_tolerance():
    boxes, sizes = scene((1., .25, .4))
    boxes[..., 5:7] = [np.sin(np.pi/4), np.cos(np.pi/4)]
    root = np.array([[0, .3, .65]])
    point, region = classify('sit', root, boxes, sizes, np.zeros((1, 3)))
    assert not point.item() and region.item()
    root[0, :2] = [.3, 0]
    assert not classify('sit', root, boxes, sizes, np.zeros((1, 3)))[1].item()


def test_rotated_top_and_upright_payload_height():
    boxes, sizes = scene((.6, .5, .4))
    boxes[0, 1, 3:7] = [np.sin(np.pi/4), 0, 0, np.cos(np.pi/4)]
    boxes[0, 0, 3:7] = boxes[0, 1, 3:7]
    target = geometry('ontop', np.zeros((1, 3)), boxes, sizes, np.zeros((1, 3)))[1]
    np.testing.assert_allclose(target[0, 2], .2+.25+.2)


def test_at_is_identical_and_push_zone_is_fixed():
    boxes, sizes = scene((1.7, 1.7, 1.7))
    boxes[0, 0, :3] = [.3, 0, .85]
    goal = np.array([[0, 0, .85]])
    point, region = classify('at', np.zeros((1, 3)), boxes, sizes, goal)
    assert np.array_equal(point, region) and not point.item()
    assert classify('push', np.zeros((1, 3)), boxes, sizes, goal)[1].item()
    boxes[0, 0, 0] = .5
    assert not classify('push', np.zeros((1, 3)), boxes, sizes, goal)[1].item()


def test_ever_counts_terminating_frame_but_masks_later_frames():
    boxes, sizes = scene()
    b = np.repeat(boxes[:, :1][None], 3, axis=0)
    b[..., 0, :3] = [1., 0, 0]
    b[1, 0, 0, :3] = 0
    trace = dict(root=np.zeros((3, 1, 3)), boxes=b, sizes=sizes[:, :1], goal=np.zeros((3, 1, 3)),
        active=np.array([[1], [1], [0]], bool), done=np.array([[0], [1], [1]], bool))
    result = score_trace('at', trace, 1/30)
    assert result['point']['ever'] == [True]
    assert result['point']['first_hit_seconds'] == [2/30]
    b[1, 0, 0, 0] = 1
    b[2, 0, 0, 0] = 0
    assert score_trace('at', trace, 1/30)['point']['ever'] == [False]


def test_statistics_use_sample_std_and_reject_duplicate_seed():
    rows = [dict(policy='p', task='sit', seed=i, scores={m: dict(rate_percent=v) for m in ('point', 'region')})
            for i, v in enumerate((0., 50., 100.))]
    summary = aggregate(rows)[0]
    assert summary['point'] == dict(mean_percent=50., sample_std_percent=50.)
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate(rows+rows[:1])

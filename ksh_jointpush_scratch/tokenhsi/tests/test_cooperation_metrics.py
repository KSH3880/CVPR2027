import unittest
import numpy as np
import torch
from utils.cooperation_metrics import surface_distance, window_counts, group_trials, calculate, summarize


class CooperationMetricsTest(unittest.TestCase):
    def test_rotated_box_surface(self):
        boxes = torch.tensor([[0., 0., 0., 0., 0., 2**-.5, 2**-.5]], dtype=torch.float64)
        sizes = torch.tensor([[2., 4., 2.]], dtype=torch.float64)
        hands = torch.tensor([[[0., 1.06, 0.], [0., 0., 0.]]], dtype=torch.float64)
        torch.testing.assert_close(surface_distance(hands, boxes, sizes), torch.tensor([[.06, 1.]], dtype=torch.float64))

    def test_backdating_success_and_no_start(self):
        good = np.array([[1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
        contact = np.ones_like(good)
        contact[3:] = False
        result = window_counts(good, contact, np.array([5, 3, 5]), 3)
        np.testing.assert_array_equal(result['start'], [1, -1, -1])
        np.testing.assert_array_equal(result['window_frames'], [5, 0, 0])
        np.testing.assert_array_equal(result['contact_frames'], [3, 0, 0])
        result = window_counts(good, contact, np.array([5, 5, 5]), 6)
        np.testing.assert_array_equal(result['window_frames'], [0, 0, 0])

    def fixture(self):
        steps, agents = 5, 4
        boxes = np.zeros((steps, 1, agents, 7), dtype=np.float32)
        boxes[..., 6] = 1
        boxes[:, 0, :2, 0] = np.array([.1, .2, .3, .4, .5])[:, None]
        hands = np.zeros((steps, 1, agents, 2, 3), dtype=np.float32)
        hands[..., 0] = boxes[..., None, 0] + .5
        sizes = np.ones((1, agents, 3), dtype=np.float32)
        goals = np.zeros((1, agents, 3), dtype=np.float32)
        goals[..., 0] = 10
        initial = boxes[0].copy()
        initial[0, :2, 0] = 0
        packet = np.zeros((1, agents, 5), dtype=np.int64)
        packet[..., 0] = 1
        packet[..., 1] = 4
        trace = dict(groups=np.array([[0, 0, 1, 2]]), packet=packet, valid=np.ones((steps, 1, agents), dtype=bool))
        raw = dict(hands=hands, boxes=boxes, sizes=sizes, initial_boxes=initial, goals=goals)
        return raw, trace

    def test_original_members_after_fall_and_all_hands(self):
        raw, trace = self.fixture()
        raw['hands'][2, 0, 1, 1, 0] += .2
        trace['valid'][3:, 0, 1] = False
        metrics, counts = calculate(raw, trace, .1)
        key = 'displacement_0.2_contact_0.06'
        np.testing.assert_array_equal(counts[key+'__start'], [2])
        np.testing.assert_array_equal(counts[key+'__window_frames'], [4])
        np.testing.assert_array_equal(counts[key+'__contact_frames'], [1])
        self.assertEqual(metrics['variants'][key]['equal_size_mean_percent'], 25.)
        self.assertEqual(len(metrics['variants']), 18)
        self.assertEqual(metrics['variants']['speed_0.2_contact_0.06']['equal_size_mean_percent'], 0.)

    def test_two_groups_and_single_exclusion(self):
        raw, trace = self.fixture()
        trace['groups'][:] = [[0, 0, 1, 1]]
        envs, reps, members = group_trials(trace['groups'], trace['packet'])
        np.testing.assert_array_equal(reps, [0, 2])
        np.testing.assert_array_equal(members.sum(-1), [2, 2])
        metrics, counts = calculate(raw, trace, .1)
        self.assertEqual(metrics['trials'], 2)
        self.assertEqual(metrics['variants']['displacement_0.2_contact_0.06']['equal_size_mean_percent'], 50.)

    def test_strict_contact_threshold(self):
        raw, trace = self.fixture()
        distances = np.full(raw['hands'].shape[:-1], .06)
        metrics, _ = calculate(raw, trace, .1, distances)
        self.assertEqual(metrics['variants']['displacement_0.2_contact_0.06']['equal_size_mean_percent'], 0.)
        self.assertEqual(metrics['variants']['displacement_0.2_contact_0.08']['equal_size_mean_percent'], 100.)

    def test_seed_sample_std(self):
        raw, trace = self.fixture()
        first, _ = calculate(raw, trace, .1)
        trace['valid'][:] = False
        second, _ = calculate(raw, trace, .1)
        result = summarize([first, second])['variants']['displacement_0.2_contact_0.06']
        self.assertEqual(result['mean_percent'], 50.)
        self.assertAlmostEqual(result['std_percent'], 50*2**.5)

    def test_equal_weight_group_sizes(self):
        raw, trace = self.fixture()
        raw = {key: np.repeat(value, 3, axis=1 if key in ('boxes', 'hands') else 0) for key, value in raw.items()}
        trace = {key: np.repeat(value, 3, axis=0 if key != 'valid' else 1) for key, value in trace.items()}
        trace['groups'][2] = [0, 0, 0, 1]
        raw['boxes'][:, 2, 2] = raw['boxes'][:, 2, 0]
        raw['initial_boxes'][2, 2] = raw['initial_boxes'][2, 0]
        raw['hands'][:, 2, 2] = raw['hands'][:, 2, 0]
        trace['valid'][:, 2, :3] = False
        metrics, _ = calculate(raw, trace, .1)
        item = metrics['variants']['displacement_0.2_contact_0.06']
        self.assertAlmostEqual(item['pooled_mean_percent'], 200/3)
        self.assertEqual(item['equal_size_mean_percent'], 50.)


if __name__ == '__main__':
    unittest.main()

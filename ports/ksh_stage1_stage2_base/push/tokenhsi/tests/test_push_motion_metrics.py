import unittest
import torch
from utils.push_motion_metrics import PushMotionMetrics, tick_motion


class PushMotionMetricsTests(unittest.TestCase):
    def test_contact_travel_is_gated_per_tick_and_retreat_is_signed(self):
        goal = torch.tensor([[[2., 0., 0.]]])
        start = torch.zeros_like(goal)
        middle = torch.tensor([[[1., 0., 0.]]])
        end = torch.tensor([[[.5, 0., 0.]]])
        yes = torch.tensor([[True]])
        a = tick_motion(start, middle, goal, yes)
        b = tick_motion(middle, end, goal, ~yes)
        torch.testing.assert_close((a+b)[0, 0], torch.tensor([1.5, 1., .5, 1., 1.]))
        retreat = tick_motion(middle, end, goal, yes)[0, 0]
        self.assertAlmostEqual(float(retreat[3]), -.5)
        self.assertEqual(float(retreat[4]), 0.)

    def test_vertical_motion_does_not_count_and_at_goal_is_finite(self):
        before = torch.zeros(1, 1, 3)
        after = before.clone(); after[..., 2] = 9
        torch.testing.assert_close(tick_motion(before, after, before, torch.ones(1, 1)),
                                   torch.zeros(1, 1, 5))

    def test_mixed_task_denominator_episode_reset_and_consume(self):
        m = PushMotionMetrics((2, 2), 'cpu')
        active = torch.tensor([[True, False], [False, True]])
        before = torch.zeros(2, 2, 3)
        after = before.clone(); after[..., 0] = 1
        goal = after*4
        contact = torch.tensor([[True, True], [True, False]])
        m.begin_step(); m.tick(before, after, goal, contact, active)
        m.finish_step(active, contact, active, torch.ones(2, 2), .5)
        m.finish_episodes(torch.tensor([True, False]), active)
        m.reset(torch.tensor([0]))
        self.assertEqual(float(m.episode[0].sum()), 0.)
        self.assertGreater(float(m.episode[1].sum()), 0.)
        values = m.consume()
        self.assertEqual(float(values['push/contact']), .5)
        self.assertEqual(float(values['push/contact_tick_fraction']), .5)
        self.assertEqual(float(values['push/travel_m_per_step']), 1.)
        self.assertEqual(float(values['push/contact_travel_m_per_step']), .5)
        self.assertEqual(float(values['push/contact_travel_m_per_s']), 1.)
        self.assertEqual(float(values['push/episode_contact_travel_m']), 1.)
        self.assertEqual(float(values['push/at']), 1.)
        self.assertEqual(float(values['push/contact_and_at']), .5)
        self.assertEqual(m.consume(), {})

    def test_subdivision_preserves_distance_and_signed_progress(self):
        before = torch.zeros(1, 1, 3)
        after = torch.tensor([[[1., 1., 0.]]])
        goal = after*4
        hit = torch.ones(1, 1, dtype=torch.bool)
        whole = tick_motion(before, after, goal, hit)
        pieces = tick_motion(before, after/2, goal, hit) + tick_motion(after/2, after, goal, hit)
        torch.testing.assert_close(whole, pieces)


if __name__ == '__main__':
    unittest.main()

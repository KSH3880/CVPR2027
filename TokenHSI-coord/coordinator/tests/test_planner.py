from __future__ import annotations

import unittest

import torch

from coordinator.model import JointCoordinator
from coordinator.planner import (
    _arrival_times,
    candidate_costs,
    select_candidate,
    timed_rollout,
)
from coordinator.executor_calibration import (
    MS18_TIMING_APPROACH_SPEEDS,
    MS18_TIMING_CARRY_SPEEDS,
    MS18_TIMING_PHASE1_REMAINING_DWELL_S,
    MS18_TIMING_PICKUP_DWELL_S,
)
from coordinator.schema import CANDIDATES
from coordinator.tests.common import make_state


class PlannerTest(unittest.TestCase):
    def test_joint_time_rollout_and_selection(self):
        state = make_state(3)
        output = JointCoordinator()(state)
        rollout = timed_rollout(
            output["path_world"], output["speed"], output["pickup_dwell"], state
        )
        self.assertEqual(rollout["root"].shape[:3], (3, CANDIDATES, 2))
        metrics = candidate_costs(output, state)
        self.assertEqual(metrics["cost"].shape, (3, CANDIDATES))
        self.assertTrue(torch.isfinite(metrics["train_cost"]).all())
        choice = select_candidate(output, state)
        self.assertEqual(choice.path_world.shape, (3, 2, 33, 2))
        self.assertEqual(choice.speed.shape, (3, 2, 33))

    def test_candidate_selector_does_not_use_uncalibrated_critic_offsets(self):
        state = make_state(1)
        output = JointCoordinator()(state)
        baseline = candidate_costs(output, state)["cost"]
        changed = dict(output)
        changed["candidate_value"] = torch.tensor(
            [[-1e6, 1e6, -1e6, 1e6]], dtype=baseline.dtype
        )
        actual = candidate_costs(changed, state)["cost"]
        self.assertTrue(torch.equal(baseline, actual))

    def test_held_state_has_no_pickup_dwell(self):
        state = make_state(1, held=True)
        output = JointCoordinator()(state)
        self.assertEqual(float(output["pickup_dwell"].abs().max()), 0.0)

    def test_measured_timing_uses_phase_speed_lookup_and_pickup_dwell(self):
        state = make_state(1)
        output = JointCoordinator()(state)
        path = output["path_world"]
        speed = torch.full_like(output["speed"], 1.5)
        _, arrival = _arrival_times(
            path, speed, output["pickup_dwell"], state=state,
            measured_executor_timing=True,
        )
        approach_length = (
            path[..., 1:17, :] - path[..., :16, :]
        ).norm(dim=-1).sum(dim=-1)
        carry_length = (
            path[..., 17:, :] - path[..., 16:-1, :]
        ).norm(dim=-1).sum(dim=-1)
        self.assertTrue(torch.allclose(
            arrival[..., 16],
            approach_length / MS18_TIMING_APPROACH_SPEEDS[-1],
        ))
        self.assertTrue(torch.allclose(
            arrival[..., 17] - arrival[..., 16],
            torch.full_like(arrival[..., 16], MS18_TIMING_PICKUP_DWELL_S),
        ))
        self.assertTrue(torch.allclose(
            arrival[..., -1] - arrival[..., 17],
            carry_length / MS18_TIMING_CARRY_SPEEDS[-1],
        ))

    def test_measured_timing_uses_remaining_dwell_in_wait_phase(self):
        state = make_state(1)
        state.phase.fill_(1.0)
        output = JointCoordinator()(state)
        _, arrival = _arrival_times(
            output["path_world"], output["speed"], output["pickup_dwell"],
            state=state, measured_executor_timing=True,
        )
        self.assertTrue(torch.allclose(
            arrival[..., 17] - arrival[..., 16],
            torch.full_like(
                arrival[..., 16], MS18_TIMING_PHASE1_REMAINING_DWELL_S
            ),
        ))


if __name__ == "__main__":
    unittest.main()

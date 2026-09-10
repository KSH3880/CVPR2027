from __future__ import annotations

import unittest
from dataclasses import fields, replace

import torch

from stack_planner.reward import (
    StackIntervalCosts,
    StackPhysicalState,
    StackRewardConfig,
    compute_stack_planner_reward,
    stack_task_potential,
)


def physical_state(batch: int = 2) -> StackPhysicalState:
    vector = torch.zeros(batch, 3)
    scalar = torch.zeros(batch)
    return StackPhysicalState(
        bottom_position_error=vector.clone(),
        bottom_linear_velocity=vector.clone(),
        bottom_angular_velocity=vector.clone(),
        bottom_tilt=scalar.clone(),
        bottom_released=torch.ones(batch),
        bottom_root_distance=torch.zeros(batch),
        clearance=torch.full((batch,), 1.5),
        top_position_error=vector.clone(),
        top_linear_velocity=vector.clone(),
        top_angular_velocity=vector.clone(),
        top_tilt=scalar.clone(),
        top_released=torch.ones(batch),
        top_root_distance=torch.zeros(batch),
        stack_success=scalar.clone(),
    )


def zero_costs(batch: int = 2) -> StackIntervalCosts:
    zero = torch.zeros(batch)
    return StackIntervalCosts(
        collision=zero.clone(),
        bottom_disturbance=zero.clone(),
        humanoid_fall=zero.clone(),
        elapsed_seconds=zero.clone(),
        invalid_plan=zero.clone(),
        unsafe_plan=zero.clone(),
    )


class StackPlannerRewardTest(unittest.TestCase):
    def test_dependency_potential_opens_bottom_then_clearance_then_top(self):
        bad = physical_state(1)
        bad.bottom_position_error[:, 0] = 1.0
        bad.clearance[:] = 0.2
        bad.top_position_error[:, 0] = 1.0

        bottom = replace(bad, bottom_position_error=torch.zeros(1, 3))
        clear = replace(bottom, clearance=torch.full((1,), 1.5))
        top = replace(clear, top_position_error=torch.zeros(1, 3))
        values = [
            stack_task_potential(state)["potential"].item()
            for state in (bad, bottom, clear, top)
        ]
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])
        self.assertLess(values[2], values[3])

    def test_interval_costs_are_applied_to_macro_transition(self):
        state = physical_state(1)
        costs = StackIntervalCosts(
            collision=torch.tensor([0.2]),
            bottom_disturbance=torch.tensor([0.4]),
            humanoid_fall=torch.tensor([1.0]),
            elapsed_seconds=torch.tensor([0.5]),
            invalid_plan=torch.tensor([1.0]),
            unsafe_plan=torch.tensor([1.0]),
        )
        config = StackRewardConfig()
        result = compute_stack_planner_reward(state, state, costs, config)
        expected = (
            -config.collision_weight * 0.2
            - config.bottom_disturbance_weight * 0.4
            - config.humanoid_fall_penalty
            - config.time_penalty_per_second * 0.5
            - config.invalid_plan_penalty
            - config.unsafe_plan_penalty
        )
        self.assertAlmostEqual(result["total"].item(), expected, places=6)
        self.assertAlmostEqual(
            result["humanoid_fall_penalty"].item(),
            -config.humanoid_fall_penalty,
        )

    def test_fall_is_a_bounded_one_shot_interval_flag(self):
        state = physical_state(1)
        costs = zero_costs(1)
        costs.humanoid_fall[:] = 1.0
        result = compute_stack_planner_reward(state, state, costs)
        self.assertEqual(result["humanoid_fall_penalty"].item(), -3.0)

        costs.humanoid_fall[:] = 2.0
        with self.assertRaises(ValueError):
            compute_stack_planner_reward(state, state, costs)

    def test_success_bonus_is_paid_only_on_transition(self):
        before = physical_state(1)
        after = replace(before, stack_success=torch.ones(1))
        first = compute_stack_planner_reward(before, after, zero_costs(1))
        repeated = compute_stack_planner_reward(after, after, zero_costs(1))
        self.assertAlmostEqual(first["success_bonus"].item(), 5.0)
        self.assertEqual(repeated["success_bonus"].item(), 0.0)

    def test_contract_contains_no_gt_path_or_virtual_box(self):
        names = {item.name for item in fields(StackPhysicalState)}
        forbidden = {"gt_path", "retreat_direction", "virtual_box", "path_error"}
        self.assertTrue(names.isdisjoint(forbidden))

    def test_invalid_config_and_state_fail_closed(self):
        with self.assertRaises(ValueError):
            StackRewardConfig(clearance_start=1.0, clearance_done=1.0)
        state = physical_state(1)
        state.clearance[:] = -1.0
        with self.assertRaises(ValueError):
            stack_task_potential(state)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from carry_planner.analytic_loss import carry_analytic_collision_loss
from carry_planner.regularization import carry_path_regularization
from carry_planner.validity_debug import carry_plan_validity_debug
from coordinator.tests.common import make_state
from stack_planner.history import StackHistoryBuffer
from stack_planner.model import StackPlannerConfig, StackTrajectoryPlanner
from stack_planner.policy import StackPlannerActorCritic


def suffix_model():
    model = StackTrajectoryPlanner(StackPlannerConfig(
        candidates=1,
        plain_carry=True,
        retreat_delta_scale=0.5,
        carry_suffix_replan=True,
    )).eval()
    with torch.no_grad():
        model.heads.paths[0][-1].weight.zero_()
        model.heads.paths[0][-1].bias.zero_()
    return model


class CarrySuffixPathTest(unittest.TestCase):
    def test_unequal_legs_use_dynamic_exact_box_anchor(self):
        state = make_state(batch=1)
        state.root_xy[:] = torch.tensor([[[0.0, 0.0], [0.0, 1.0]]])
        state.box_xyz[..., :2] = torch.tensor([[[8.0, 0.0], [2.0, 1.0]]])
        state.goal_xy[:] = torch.tensor([[[9.0, 0.0], [8.0, 1.0]]])
        output = suffix_model()(state)
        index = output["box_index"][:, 0]
        self.assertGreater(int(index[0, 0]), 16)
        self.assertLess(int(index[0, 1]), 16)
        path = output["path_world"][:, 0]
        gather = index[..., None, None].expand(-1, -1, 1, 2)
        sampled_box = path.gather(2, gather).squeeze(2)
        self.assertTrue(torch.allclose(
            sampled_box, state.box_xyz[..., :2], atol=1e-5,
        ))
        self.assertTrue(torch.allclose(path[..., 0, :], state.root_xy, atol=1e-5))
        self.assertTrue(torch.allclose(path[..., -1, :], state.goal_xy, atol=1e-5))

    def test_held_agent_generates_only_current_to_goal_suffix(self):
        state = make_state(batch=1)
        state.root_xy[:] = torch.tensor([[[3.0, 2.0], [0.0, 1.0]]])
        state.box_xyz[..., :2] = torch.tensor([[[3.2, 2.0], [2.0, 1.0]]])
        state.goal_xy[:] = torch.tensor([[[7.0, 2.0], [8.0, 1.0]]])
        state.held[:, 0] = 1.0
        output = suffix_model()(state)
        self.assertEqual(int(output["box_index"][0, 0, 0]), -1)
        path = output["path_world"][0, 0, 0]
        self.assertTrue(torch.allclose(path[0], state.root_xy[0, 0], atol=1e-5))
        self.assertTrue(torch.allclose(path[-1], state.goal_xy[0, 0], atol=1e-5))

    def test_dynamic_pickup_corner_is_validated_at_actual_index(self):
        state = make_state(batch=1)
        state.root_xy[:] = torch.tensor([[[-4.0, 0.0], [0.0, -2.0]]])
        state.box_xyz[..., :2] = torch.tensor([[[0.0, 0.0], [0.0, -1.0]]])
        state.goal_xy[:] = torch.tensor([[[0.0, 1.0], [0.0, 3.0]]])
        output = suffix_model()(state)
        path = output["path_world"][:, 0]
        speed = output["speed"][:, 0]
        debug = carry_plan_validity_debug(
            state, path, speed,
            box_index=output["box_index"][:, 0],
            suffix_replan=True,
        )
        self.assertTrue(bool(debug["anchors"].all()))
        self.assertTrue(bool(debug["curve"].all()))
        self.assertGreater(int(output["box_index"][0, 0, 0]), 16)

    def test_policy_and_auxiliary_losses_preserve_suffix_metadata(self):
        state = make_state(batch=2)
        model = suffix_model()
        policy = StackPlannerActorCritic(model)
        history = StackHistoryBuffer(2, 1, "cpu")
        observation = history.observe(state)
        output, _, _, _ = policy.sample_all(observation)
        self.assertTrue(output["suffix_replan"])
        self.assertEqual(output["box_index"].shape, (2, 1, 2))
        self.assertEqual(output["mean_box_index"].shape, (2, 1, 2))
        analytic = carry_analytic_collision_loss(
            output, state, observation.path_progress,
        )
        regularization = carry_path_regularization(
            output, observation, analytic["per_sample_loss"],
        )
        self.assertTrue(torch.isfinite(analytic["loss"]))
        self.assertTrue(torch.isfinite(regularization["excess_length_loss"]))
        self.assertEqual(float(regularization["consistency_loss"]), 0.0)

    def test_suffix_path_gradient_reaches_all_sparse_controls(self):
        state = make_state(batch=2)
        model = suffix_model().train()
        output = model(state)
        loss = output["path_world"][..., 1:-1, :].square().mean()
        loss.backward()
        gradient = model.heads.paths[0][-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

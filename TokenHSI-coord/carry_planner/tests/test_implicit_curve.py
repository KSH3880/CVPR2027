import tempfile
import unittest
from pathlib import Path

import torch

from carry_planner.analytic_loss import carry_analytic_collision_loss
from carry_planner.validity_debug import carry_plan_validity_debug
from coordinator.tests.common import make_state
from stack_planner.carry_implicit import decode_carry_implicit_suffix
from stack_planner.checkpoint import load_stack_checkpoint, save_stack_checkpoint
from stack_planner.model import StackPlannerConfig, StackTrajectoryPlanner
from stack_planner.policy import StackPlannerActorCritic


def implicit_model():
    return StackTrajectoryPlanner(StackPlannerConfig(
        candidates=1, plain_carry=True, carry_suffix_replan=True,
        carry_implicit_curve=True, retreat_delta_scale=0.5,
    ))


class CarryImplicitCurveTest(unittest.TestCase):
    def test_initial_sampled_and_mean_paths_pass_hard_validation(self):
        for held in (False, True):
            torch.manual_seed(7)
            state = make_state(batch=16, held=held)
            policy = StackPlannerActorCritic(implicit_model().eval())
            with torch.no_grad():
                output, _, _, _ = policy.sample_all(state)
            for prefix in ("", "mean_"):
                debug = carry_plan_validity_debug(
                    state, output[prefix + "path_world"][:, 0],
                    output[prefix + "speed"][:, 0],
                    box_index=output[prefix + "box_index"][:, 0],
                    suffix_replan=True,
                )
                for predicate in (
                    "finite", "anchors", "buffer", "speed", "curve",
                ):
                    self.assertTrue(
                        bool(debug[predicate].all()),
                        f"held={held}, prefix={prefix}, failed={predicate}",
                    )

    def test_unequal_legs_preserve_exact_dynamic_box_anchor(self):
        state = make_state(batch=1)
        state.root_xy[:] = torch.tensor([[[0.0, 0.0], [0.0, 1.0]]])
        state.box_xyz[..., :2] = torch.tensor([[[8.0, 0.0], [2.0, 1.0]]])
        state.goal_xy[:] = torch.tensor([[[9.0, 0.0], [8.0, 1.0]]])
        output = implicit_model().eval()(state)
        index = output["box_index"][:, 0]
        self.assertGreater(int(index[0, 0]), 16)
        self.assertLess(int(index[0, 1]), 16)
        path = output["path_world"][:, 0]
        box = path.gather(
            2, index[..., None, None].expand(-1, -1, 1, 2),
        ).squeeze(2)
        self.assertTrue(torch.allclose(box, state.box_xyz[..., :2], atol=1e-5))
        self.assertTrue(torch.allclose(path[..., 0, :], state.root_xy, atol=1e-5))
        self.assertTrue(torch.allclose(path[..., -1, :], state.goal_xy, atol=1e-5))
        debug = carry_plan_validity_debug(
            state, path, output["speed"][:, 0],
            box_index=index, suffix_replan=True,
        )
        self.assertTrue(bool(debug["anchors"].all()))
        self.assertTrue(bool(torch.isfinite(path).all()))

    def test_held_curve_can_bend_more_than_short_direct_distance(self):
        root = torch.zeros(1, 1, 2, 2)
        goal = torch.tensor([[[[0.4, 0.0], [0.4, 0.0]]]])
        held = torch.ones(1, 1, 2)
        path_raw = torch.zeros(1, 1, 2, 4, 2)
        speed_raw = torch.zeros(1, 1, 2, 7)

        def wide_bend(features):
            coefficients = features.new_zeros(*features.shape[:-1], 15)
            coefficients[..., 4] = 1.0
            return coefficients

        path, speed, box_index = decode_carry_implicit_suffix(
            root, root, goal, held, path_raw, speed_raw, 4.0, wide_bend,
        )
        self.assertGreater(float(path[..., 1:-1, 1].max()), 1.0)
        self.assertTrue(torch.equal(box_index, torch.full_like(box_index, -1)))
        self.assertEqual(path.shape, (1, 1, 2, 33, 2))
        self.assertEqual(speed.shape, (1, 1, 2, 33))

    def test_policy_and_analytic_path_gradients_reach_latent_head(self):
        state = make_state(batch=2)
        model = implicit_model().train()
        policy = StackPlannerActorCritic(model)
        output, action, _, _ = policy.sample_all(state)
        self.assertEqual(action.shape[-1], policy.action_dim)
        self.assertEqual(output["mean_path_world"].shape, (2, 1, 2, 33, 2))
        analytic = carry_analytic_collision_loss(
            output, state, torch.zeros(2, 2),
        )
        self.assertTrue(torch.isfinite(analytic["loss"]))
        loss = (
            output["mean_path_world"][..., 1:-1, :].square().mean()
            + analytic["loss"]
        )
        loss.backward()
        self.assertGreater(
            float(model.heads.paths[0][-1].weight.grad.abs().sum()), 0.0,
        )
        self.assertGreater(
            float(model.heads.implicit_coefficients[-1].weight.grad.abs().sum()), 0.0,
        )

    def test_checkpoint_round_trip_and_legacy_sparse_loading(self):
        model = implicit_model().eval()
        state = make_state(batch=1)
        with torch.inference_mode():
            before = model(state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "implicit.pth"
            save_stack_checkpoint(path, model)
            loaded, payload = load_stack_checkpoint(path)
            self.assertTrue(payload["carry_implicit_curve"])
            self.assertTrue(loaded.config.carry_implicit_curve)
            with torch.inference_mode():
                after = loaded(state)
            self.assertTrue(torch.equal(
                before["path_world"], after["path_world"],
            ))
            legacy = StackTrajectoryPlanner(StackPlannerConfig(
                plain_carry=True, carry_suffix_replan=True,
            )).eval()
            save_stack_checkpoint(path, legacy)
            older_payload = torch.load(path, weights_only=False)
            del older_payload["model_config"]["carry_implicit_curve"]
            torch.save(older_payload, path)
            loaded_legacy, _ = load_stack_checkpoint(path)
            self.assertFalse(loaded_legacy.config.carry_implicit_curve)

    def test_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            StackPlannerConfig(carry_implicit_curve=True)


if __name__ == "__main__":
    unittest.main()

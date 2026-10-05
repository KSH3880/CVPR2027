"""PPO for plain simultaneous Carry with the low-level ms18 policy frozen."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

# Isaac Gym must be imported before torch.
from isaacgym import gymapi as _gymapi  # noqa: F401

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT.parent
TOKENHSI_ROOT = REPO_ROOT / "tokenhsi"
if str(TOKENHSI_ROOT) not in sys.path:
    sys.path.insert(0, str(TOKENHSI_ROOT))

import run as tokenhsi_run  # noqa: E402
import utils.parse_task as task_registry  # noqa: E402
from carry_planner.env_adapter import HumanoidMACarryPlannerTrain  # noqa: E402
from carry_planner.episode_collision import EpisodeCollisionTracker  # noqa: E402
from carry_planner.analytic_loss import carry_analytic_collision_loss  # noqa: E402
from carry_planner.regularization import carry_path_regularization  # noqa: E402
from carry_planner.reward import apply_invalid_plan_penalty  # noqa: E402
from carry_planner.validity_debug import carry_plan_validity_debug  # noqa: E402
from coordinator.schema import AGENTS  # noqa: E402
from carry_planner.tensorboard_metrics import make_writer, write_scalars  # noqa: E402
from stack_planner.checkpoint import (  # noqa: E402
    load_stack_checkpoint, save_stack_checkpoint,
)
from stack_planner.history import (  # noqa: E402
    StackHistoryBuffer, flatten_observations,
)
from stack_planner.model import StackPlannerConfig, StackTrajectoryPlanner  # noqa: E402
from stack_planner.policy import StackPlannerActorCritic  # noqa: E402
from utils.config import get_args, load_cfg, set_np_formatting, set_seed  # noqa: E402

task_registry.HumanoidMACarryPlannerTrain = HumanoidMACarryPlannerTrain


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _refresh_obs(player):
    player.env.task._compute_observations()
    return torch.clamp(
        player.env.task.obs_buf, -player.env.clip_obs, player.env.clip_obs,
    ).to(player.device)


@torch.no_grad()
def _macro_step(player, low_steps: int, collision_coef: float,
                progress_coef: float, episode_collisions: EpisodeCollisionTracker):
    task = player.env.task
    n = task.num_envs
    start_distance = task.planner_task_distance()
    reward_sum = torch.zeros(n, device=task.device)
    collision_sum = torch.zeros(n, device=task.device)
    collision_steps = torch.zeros(n, device=task.device)
    collided_episodes = torch.zeros((), device=task.device)
    completed_episodes = torch.zeros((), device=task.device)
    component_sum = {
        name: torch.zeros(n, device=task.device)
        for name in ("agent_agent", "agent_box", "box_box")
    }
    active = torch.ones(n, dtype=torch.bool, device=task.device)
    done_env = torch.zeros_like(active)
    end_distance = start_distance.clone()
    obs = _refresh_obs(player)
    player.get_batch_size(obs, 1)
    executed = torch.zeros(n, device=task.device)
    for _ in range(low_steps):
        action = player.get_action({"obs": obs}, is_determenistic=True)
        obs, reward, done_rows, _ = player.env.step(action)
        reward_env = reward.reshape(n, AGENTS).mean(dim=1)
        state = task.planner_state()
        distance = task.planner_task_distance(state)
        collision = task.planner_collision_terms(state)
        reward_sum += torch.where(active, reward_env, torch.zeros_like(reward_env))
        collision_sum += torch.where(
            active, collision["total"], torch.zeros_like(collision["total"]),
        )
        collision_hit = (collision["total"] > 0) & active
        collision_steps += collision_hit.float()
        for name in component_sum:
            component_sum[name] += torch.where(
                active, collision[name], torch.zeros_like(collision[name]),
            )
        executed += active.float()
        end_distance = torch.where(active, distance, end_distance)
        just_done = done_rows.reshape(n, AGENTS).any(dim=1) & active
        collided_now, completed_now = episode_collisions.update(
            collision_hit, just_done,
        )
        collided_episodes += collided_now
        completed_episodes += completed_now
        if just_done.any():
            done_env |= just_done
            active &= ~just_done
            rows = torch.nonzero(
                just_done.repeat_interleave(AGENTS), as_tuple=False,
            ).squeeze(-1)
            obs = player.env.reset(rows)
    denom = executed.clamp(min=1.0)
    progress = start_distance - end_distance
    collision_mean = collision_sum / denom
    reward = (
        reward_sum / denom + progress_coef * progress
        - collision_coef * collision_mean - 0.01
    )
    diagnostics = {
        "base_reward": reward_sum.sum(),
        "progress": progress.sum(),
        "collision_cost": collision_sum.sum(),
        "collision_steps": collision_steps.sum(),
        "executed_steps": executed.sum(),
        "collision_episodes": collided_episodes,
        "completed_episodes": completed_episodes,
        "samples": executed.gt(0).float().sum(),
    }
    for name, value in component_sum.items():
        diagnostics[f"collision_{name}_cost"] = value.sum()
        diagnostics[f"collision_{name}_steps"] = (value > 0).float().sum()
    return reward, done_env, diagnostics


def _make_player(args, cfg, cfg_train):
    tokenhsi_run.args = args
    tokenhsi_run.cfg = cfg
    tokenhsi_run.cfg_train = cfg_train
    runner = tokenhsi_run.build_alg_runner(tokenhsi_run.RLGPUAlgoObserver())
    runner.load(cfg_train)
    runner.reset()
    player = runner.create_player()
    if args.checkpoint == "Base":
        raise ValueError("--checkpoint must point to the frozen ms18 executor")
    player.restore(args.checkpoint)
    player.model.eval()
    for parameter in player.model.parameters():
        parameter.requires_grad_(False)
    if not isinstance(player.env.task, HumanoidMACarryPlannerTrain):
        raise ValueError("--task must be HumanoidMACarryPlannerTrain")
    return player


def _ppo_update(policy, optimizer, observations, actions, old_log_prob,
                returns, advantages, epochs, minibatch, clip_ratio,
                value_coef, entropy_coef, smoothness_coef,
                speed_smoothness_coef, analytic_collision_coef,
                analytic_curvature_coef, analytic_focus_steps,
                analytic_time_uncertainty, analytic_time_samples,
                consistency_coef, excess_length_coef, free_detour_ratio,
                absolute_length_slack, length_huber_beta, direction_coef,
                direction_lookahead, direction_free_angle_deg,
                direction_min_speed, direction_full_speed,
                pickup_fallback_weight, regularization_scale):
    total = actions.shape[0]
    sums = {
        "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
        "path_smoothness_loss": 0.0, "speed_smoothness_loss": 0.0,
        "analytic_collision_loss": 0.0,
        "analytic_curvature_loss": 0.0,
        "analytic_active_fraction": 0.0,
        "analytic_min_hh": 0.0,
        "analytic_min_bb_margin": 0.0,
        "analytic_min_hb_margin": 0.0,
        "replan_consistency_loss": 0.0,
        "excess_length_loss": 0.0,
        "direction_loss": 0.0,
        "weighted_consistency_loss": 0.0,
        "weighted_excess_length_loss": 0.0,
        "weighted_direction_loss": 0.0,
        "path_regularization_safe_weight": 0.0,
        "mean_replan_displacement": 0.0,
        "mean_future_length_ratio": 0.0,
        "mean_future_excess_m": 0.0,
        "max_future_excess_m": 0.0,
        "mean_direction_error_deg": 0.0,
        "direction_active_fraction": 0.0,
        "direction_fallback_fraction": 0.0,
    }
    updates = 0
    kl_before_sum = kl_after_sum = 0.0
    kl_before_max = kl_after_max = 0.0
    for _ in range(epochs):
        for index in torch.randperm(total, device=actions.device).split(minibatch):
            observation = observations.index(index)
            log_prob, entropy, value, _ = policy.evaluate(
                observation, actions[index],
            )
            logratio = log_prob - old_log_prob[index]
            ratio = logratio.exp()
            with torch.no_grad():
                kl_before = float((torch.expm1(logratio.float()) - logratio.float()).mean())
            objective = torch.minimum(
                ratio * advantages[index],
                ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
                * advantages[index],
            )
            policy_loss = -objective.mean()
            value_loss = F.mse_loss(value, returns[index])
            entropy_mean = entropy.mean()
            mean_output = policy.all_mean_outputs(observation)
            regularization = policy.diversity(
                observation, output=mean_output,
            )
            analytic = carry_analytic_collision_loss(
                mean_output, observation.state,
                observation.path_progress,
                focus_steps=analytic_focus_steps,
                time_uncertainty=analytic_time_uncertainty,
                time_samples=analytic_time_samples,
            )
            path_regularization = carry_path_regularization(
                mean_output, observation,
                analytic["per_sample_loss"],
                free_detour_ratio=free_detour_ratio,
                absolute_length_slack=absolute_length_slack,
                length_huber_beta=length_huber_beta,
                direction_lookahead=direction_lookahead,
                direction_free_angle_deg=direction_free_angle_deg,
                direction_min_speed=direction_min_speed,
                direction_full_speed=direction_full_speed,
                pickup_fallback_weight=pickup_fallback_weight,
            )
            weighted_consistency = (
                regularization_scale * consistency_coef
                * path_regularization["consistency_loss"]
            )
            weighted_excess_length = (
                regularization_scale * excess_length_coef
                * path_regularization["excess_length_loss"]
            )
            weighted_direction = (
                regularization_scale * direction_coef
                * path_regularization["direction_loss"]
            )
            loss = (
                policy_loss + value_coef * value_loss
                - entropy_coef * entropy_mean
            )
            # Disabled auxiliaries must not enter backward, including 0 * NaN.
            for coefficient, auxiliary in (
                (smoothness_coef, regularization["smoothness_loss"]),
                (speed_smoothness_coef, regularization["speed_smoothness_loss"]),
                (analytic_collision_coef, analytic["loss"]),
                (analytic_curvature_coef, analytic["curvature_loss"]),
                (regularization_scale * consistency_coef, path_regularization["consistency_loss"]),
                (regularization_scale * excess_length_coef, path_regularization["excess_length_loss"]),
                (regularization_scale * direction_coef, path_regularization["direction_loss"]),
            ):
                if coefficient != 0:
                    loss = loss + coefficient * auxiliary
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                log_prob_after, _, _, _ = policy.evaluate(observation, actions[index])
                logratio_after = log_prob_after.float() - old_log_prob[index].float()
                kl_after = float((torch.expm1(logratio_after) - logratio_after).mean())
            kl_before_sum += kl_before
            kl_after_sum += kl_after
            kl_before_max = max(kl_before_max, kl_before)
            kl_after_max = max(kl_after_max, kl_after)
            sums["policy_loss"] += float(policy_loss.detach())
            sums["value_loss"] += float(value_loss.detach())
            sums["entropy"] += float(entropy_mean.detach())
            sums["path_smoothness_loss"] += float(
                regularization["smoothness_loss"].detach()
            )
            sums["speed_smoothness_loss"] += float(
                regularization["speed_smoothness_loss"].detach()
            )
            sums["analytic_collision_loss"] += float(
                analytic["loss"].detach()
            )
            sums["analytic_curvature_loss"] += float(
                analytic["curvature_loss"].detach()
            )
            sums["analytic_active_fraction"] += float(
                analytic["active_fraction"].detach()
            )
            sums["replan_consistency_loss"] += float(
                path_regularization["consistency_loss"].detach()
            )
            sums["excess_length_loss"] += float(
                path_regularization["excess_length_loss"].detach()
            )
            sums["direction_loss"] += float(
                path_regularization["direction_loss"].detach()
            )
            sums["weighted_consistency_loss"] += float(
                weighted_consistency.detach()
            )
            sums["weighted_excess_length_loss"] += float(
                weighted_excess_length.detach()
            )
            sums["weighted_direction_loss"] += float(
                weighted_direction.detach()
            )
            for name in (
                "safe_weight", "mean_replan_displacement",
                "mean_future_length_ratio", "mean_future_excess_m",
                "mean_direction_error_deg", "direction_active_fraction",
                "direction_fallback_fraction",
            ):
                sums[
                    "path_regularization_safe_weight"
                    if name == "safe_weight" else name
                ] += float(path_regularization[name])
            sums["max_future_excess_m"] = max(
                sums["max_future_excess_m"],
                float(path_regularization["max_future_excess_m"]),
            )
            for name in ("min_hh", "min_bb_margin", "min_hb_margin"):
                sums[f"analytic_{name}"] += float(analytic[name])
            updates += 1
    averaged = {
        key: value / max(updates, 1) for key, value in sums.items()
    }
    averaged["max_future_excess_m"] = sums["max_future_excess_m"]
    averaged.update(
        approx_kl_before_mean=kl_before_sum / max(updates, 1),
        approx_kl_before_max=kl_before_max,
        approx_kl_after_mean=kl_after_sum / max(updates, 1),
        approx_kl_after_max=kl_after_max,
        ppo_updates=updates,
    )
    return averaged


def main():
    set_np_formatting()
    args = get_args()
    os.environ["COORD_PROVIDER"] = "external"
    cfg, cfg_train, _ = load_cfg(args)
    seed = set_seed(
        cfg_train["params"].get("seed", 0),
        cfg_train["params"].get("torch_deterministic", False),
    )
    cfg_train["params"]["seed"] = seed
    cfg_train["params"]["config"]["seed"] = seed
    cfg_train["params"]["config"]["train_dir"] = args.output_path
    if args.motion_file:
        cfg["env"]["motion_file"] = args.motion_file
    player = _make_player(args, cfg, cfg_train)
    task = player.env.task
    device = torch.device(player.device)

    history_steps = _env_int("CARRY_PLANNER_HISTORY_STEPS", 4)
    delta_scale = _env_float("CARRY_PLANNER_DELTA_SCALE", 0.5)
    control_scale = _env_float("CARRY_PLANNER_CONTROL_SCALE", 4.0)
    path_update_alpha = _env_float("CARRY_PLANNER_PATH_UPDATE_ALPHA", 0.5)
    implicit_curve = _env_int("CARRY_PLANNER_IMPLICIT_CURVE", 1)
    if implicit_curve not in (0, 1):
        raise ValueError("CARRY_PLANNER_IMPLICIT_CURVE must be 0 or 1")
    implicit_leg_scale = _env_int("CARRY_PLANNER_IMPLICIT_LEG_SCALE", implicit_curve)
    if implicit_leg_scale not in (0, 1):
        raise ValueError("CARRY_PLANNER_IMPLICIT_LEG_SCALE must be 0 or 1")
    if implicit_leg_scale and not implicit_curve:
        raise ValueError("implicit leg scale requires implicit curve")
    init = os.environ.get("CARRY_PLANNER_INIT", "")
    payload = None
    if init:
        planner, payload = load_stack_checkpoint(init, device)
        expected = (
            history_steps, delta_scale, delta_scale,
            path_update_alpha, True, control_scale, True,
            bool(implicit_curve), bool(implicit_leg_scale),
        )
        actual = (
            planner.config.history_steps, planner.config.delta_scale,
            planner.config.retreat_delta_scale,
            planner.config.path_update_alpha, planner.config.plain_carry,
            planner.config.carry_control_scale,
            planner.config.carry_suffix_replan,
            planner.config.carry_implicit_curve,
            planner.config.carry_implicit_leg_scale,
        )
        if actual != expected or planner.config.candidates != 1:
            raise ValueError(
                f"carry planner init config mismatch: expected={expected}, actual={actual}"
            )
    else:
        planner = StackTrajectoryPlanner(StackPlannerConfig(
            candidates=1,
            history_steps=history_steps,
            delta_scale=delta_scale,
            retreat_delta_scale=delta_scale,
            path_update_alpha=path_update_alpha,
            plain_carry=True,
            carry_control_scale=control_scale,
            carry_suffix_replan=True,
            carry_implicit_curve=bool(implicit_curve),
            carry_implicit_leg_scale=bool(implicit_leg_scale),
        )).to(device)
    history = StackHistoryBuffer(task.num_envs, history_steps, device)
    policy = StackPlannerActorCritic(
        planner,
        point_std=_env_float("CARRY_PLANNER_DELTA_STD", 0.10),
        endpoint_std=_env_float("CARRY_PLANNER_ENDPOINT_STD", 0.03),
        anchor_std=_env_float("CARRY_PLANNER_ANCHOR_STD", 0.03),
        speed_std=_env_float("CARRY_PLANNER_SPEED_STD", 0.20),
    ).to(device)
    if payload and payload.get("extras", {}).get("action_log_std") is not None:
        policy.action_log_std.data.copy_(
            payload["extras"]["action_log_std"].to(device)
        )
    learning_rate = _env_float("CARRY_PLANNER_LR", 3e-4)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    if payload and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
        # Restore Adam moments, then apply this run's requested learning rate.
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    policy.train()

    iterations = _env_int("CARRY_PLANNER_ITERS", 200)
    horizon = _env_int("CARRY_PLANNER_HORIZON", 32)
    low_steps = _env_int("CARRY_PLANNER_LOW_STEPS", 12)
    gamma = _env_float("CARRY_PLANNER_GAMMA", 0.99)
    collision_coef = _env_float("CARRY_PLANNER_COLLISION_COEF", 5.0)
    progress_coef = _env_float("CARRY_PLANNER_PROGRESS_COEF", 2.0)
    ppo_epochs = _env_int("CARRY_PLANNER_PPO_EPOCHS", 3)
    minibatch = _env_int("CARRY_PLANNER_MINIBATCH", 512)
    smoothness_coef = _env_float("CARRY_PLANNER_SMOOTHNESS_COEF", 10.0)
    speed_smoothness_coef = _env_float(
        "CARRY_PLANNER_SPEED_SMOOTHNESS_COEF", 1.0,
    )
    analytic_collision_coef = _env_float(
        "CARRY_PLANNER_ANALYTIC_COLLISION_COEF", 1.0,
    )
    analytic_curvature_coef = _env_float(
        "CARRY_PLANNER_ANALYTIC_CURVATURE_COEF", 20.0,
    )
    invalid_plan_coef = _env_float(
        "CARRY_PLANNER_INVALID_PLAN_COEF", 0.25,
    )
    analytic_focus_steps = _env_int(
        "CARRY_PLANNER_ANALYTIC_FOCUS_STEPS", 8,
    )
    analytic_time_uncertainty = _env_float(
        "CARRY_PLANNER_ANALYTIC_TIME_UNCERTAINTY", 1.5,
    )
    analytic_time_samples = _env_int(
        "CARRY_PLANNER_ANALYTIC_TIME_SAMPLES", 7,
    )
    consistency_coef = _env_float(
        "CARRY_PLANNER_REPLAN_CONSISTENCY_COEF", 0.05,
    )
    excess_length_coef = _env_float(
        "CARRY_PLANNER_EXCESS_LENGTH_COEF", 0.05,
    )
    free_detour_ratio = _env_float(
        "CARRY_PLANNER_FREE_DETOUR_RATIO", 1.15,
    )
    absolute_length_slack = _env_float(
        "CARRY_PLANNER_ABSOLUTE_LENGTH_SLACK", 0.25,
    )
    length_huber_beta = _env_float(
        "CARRY_PLANNER_LENGTH_HUBER_BETA", 0.50,
    )
    direction_coef = _env_float(
        "CARRY_PLANNER_DIRECTION_COEF", 0.10,
    )
    direction_lookahead = _env_float(
        "CARRY_PLANNER_DIRECTION_LOOKAHEAD", 0.50,
    )
    direction_free_angle_deg = _env_float(
        "CARRY_PLANNER_DIRECTION_FREE_ANGLE_DEG", 15.0,
    )
    direction_min_speed = _env_float(
        "CARRY_PLANNER_DIRECTION_MIN_SPEED", 0.20,
    )
    direction_full_speed = _env_float(
        "CARRY_PLANNER_DIRECTION_FULL_SPEED", 0.80,
    )
    pickup_fallback_weight = _env_float(
        "CARRY_PLANNER_PICKUP_DIRECTION_FALLBACK_WEIGHT", 0.50,
    )
    regularization_warmup = _env_int(
        "CARRY_PLANNER_PATH_REGULARIZATION_WARMUP", 5,
    )
    if min(iterations, horizon, low_steps, ppo_epochs, minibatch) <= 0:
        raise ValueError("iteration/horizon/step/minibatch values must be positive")
    if (not math.isfinite(progress_coef) or progress_coef < 0
            or collision_coef < 0 or smoothness_coef < 0
            or speed_smoothness_coef < 0 or analytic_collision_coef < 0
            or invalid_plan_coef < 0 or consistency_coef < 0
            or excess_length_coef < 0 or direction_coef < 0):
        raise ValueError("reward and regularization coefficients must be non-negative")
    if analytic_curvature_coef < 0:
        raise ValueError("analytic curvature coefficient must be non-negative")
    if not 1 <= analytic_focus_steps <= 96:
        raise ValueError("CARRY_PLANNER_ANALYTIC_FOCUS_STEPS must be in [1, 96]")
    if analytic_time_uncertainty < 0:
        raise ValueError("analytic time uncertainty must be non-negative")
    if (analytic_time_samples < 1
            or (analytic_time_samples > 1 and analytic_time_samples % 2 == 0)):
        raise ValueError("analytic time samples must be one or an odd integer")
    if free_detour_ratio < 1.0:
        raise ValueError("free detour ratio must be at least one")
    if absolute_length_slack < 0:
        raise ValueError("absolute length slack must be non-negative")
    if length_huber_beta <= 0:
        raise ValueError("length Huber beta must be positive")
    if direction_lookahead <= 0:
        raise ValueError("direction lookahead must be positive")
    if not 0 <= direction_free_angle_deg < 180:
        raise ValueError("direction free angle must be in [0, 180)")
    if direction_min_speed < 0 or direction_full_speed <= direction_min_speed:
        raise ValueError("direction speed range must be increasing")
    if not 0 <= pickup_fallback_weight <= 1:
        raise ValueError("pickup direction fallback weight must be in [0, 1]")
    if regularization_warmup < 0:
        raise ValueError("path regularization warmup must be non-negative")

    output_dir = Path(os.environ.get(
        "CARRY_PLANNER_OUTPUT", str(WORKSPACE / "runs/carry_planner/default"),
    )).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    tensorboard_enabled = _env_int("CARRY_PLANNER_TENSORBOARD", 1)
    if tensorboard_enabled not in (0, 1):
        raise ValueError("CARRY_PLANNER_TENSORBOARD must be 0 or 1")
    tensorboard_writer = None
    if tensorboard_enabled:
        try:
            tensorboard_writer = make_writer(output_dir / "tensorboard")
        except RuntimeError as exc:
            print(f"[carry-planner-train] TensorBoard disabled: {exc}", flush=True)
        else:
            print(
                f"[carry-planner-train] TensorBoard: {output_dir / 'tensorboard'}",
                flush=True,
            )

    save_every = _env_int("CARRY_PLANNER_SAVE_EVERY", 5)
    _refresh_obs(player)
    print(
        f"[carry-planner-train] envs={task.num_envs} horizon={horizon} "
        f"low_steps={low_steps} history={history_steps} delta={delta_scale:g} "
        f"control_scale={control_scale:g} "
        f"path_alpha={path_update_alpha:g} suffix_replan=True dynamic_box=True "
        f"implicit_curve={implicit_curve} implicit_leg_scale={implicit_leg_scale} "
        f"randomize_agent_slots={int(task._carry_randomize_agent_slots)} "
        f"delta_std={policy.action_log_std[0, 0].exp().item():g} "
        f"lr={optimizer.param_groups[0]['lr']:g} "
        f"clip={_env_float('CARRY_PLANNER_CLIP', 0.2):g} "
        f"progress_coef={progress_coef:g} "
        f"collision_coef={collision_coef:g} "
        f"analytic_collision_coef={analytic_collision_coef:g} "
        f"analytic_curvature_coef={analytic_curvature_coef:g} "
        f"invalid_plan_coef={invalid_plan_coef:g} "
        f"analytic_focus_steps={analytic_focus_steps} "
        f"analytic_time_uncertainty={analytic_time_uncertainty:g} "
        f"analytic_time_samples={analytic_time_samples} "
        f"consistency_coef={consistency_coef:g} "
        f"excess_length_coef={excess_length_coef:g} "
        f"free_detour_ratio={free_detour_ratio:g} "
        f"absolute_length_slack={absolute_length_slack:g} "
        f"length_huber_beta={length_huber_beta:g} "
        f"direction_coef={direction_coef:g} "
        f"direction_lookahead={direction_lookahead:g} "
        f"direction_free_angle_deg={direction_free_angle_deg:g} "
        f"direction_speed_range={direction_min_speed:g}:{direction_full_speed:g} "
        f"pickup_direction_fallback_weight={pickup_fallback_weight:g} "
        f"path_regularization_warmup={regularization_warmup} "
        f"converge_prob={task._carry_converge_prob:g} "
        f"goal_margin={task._carry_goal_margin:g} frozen={args.checkpoint}",
        flush=True,
    )

    first = 1 if payload is None else int(payload.get("step", 0)) + 1
    previous_done = torch.ones(task.num_envs, dtype=torch.bool, device=device)
    episode_collisions = EpisodeCollisionTracker(task.num_envs, device)
    for iteration in range(first, first + iterations):
        observations = []
        actions: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        returns: List[torch.Tensor] = []
        advantages: List[torch.Tensor] = []
        rewards: List[torch.Tensor] = []
        dones: List[torch.Tensor] = []
        diag: Dict[str, float] = {}
        for _ in range(horizon):
            state = task.planner_state()
            observation = history.observe(
                state, reset_mask=previous_done, commit=True,
            )
            with torch.no_grad():
                output, packed, log_prob, value = policy.sample_all(observation)
                candidate_output = {
                    "path_world": output["path_world"],
                    "speed": output["speed"],
                    "box_index": output.get("box_index"),
                    "suffix_replan": output.get("suffix_replan", False),
                }
                valid = task.install_external_plan(candidate_output)
                validity_debug = task.last_plan_validity_debug()
                for name in (
                    "finite", "anchors", "buffer", "speed", "curve",
                    "zero_safe_curve", "zero_turn_false_reject",
                    "relevant_degenerate_turn",
                ):
                    key = f"plan_debug_{name}"
                    diag[key] = diag.get(key, 0.0) + float(
                        validity_debug[name].float().sum()
                    )
                for name in ("max_turn_deg", "path_length_m"):
                    key = f"plan_debug_{name}"
                    diag[key] = diag.get(key, 0.0) + float(
                        validity_debug[name].sum()
                    )
                mean_validity_debug = carry_plan_validity_debug(
                    state,
                    output["mean_path_world"][:, 0],
                    output["mean_speed"][:, 0],
                    box_index=output["mean_box_index"][:, 0],
                    suffix_replan=True,
                )
                mean_checks = torch.stack((
                    mean_validity_debug["finite"],
                    mean_validity_debug["anchors"],
                    mean_validity_debug["buffer"],
                    mean_validity_debug["speed"],
                    mean_validity_debug["curve"],
                ), dim=-1)
                diag["mean_plan_valid"] = diag.get(
                    "mean_plan_valid", 0.0,
                ) + float(mean_checks.all(dim=-1).float().sum())
                diag["mean_plan_curve"] = diag.get(
                    "mean_plan_curve", 0.0,
                ) + float(mean_validity_debug["curve"].float().sum())
                diag["mean_plan_max_turn_deg"] = diag.get(
                    "mean_plan_max_turn_deg", 0.0,
                ) + float(mean_validity_debug["max_turn_deg"].sum())
                diag["mean_plan_count"] = diag.get(
                    "mean_plan_count", 0.0,
                ) + mean_validity_debug["curve"].numel()
                diag["plan_debug_count"] = diag.get(
                    "plan_debug_count", 0.0,
                ) + valid.numel()
                sampled_path = output["path_world"][:, 0]
                base_path = output["base_path_world"]
                sampled_deviation = torch.linalg.vector_norm(
                    sampled_path - base_path, dim=-1,
                )
                mean_deviation = torch.linalg.vector_norm(
                    output["mean_path_world"][:, 0] - base_path, dim=-1,
                )
                diag["plan_valid"] = diag.get("plan_valid", 0.0) + float(
                    valid.float().sum()
                )
                diag["plan_count"] = diag.get("plan_count", 0.0) + valid.numel()
                diag["sample_path_deviation"] = diag.get(
                    "sample_path_deviation", 0.0,
                ) + float(sampled_deviation.sum())
                diag["mean_path_deviation"] = diag.get(
                    "mean_path_deviation", 0.0,
                ) + float(mean_deviation.sum())
                diag["path_point_count"] = diag.get(
                    "path_point_count", 0.0,
                ) + sampled_deviation.numel()
                diag["sample_path_max_deviation"] = max(
                    diag.get("sample_path_max_deviation", 0.0),
                    float(sampled_deviation.max()),
                )
                diag["mean_path_max_deviation"] = max(
                    diag.get("mean_path_max_deviation", 0.0),
                    float(mean_deviation.max()),
                )
                reward, done, macro_diag = _macro_step(
                    player, low_steps, collision_coef, progress_coef,
                    episode_collisions,
                )
                reward, invalid_penalty = apply_invalid_plan_penalty(
                    reward, valid, invalid_plan_coef,
                )
                diag["invalid_plan_penalty"] = diag.get(
                    "invalid_plan_penalty", 0.0,
                ) + float(invalid_penalty.sum())
                diag["invalid_plan_samples"] = diag.get(
                    "invalid_plan_samples", 0.0,
                ) + invalid_penalty.numel()
                mean_path = output["mean_path_world"][:, 0]
                commit = valid & ~done
                history.commit_path(
                    mean_path, update_mask=commit,
                    base_path_world=output["base_path_world"],
                    reset_progress_mask=commit[:, None].expand(-1, AGENTS),
                )
                next_state = task.planner_state()
                next_observation = history.observe(
                    next_state, reset_mask=done, commit=False,
                )
                next_value = policy.value(next_observation)
                target = reward + gamma * next_value * (~done).float()
            observations.append(observation.clone())
            actions.append(packed[:, 0])
            log_probs.append(log_prob[:, 0])
            returns.append(target.detach())
            advantages.append((target - value).detach())
            rewards.append(reward)
            dones.append(done)
            previous_done = done.detach().clone()
            for key, tensor in macro_diag.items():
                diag[key] = diag.get(key, 0.0) + float(tensor)

        flat_advantage = torch.cat(advantages)
        flat_advantage = (
            (flat_advantage - flat_advantage.mean())
            / flat_advantage.std().clamp(min=1e-6)
        )
        regularization_scale = (
            1.0 if regularization_warmup == 0 else
            min(float(iteration) / float(regularization_warmup), 1.0)
        )
        update = _ppo_update(
            policy, optimizer, flatten_observations(observations),
            torch.cat(actions), torch.cat(log_probs), torch.cat(returns),
            flat_advantage, ppo_epochs, minibatch,
            _env_float("CARRY_PLANNER_CLIP", 0.2),
            _env_float("CARRY_PLANNER_VALUE_COEF", 0.5),
            _env_float("CARRY_PLANNER_ENTROPY_COEF", 1e-4),
            smoothness_coef, speed_smoothness_coef,
            analytic_collision_coef, analytic_curvature_coef,
            analytic_focus_steps, analytic_time_uncertainty,
            analytic_time_samples, consistency_coef,
            excess_length_coef, free_detour_ratio,
            absolute_length_slack, length_huber_beta, direction_coef,
            direction_lookahead, direction_free_angle_deg,
            direction_min_speed, direction_full_speed,
            pickup_fallback_weight, regularization_scale,
        )

        def ratio(numerator, denominator):
            return diag.get(numerator, 0.0) / max(
                diag.get(denominator, 0.0), 1e-8,
            )

        metrics = {
            "iteration": iteration,
            "implicit_curve": implicit_curve,
            "implicit_leg_scale": implicit_leg_scale,
            "progress_coef": progress_coef,
            "reward": float(torch.stack(rewards).mean()),
            "done_rate": float(torch.stack(dones).float().mean()),
            "progress": ratio("progress", "samples"),
            "base_reward": ratio("base_reward", "executed_steps"),
            "collision_cost": ratio("collision_cost", "executed_steps"),
            "collision_ratio": ratio("collision_steps", "executed_steps"),
            "collision_step_fraction": ratio("collision_steps", "executed_steps"),
            "collision_episode_fraction": (
                ratio("collision_episodes", "completed_episodes")
                if diag.get("completed_episodes", 0) > 0 else float("nan")
            ),
            "completed_episodes": int(diag.get("completed_episodes", 0)),
            "collision_agent_agent_cost": ratio(
                "collision_agent_agent_cost", "executed_steps",
            ),
            "collision_agent_box_cost": ratio(
                "collision_agent_box_cost", "executed_steps",
            ),
            "collision_box_box_cost": ratio(
                "collision_box_box_cost", "executed_steps",
            ),
            "plan_valid_fraction": ratio("plan_valid", "plan_count"),
            "sample_plan_valid_fraction": ratio("plan_valid", "plan_count"),
            "plan_finite_fraction": ratio(
                "plan_debug_finite", "plan_debug_count",
            ),
            "plan_anchor_fraction": ratio(
                "plan_debug_anchors", "plan_debug_count",
            ),
            "plan_buffer_fraction": ratio(
                "plan_debug_buffer", "plan_debug_count",
            ),
            "plan_speed_fraction": ratio(
                "plan_debug_speed", "plan_debug_count",
            ),
            "plan_curve_fraction": ratio(
                "plan_debug_curve", "plan_debug_count",
            ),
            "sample_plan_curve_fraction": ratio(
                "plan_debug_curve", "plan_debug_count",
            ),
            "mean_plan_valid_fraction": ratio(
                "mean_plan_valid", "mean_plan_count",
            ),
            "mean_plan_curve_fraction": ratio(
                "mean_plan_curve", "mean_plan_count",
            ),
            "plan_zero_safe_curve_fraction": ratio(
                "plan_debug_zero_safe_curve", "plan_debug_count",
            ),
            "plan_zero_turn_false_reject_fraction": ratio(
                "plan_debug_zero_turn_false_reject", "plan_debug_count",
            ),
            "plan_degenerate_turn_fraction": ratio(
                "plan_debug_relevant_degenerate_turn", "plan_debug_count",
            ),
            "plan_mean_max_turn_deg": ratio(
                "plan_debug_max_turn_deg", "plan_debug_count",
            ),
            "sample_plan_max_turn_deg": ratio(
                "plan_debug_max_turn_deg", "plan_debug_count",
            ),
            "mean_plan_max_turn_deg": ratio(
                "mean_plan_max_turn_deg", "mean_plan_count",
            ),
            "plan_mean_length_m": ratio(
                "plan_debug_path_length_m", "plan_debug_count",
            ),
            "invalid_plan_penalty": ratio(
                "invalid_plan_penalty", "invalid_plan_samples",
            ),
            "sample_path_deviation": ratio(
                "sample_path_deviation", "path_point_count",
            ),
            "sample_path_max_deviation": diag.get(
                "sample_path_max_deviation", 0.0,
            ),
            "mean_path_deviation": ratio(
                "mean_path_deviation", "path_point_count",
            ),
            "mean_path_max_deviation": diag.get(
                "mean_path_max_deviation", 0.0,
            ),
            "path_delta_std": float(policy.action_log_std[0, 0].exp()),
            "converge_fraction": float(
                task._carry_converge_layout.float().mean()
            ),
            "path_regularization_scale": regularization_scale,
            **update,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        if tensorboard_writer is not None:
            write_scalars(tensorboard_writer, metrics)
            tensorboard_writer.flush()
        console_keys = (
            "iteration", "reward", "done_rate", "progress",
            "collision_episode_fraction", "completed_episodes",
            "collision_step_fraction", "sample_plan_valid_fraction",
            "sample_plan_curve_fraction", "mean_plan_curve_fraction",
            "sample_plan_max_turn_deg", "mean_plan_max_turn_deg",
            "invalid_plan_penalty", "path_delta_std",
            "sample_path_deviation", "mean_path_deviation",
            "analytic_collision_loss", "mean_future_excess_m",
            "mean_direction_error_deg",
            "approx_kl_before_max", "approx_kl_after_max", "ppo_updates",
        )
        print(
            "[carry-planner-train] " + " ".join(
                f"{key}={metrics[key]:.5g}" for key in console_keys
            ),
            flush=True,
        )
        if iteration % save_every == 0 or iteration == first + iterations - 1:
            save_stack_checkpoint(
                output_dir / f"planner_{iteration:06d}.pth",
                policy.planner, step=iteration, metrics=metrics,
                optimizer=optimizer,
                extras={
                    "action_log_std": policy.action_log_std.detach().cpu(),
                    "frozen_executor": str(Path(args.checkpoint).resolve()),
                    "planner_task": "plain_carry_collision_avoidance",
                    "implicit_curve": bool(implicit_curve),
                    "implicit_leg_scale": bool(implicit_leg_scale),
                    "progress_coef": progress_coef,
                    "collision_coef": collision_coef,
                    "analytic_collision_coef": analytic_collision_coef,
                    "analytic_curvature_coef": analytic_curvature_coef,
                    "invalid_plan_coef": invalid_plan_coef,
                    "analytic_focus_steps": analytic_focus_steps,
                    "analytic_time_uncertainty": analytic_time_uncertainty,
                    "analytic_time_samples": analytic_time_samples,
                    "replan_consistency_coef": consistency_coef,
                    "excess_length_coef": excess_length_coef,
                    "free_detour_ratio": free_detour_ratio,
                    "absolute_length_slack": absolute_length_slack,
                    "length_huber_beta": length_huber_beta,
                    "direction_coef": direction_coef,
                    "direction_lookahead": direction_lookahead,
                    "direction_free_angle_deg": direction_free_angle_deg,
                    "direction_min_speed": direction_min_speed,
                    "direction_full_speed": direction_full_speed,
                    "pickup_direction_fallback_weight": pickup_fallback_weight,
                    "path_regularization_warmup": regularization_warmup,
                    "commit_steps": low_steps,
                    "converge_probability": task._carry_converge_prob,
                    "goal_margin": task._carry_goal_margin,
                },
            )

    if tensorboard_writer is not None:
        tensorboard_writer.close()

if __name__ == "__main__":
    main()

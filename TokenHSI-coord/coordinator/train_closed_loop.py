"""High-level PPO for coordinators with the ms18 action policy frozen.

This module deliberately reuses TokenHSI's player only as a deterministic
low-level executor.  Its parameters are excluded from the optimizer.  One PPO
transition spans ``COORD_LOW_STEPS`` physics/action-policy steps.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Isaac Gym's binary loader requires this import to precede torch.
from isaacgym import gymapi as _gymapi  # noqa: F401

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENHSI_ROOT = REPO_ROOT / "tokenhsi"
if str(TOKENHSI_ROOT) not in sys.path:
    sys.path.insert(0, str(TOKENHSI_ROOT))

import run as tokenhsi_run  # noqa: E402
from coordinator.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from coordinator.c2_checkpoint import (  # noqa: E402
    c2_config,
    load_c2_checkpoint,
    save_c2_checkpoint,
)
from coordinator.c2_losses import (  # noqa: E402
    build_trajectory_consistency_target,
    compute_c2_auxiliary_loss,
)
from coordinator.losses import compute_auxiliary_loss  # noqa: E402
from coordinator.model import CoordinatorConfig, JointCoordinator  # noqa: E402
from coordinator.planner import apply_fixed_priority  # noqa: E402
from coordinator.policy import CoordinatorActorCritic  # noqa: E402
from coordinator.simple_checkpoint import (  # noqa: E402
    load_simple_checkpoint,
    save_simple_checkpoint,
)
from coordinator.simple_model import SimpleCoordinatorConfig, SimpleJointCoordinator  # noqa: E402
from coordinator.simple_policy import SimpleCoordinatorActorCritic  # noqa: E402
from coordinator.schema import AGENTS, CoordinatorState, STATE_KEYS  # noqa: E402
from utils.config import get_args, load_cfg, set_np_formatting, set_seed  # noqa: E402


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() not in {
        "", "0", "false", "no", "off"
    }


def _flatten_states(states: List[CoordinatorState]) -> CoordinatorState:
    return CoordinatorState(**{
        key: torch.cat([getattr(state, key) for state in states], dim=0)
        for key in STATE_KEYS
    })


def _flatten_consistency_targets(
    targets: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor] | None:
    if not targets:
        return None
    return {
        key: torch.cat([target[key] for target in targets], dim=0)
        for key in ("position", "speed", "valid")
    }


def _flatten_priority_agents(
    priority_agents: List[torch.Tensor],
) -> torch.Tensor | None:
    return torch.cat(priority_agents) if priority_agents else None


def _flatten_masks(masks: List[torch.Tensor]) -> torch.Tensor | None:
    return torch.cat(masks) if masks else None


def _task_distance(state: CoordinatorState) -> torch.Tensor:
    carrying = state.held >= 0.5
    source = torch.where(carrying[..., None], state.box_xyz[..., :2], state.root_xy)
    target = torch.where(carrying[..., None], state.goal_xy, state.box_xyz[..., :2])
    return (source - target).norm(dim=-1).mean(dim=1)


def _collision_violation(state: CoordinatorState) -> torch.Tensor:
    root, box = state.root_xy, state.box_xyz[..., :2]
    radius = 0.5 * state.box_size_xy.norm(dim=-1)
    hh = F.relu(1.0 - (root[:, 0] - root[:, 1]).norm(dim=-1)).square()
    bb_limit = radius.sum(dim=-1) + 0.15
    bb = F.relu(bb_limit - (box[:, 0] - box[:, 1]).norm(dim=-1)).square()
    hb01 = F.relu(0.35 + radius[:, 1] - (root[:, 0] - box[:, 1]).norm(dim=-1)).square()
    hb10 = F.relu(0.35 + radius[:, 0] - (root[:, 1] - box[:, 0]).norm(dim=-1)).square()
    return hh + bb + 0.5 * (hb01 + hb10)


def _refresh_obs(player) -> torch.Tensor:
    player.env.task._compute_observations()
    return torch.clamp(
        player.env.task.obs_buf, -player.env.clip_obs, player.env.clip_obs
    ).to(player.device)


@torch.no_grad()
def _macro_step(
    player, low_steps: int, collision_coef: float
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    task = player.env.task
    num_envs = task.num_envs
    start = task.coord_state()
    start_distance = _task_distance(start)
    reward_sum = torch.zeros(num_envs, device=task.device)
    collision_sum = torch.zeros(num_envs, device=task.device)
    active = torch.ones(num_envs, device=task.device, dtype=torch.bool)
    done_env = torch.zeros_like(active)
    end_distance = start_distance.clone()
    obs = _refresh_obs(player)
    # The stock rl_games player normally discovers this in Player.run().  The
    # coordinator owns the rollout loop, so initialise the same batch flag here
    # before the first deterministic executor action.
    player.get_batch_size(obs, 1)
    for _ in range(low_steps):
        # TransPlayer expects the same dictionary contract used by its normal
        # rollout loop.  ``obs`` is already a clipped tensor on player.device;
        # running obs_to_torch() again would wrap/reshape it a second time.
        action = player.get_action({"obs": obs}, is_determenistic=True)
        obs, reward, done_rows, _ = player.env.step(action)
        reward_env = reward.reshape(num_envs, AGENTS).mean(dim=1)
        current_state = task.coord_state()
        current_distance = _task_distance(current_state)
        collision = _collision_violation(current_state)
        reward_sum += torch.where(active, reward_env, torch.zeros_like(reward_env))
        collision_sum += torch.where(active, collision, torch.zeros_like(collision))
        end_distance = torch.where(active, current_distance, end_distance)
        just_done = done_rows.reshape(num_envs, AGENTS).any(dim=1) & active
        if just_done.any():
            done_env |= just_done
            active &= ~just_done
            row_ids = torch.nonzero(just_done.repeat_interleave(AGENTS), as_tuple=False)[:, 0]
            obs = player.env.reset(row_ids)
    # A completed environment is reset inside the macro step.  Use its terminal
    # distance captured above, never the unrelated state after reset.
    progress = start_distance - end_distance
    denom = float(max(low_steps, 1))
    reward = (
        reward_sum / denom + 2.0 * progress
        - collision_coef * collision_sum / denom - 0.01
    )
    diagnostics = {
        "base_reward": float((reward_sum / denom).mean().item()),
        "progress": float(progress.mean().item()),
        "collision": float((collision_sum / denom).mean().item()),
    }
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
        raise ValueError("--checkpoint must point to the frozen ms18 executor PTH")
    player.restore(args.checkpoint)
    player.model.eval()
    for parameter in player.model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in player.model.parameters()):
        raise RuntimeError("failed to freeze ms18 executor")
    if player.env.task.__class__.__name__ != "HumanoidMACoordCarry":
        raise ValueError("closed-loop trainer requires --task HumanoidMACoordCarry")
    if player.env.task._coord_provider != "external":
        raise ValueError("closed-loop trainer requires COORD_PROVIDER=external")
    return player


def _ppo_update(
    policy: CoordinatorActorCritic,
    optimizer: torch.optim.Optimizer,
    states: CoordinatorState,
    consistency_targets: Dict[str, torch.Tensor] | None,
    priority_agents: torch.Tensor | None,
    priority_choice_masks: torch.Tensor | None,
    initial_plan_masks: torch.Tensor | None,
    actions: torch.Tensor,
    old_log_prob: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    epochs: int,
    minibatch: int,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    aux_coef: float,
    model_kind: str,
    c2_peak_collision: bool,
    c2_human_clearance: float,
    c2_arrival_gap: bool,
    c2_arrival_time_margin: float,
    c2_arrival_gap_coef: float,
    c2_fixed_yield_agent1: bool,
    c2_bidirectional_target_gap: bool,
    c2_state_ordered_target_gap: bool,
    c2_fixed_yield_speed: float,
    c2_path_collision_only: bool,
    c2_collision_focus_steps: int,
    c2_collision_time_uncertainty: float,
    c2_proximity_collision: bool,
    c2_proximity_approach_beta: float,
    c2_proximity_margin: float,
    c2_future_collision_coef: float,
    c2_speed_efficiency_coef: float,
    c2_path_residual_coef: float,
    c2_path_smooth_coef: float,
    c2_measured_executor_timing: bool,
    c2_consistency_coef: float,
    c2_random_priority: bool,
    c2_learned_priority: bool,
    c2_speed_smooth_coef: float,
    c2_unnecessary_slow_coef: float,
    c2_extra_delay_coef: float,
    c2_detour_delay_coef: float,
    c2_dual_delay_coef: float,
    c2_explicit_gap_coef: float,
    c2_explicit_anchor_coef: float,
    c2_explicit_post_coef: float,
    c2_initial_plan_weight: float,
) -> Dict[str, float]:
    total = actions.shape[0]
    sums = {
        "policy": 0.0,
        "value": 0.0,
        "entropy": 0.0,
        "aux": 0.0,
        "best_k": 0.0,
        "nom_speed": 0.0,
        "acc_smooth": 0.0,
        "diversity_penalty": 0.0,
        "risk": 0.0,
        "future_collision": 0.0,
        "proximity_collision": 0.0,
        "proximity_collision_mode": 0.0,
        "proximity_approach_beta": 0.0,
        "proximity_margin": 0.0,
        "speed_smooth": 0.0,
        "speed_smooth_coef": 0.0,
        "unnecessary_slow": 0.0,
        "unnecessary_slow_coef": 0.0,
        "nominal_point_risk": 0.0,
        "slowdown_request_mean": 0.0,
        "extra_delay": 0.0,
        "extra_delay_coef": 0.0,
        "detour_delay": 0.0,
        "detour_delay_coef": 0.0,
        "dual_delay": 0.0,
        "dual_delay_coef": 0.0,
        "explicit_gap": 0.0,
        "explicit_gap_coef": 0.0,
        "explicit_anchor": 0.0,
        "explicit_anchor_coef": 0.0,
        "explicit_post": 0.0,
        "explicit_post_coef": 0.0,
        "explicit_need_rate": 0.0,
        "explicit_assigned_gap": 0.0,
        "initial_plan_weight": 0.0,
        "nominal_conflict_rate": 0.0,
        "peak_collision_mode": 0.0,
        "human_clearance": 0.0,
        "arrival_gap_loss": 0.0,
        "crossing_time_gap": 0.0,
        "crossing_conflict_rate": 0.0,
        "arrival_gap_mode": 0.0,
        "arrival_time_margin": 0.0,
        "arrival_gap_coef": 0.0,
        "fixed_yield_agent1": 0.0,
        "priority_speed": 0.0,
        "bidirectional_target_gap": 0.0,
        "state_ordered_target_gap": 0.0,
        "fixed_yield_speed": 0.0,
        "path_collision_only": 0.0,
        "collision_focus_steps": 0.0,
        "collision_time_uncertainty": 0.0,
        "future_collision_coef": 0.0,
        "speed_efficiency": 0.0,
        "speed_efficiency_coef": 0.0,
        "bidirectional_gap_loss": 0.0,
        "profile_efficiency": 0.0,
        "target_gap_need_rate": 0.0,
        "target_gap_actual": 0.0,
        "yield_agent1_fraction": 0.0,
        "prefix_speed_target_loss": 0.0,
        "restore_speed_loss": 0.0,
        "target_yield_speed": 0.0,
        "path_residual": 0.0,
        "path_residual_coef": 0.0,
        "path_smooth": 0.0,
        "path_smooth_coef": 0.0,
        "measured_executor_timing": 0.0,
        "consistency": 0.0,
        "consistency_position": 0.0,
        "consistency_speed": 0.0,
        "consistency_valid_fraction": 0.0,
        "consistency_coef": 0.0,
        "grad": 0.0,
    }
    updates = 0
    for _ in range(epochs):
        order = torch.randperm(total, device=actions.device)
        for start in range(0, total, minibatch):
            index = order[start:start + minibatch]
            state = states.index(index)
            consistency_target = (
                None if consistency_targets is None else {
                    key: value[index] for key, value in consistency_targets.items()
                }
            )
            log_prob, entropy, value, mean_output = policy.evaluate(state, actions[index])
            if c2_learned_priority:
                if priority_agents is None or priority_choice_masks is None:
                    raise ValueError(
                        "learned priority requires rollout choices and masks"
                    )
                choice_mask = priority_choice_masks[index].to(log_prob.dtype)
                priority_distribution = Categorical(
                    logits=mean_output["priority_logits"]
                )
                log_prob = log_prob + choice_mask * priority_distribution.log_prob(
                    priority_agents[index]
                )
                entropy = entropy + choice_mask * priority_distribution.entropy()
            if c2_random_priority or c2_learned_priority:
                if priority_agents is None:
                    raise ValueError("priority mode requires rollout priorities")
                mean_output = apply_fixed_priority(
                    mean_output, state, priority_agents[index]
                )
            ratio = (log_prob - old_log_prob[index]).exp()
            unclipped = ratio * advantages[index]
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[index]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(value, returns[index])
            entropy_value = entropy.mean()
            zero = policy_loss.detach().new_zeros(())
            if aux_coef > 0.0:
                auxiliary = (
                    compute_c2_auxiliary_loss(
                        mean_output, state,
                        peak_collision=c2_peak_collision,
                        human_clearance=c2_human_clearance,
                        arrival_gap=c2_arrival_gap,
                        arrival_time_margin=c2_arrival_time_margin,
                        arrival_gap_coef=c2_arrival_gap_coef,
                        fixed_yield_agent1=c2_fixed_yield_agent1,
                        bidirectional_target_gap=c2_bidirectional_target_gap,
                        state_ordered_target_gap=c2_state_ordered_target_gap,
                        fixed_yield_speed=c2_fixed_yield_speed,
                        path_collision_only=c2_path_collision_only,
                        collision_focus_steps=c2_collision_focus_steps,
                        collision_time_uncertainty=c2_collision_time_uncertainty,
                        proximity_collision=c2_proximity_collision,
                        proximity_approach_beta=c2_proximity_approach_beta,
                        proximity_margin=c2_proximity_margin,
                        future_collision_coef=c2_future_collision_coef,
                        speed_efficiency_coef=c2_speed_efficiency_coef,
                        path_residual_coef=c2_path_residual_coef,
                        path_smooth_coef=c2_path_smooth_coef,
                        measured_executor_timing=c2_measured_executor_timing,
                        consistency_target=consistency_target,
                        consistency_coef=c2_consistency_coef,
                        speed_smooth_coef=c2_speed_smooth_coef,
                        unnecessary_slow_coef=c2_unnecessary_slow_coef,
                        extra_delay_coef=c2_extra_delay_coef,
                        detour_delay_coef=c2_detour_delay_coef,
                        dual_delay_coef=c2_dual_delay_coef,
                        priority_agent=(
                            priority_agents[index]
                            if (c2_random_priority or c2_learned_priority)
                            else None
                        ),
                        explicit_gap_coef=c2_explicit_gap_coef,
                        explicit_anchor_coef=c2_explicit_anchor_coef,
                        explicit_post_coef=c2_explicit_post_coef,
                        initial_plan_mask=(
                            initial_plan_masks[index]
                            if initial_plan_masks is not None else None
                        ),
                        initial_plan_weight=c2_initial_plan_weight,
                    )
                    if model_kind == "c2"
                    else compute_auxiliary_loss(mean_output, state)
                )
                aux_loss = auxiliary["total"]
            else:
                auxiliary = {
                    "total": zero,
                    "best_of_k": zero,
                    "speed_nominal": zero,
                    "accel_smooth": zero,
                    "diversity": zero,
                    "risk": zero,
                    "future_collision": zero,
                    "proximity_collision": zero,
                    "proximity_collision_mode": zero,
                    "proximity_approach_beta": zero,
                    "proximity_margin": zero,
                    "speed_smooth": zero,
                    "speed_smooth_coef": zero,
                    "unnecessary_slow": zero,
                    "unnecessary_slow_coef": zero,
                    "nominal_point_risk": zero,
                    "slowdown_request_mean": zero,
                    "extra_delay": zero,
                    "extra_delay_coef": zero,
                    "detour_delay": zero,
                    "detour_delay_coef": zero,
                    "dual_delay": zero,
                    "dual_delay_coef": zero,
                    "explicit_gap": zero,
                    "explicit_gap_coef": zero,
                    "explicit_anchor": zero,
                    "explicit_anchor_coef": zero,
                    "explicit_post": zero,
                    "explicit_post_coef": zero,
                    "explicit_need_rate": zero,
                    "explicit_assigned_gap": zero,
                    "initial_plan_weight": zero,
                    "nominal_conflict_rate": zero,
                    "peak_collision_mode": zero,
                    "human_clearance": zero,
                    "arrival_gap_loss": zero,
                    "crossing_time_gap": zero,
                    "crossing_conflict_rate": zero,
                    "arrival_gap_mode": zero,
                    "arrival_time_margin": zero,
                    "arrival_gap_coef": zero,
                    "fixed_yield_agent1": zero,
                    "priority_speed": zero,
                    "bidirectional_target_gap": zero,
                    "state_ordered_target_gap": zero,
                    "fixed_yield_speed": zero,
                    "path_collision_only": zero,
                    "collision_focus_steps": zero,
                    "collision_time_uncertainty": zero,
                    "future_collision_coef": zero,
                    "speed_efficiency": zero,
                    "speed_efficiency_coef": zero,
                    "bidirectional_gap_loss": zero,
                    "profile_efficiency": zero,
                    "target_gap_need_rate": zero,
                    "target_gap_actual": zero,
                    "yield_agent1_fraction": zero,
                    "prefix_speed_target_loss": zero,
                    "restore_speed_loss": zero,
                    "target_yield_speed": zero,
                    "path_residual": zero,
                    "path_residual_coef": zero,
                    "path_smooth": zero,
                    "path_smooth_coef": zero,
                    "measured_executor_timing": zero,
                    "consistency": zero,
                    "consistency_position": zero,
                    "consistency_speed": zero,
                    "consistency_valid_fraction": zero,
                    "consistency_coef": zero,
                }
                aux_loss = zero
            loss = (
                policy_loss + value_coef * value_loss
                - entropy_coef * entropy_value + aux_coef * aux_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            sums["policy"] += float(policy_loss.detach())
            sums["value"] += float(value_loss.detach())
            sums["entropy"] += float(entropy_value.detach())
            sums["aux"] += float(aux_loss.detach())
            sums["best_k"] += float(auxiliary.get("best_of_k", zero).detach())
            sums["nom_speed"] += float(auxiliary.get("speed_nominal", zero).detach())
            sums["acc_smooth"] += float(auxiliary.get("accel_smooth", zero).detach())
            sums["diversity_penalty"] += float(auxiliary.get("diversity", zero).detach())
            sums["risk"] += float(auxiliary.get("risk", zero).detach())
            for key in (
                "future_collision", "speed_smooth", "speed_smooth_coef",
                "unnecessary_slow", "unnecessary_slow_coef",
                "nominal_point_risk", "slowdown_request_mean",
                "extra_delay", "extra_delay_coef",
                "detour_delay", "detour_delay_coef",
                "dual_delay", "dual_delay_coef",
                "explicit_gap", "explicit_gap_coef",
                "explicit_anchor", "explicit_anchor_coef",
                "explicit_post", "explicit_post_coef",
                "explicit_need_rate", "explicit_assigned_gap",
                "initial_plan_weight",
                "nominal_conflict_rate",
                "peak_collision_mode",
                "human_clearance",
                "arrival_gap_loss", "crossing_time_gap",
                "crossing_conflict_rate", "arrival_gap_mode",
                "arrival_time_margin",
                "arrival_gap_coef",
                "fixed_yield_agent1", "priority_speed",
                "bidirectional_target_gap", "bidirectional_gap_loss",
                "state_ordered_target_gap",
                "fixed_yield_speed", "path_collision_only",
                "collision_focus_steps", "speed_efficiency",
                "collision_time_uncertainty",
                "proximity_collision", "proximity_collision_mode",
                "proximity_approach_beta",
                "proximity_margin",
                "future_collision_coef",
                "speed_efficiency_coef",
                "profile_efficiency", "target_gap_need_rate",
                "target_gap_actual", "yield_agent1_fraction",
                "prefix_speed_target_loss", "restore_speed_loss",
                "target_yield_speed",
                "path_residual", "path_residual_coef",
                "path_smooth", "path_smooth_coef",
                "measured_executor_timing",
                "consistency", "consistency_position", "consistency_speed",
                "consistency_valid_fraction", "consistency_coef",
            ):
                sums[key] += float(auxiliary.get(key, zero).detach())
            sums["grad"] += float(grad)
            updates += 1
    return {key: value / max(updates, 1) for key, value in sums.items()}


def main() -> None:
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

    model_kind = os.environ.get("COORD_MODEL", "c1").strip().lower()
    if model_kind not in {"c1", "c2", "simple"}:
        raise ValueError("COORD_MODEL must be c1, c2, or simple")
    c2_learned_priority = _env_flag("COORD_C2_LEARNED_PRIORITY")
    init = os.environ.get("COORD_INIT", "")
    payload = None
    if init:
        if model_kind == "simple":
            coordinator, payload = load_simple_checkpoint(init, device)
            policy = SimpleCoordinatorActorCritic(coordinator).to(device)
        elif model_kind == "c2":
            coordinator, payload = load_c2_checkpoint(init, device)
            policy = CoordinatorActorCritic(coordinator).to(device)
        else:
            coordinator, payload = load_checkpoint(init, device)
            policy = CoordinatorActorCritic(coordinator).to(device)
        if "extras" in payload and "action_log_std" in payload["extras"]:
            policy.action_log_std.data.copy_(payload["extras"]["action_log_std"].to(device))
    else:
        if model_kind == "simple":
            simple_config = SimpleCoordinatorConfig(
                safe_bow=_env_flag("COORD_SIMPLE_SAFE_BOW")
            )
            policy = SimpleCoordinatorActorCritic(
                SimpleJointCoordinator(simple_config)
            ).to(device)
        elif model_kind == "c2":
            policy = CoordinatorActorCritic(
                JointCoordinator(c2_config(
                    actual_initial_speed=_env_flag("COORD_C2_ACTUAL_INITIAL_SPEED"),
                    immediate_slowdown=_env_flag("COORD_C2_IMMEDIATE_SLOWDOWN"),
                    direct_speed_profile=_env_flag("COORD_C2_DIRECT_SPEED_PROFILE"),
                    arrival_features=_env_flag("COORD_C2_ARRIVAL_FEATURES"),
                    mlp_backbone=_env_flag("COORD_C2_MLP_BACKBONE"),
                    mlp_conflict_features=_env_flag(
                        "COORD_C2_MLP_CONFLICT_FEATURES"
                    ),
                    direct_waypoints=_env_flag("COORD_C2_DIRECT_WAYPOINTS"),
                    joint_point_speed=_env_flag("COORD_C2_JOINT_POINT_SPEED"),
                    physical_speed_caps=_env_flag(
                        "COORD_C2_PHYSICAL_SPEED_CAPS"
                    ),
                    slowdown_window=_env_flag("COORD_C2_SLOWDOWN_WINDOW"),
                    slowdown_width_max=_env_float(
                        "COORD_C2_SLOWDOWN_WIDTH_MAX", 0.0
                    ),
                    conflict_window=_env_flag("COORD_C2_CONFLICT_WINDOW"),
                    conflict_min_at_crossing=_env_flag(
                        "COORD_C2_CONFLICT_MIN_AT_CROSSING"
                    ),
                    conflict_plateau=_env_flag(
                        "COORD_C2_CONFLICT_PLATEAU"
                    ),
                    smooth_depth=_env_flag("COORD_C2_SMOOTH_DEPTH"),
                    waypoint_smoothing_passes=_env_int(
                        "COORD_C2_WAYPOINT_SMOOTHING_PASSES", 0
                    ),
                    waypoint_distance_scaling=_env_flag(
                        "COORD_C2_WAYPOINT_DISTANCE_SCALING"
                    ),
                    learned_priority=c2_learned_priority,
                ))
            ).to(device)
        else:
            analytic_prior = _env_flag("COORD_ANALYTIC_PRIOR")
            config = CoordinatorConfig(
                residual_scale=1.0 if analytic_prior else 3.0,
                analytic_prior=analytic_prior,
            )
            policy = CoordinatorActorCritic(JointCoordinator(config)).to(device)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=_env_float("COORD_LR", 3e-4))
    if payload is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    iterations = _env_int("COORD_ITERS", 200)
    horizon = _env_int("COORD_HORIZON", 32)
    low_steps = _env_int("COORD_LOW_STEPS", 6)
    ppo_epochs = _env_int("COORD_PPO_EPOCHS", 3)
    minibatch = _env_int("COORD_MINIBATCH", 512)
    gamma = _env_float("COORD_GAMMA", 0.99)
    gae_lambda = _env_float("COORD_GAE", 0.95)
    save_every = _env_int("COORD_SAVE_EVERY", 10)
    collision_coef = _env_float(
        "COORD_COLLISION_COEF", 50.0 if model_kind in {"c2", "simple"} else 2.0
    )
    default_invalid_coef = 1.0 if model_kind == "c2" else (0.2 if model_kind == "c1" else 0.0)
    default_unsafe_coef = 0.05 if model_kind == "c1" else 0.0
    invalid_coef = _env_float("COORD_INVALID_COEF", default_invalid_coef)
    unsafe_coef = _env_float("COORD_UNSAFE_COEF", default_unsafe_coef)
    output_dir = Path(os.environ.get("COORD_OUTPUT", "../runs/coord/c1"))
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    c2_consistency_coef = _env_float("COORD_C2_CONSISTENCY_COEF", 0.0)
    if c2_consistency_coef < 0.0:
        raise ValueError("COORD_C2_CONSISTENCY_COEF must be non-negative")
    if c2_consistency_coef > 0.0 and model_kind != "c2":
        raise ValueError("trajectory consistency is available only for C2")
    c2_random_priority = _env_flag("COORD_C2_RANDOM_PRIORITY")
    if c2_random_priority and model_kind != "c2":
        raise ValueError("random fixed priority is available only for C2")
    if c2_learned_priority and model_kind != "c2":
        raise ValueError("learned persistent priority is available only for C2")
    if c2_random_priority and c2_learned_priority:
        raise ValueError("random and learned priority are exclusive")
    if payload is not None and model_kind == "c2":
        checkpoint_learned = bool(policy.coordinator.config.learned_priority)
        if checkpoint_learned != c2_learned_priority:
            raise ValueError(
                "COORD_C2_LEARNED_PRIORITY disagrees with COORD_INIT"
            )
    measured_executor_timing = _env_flag("COORD_C2_MEASURED_EXECUTOR_TIMING")
    previous_mean_output: Dict[str, torch.Tensor] | None = None
    previous_state: CoordinatorState | None = None
    previous_done = torch.ones(task.num_envs, device=device, dtype=torch.bool)

    player.env.reset()
    print(
        f"[coord-train] envs={task.num_envs} low_steps={low_steps} horizon={horizon} "
        f"model={model_kind} iterations={iterations} collision_coef={collision_coef:g} "
        f"invalid_coef={invalid_coef:g} unsafe_coef={unsafe_coef:g} "
        f"frozen_ms18={args.checkpoint}",
        flush=True,
    )
    first_iteration = 1 if payload is None else int(payload.get("step", 0)) + 1
    for iteration in range(first_iteration, first_iteration + iterations):
        state_rollout: List[CoordinatorState] = []
        consistency_rollout: List[Dict[str, torch.Tensor]] = []
        priority_rollout: List[torch.Tensor] = []
        priority_choice_rollout: List[torch.Tensor] = []
        initial_plan_rollout: List[torch.Tensor] = []
        actions: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        rewards: List[torch.Tensor] = []
        dones: List[torch.Tensor] = []
        diagnostic_sum = {"base_reward": 0.0, "progress": 0.0, "collision": 0.0}
        invalid_sum = 0.0
        unsafe_sum = 0.0
        for _ in range(horizon):
            state = task.coord_state()
            initial_plan_rollout.append(
                (task.progress_buf <= low_steps).detach().clone()
            )
            with torch.no_grad():
                output, action, log_prob, value = policy.act(state)
                priority_choice = None
                if c2_learned_priority:
                    priority_choice = ~task.coord_priority_locked()
                    priority_distribution = Categorical(
                        logits=output["priority_logits"]
                    )
                    proposed_priority = priority_distribution.sample()
                    task.set_coord_priority_agent(
                        proposed_priority, priority_choice
                    )
                    priority_agent = task.coord_priority_agent()
                    log_prob = log_prob + priority_choice.to(
                        log_prob.dtype
                    ) * priority_distribution.log_prob(priority_agent)
                elif c2_random_priority:
                    priority_agent = task.coord_priority_agent()
                else:
                    priority_agent = None
                if c2_random_priority or c2_learned_priority:
                    output = apply_fixed_priority(output, state, priority_agent)
                if c2_consistency_coef > 0.0:
                    mean_output = policy.coordinator(state)
                    if c2_random_priority or c2_learned_priority:
                        mean_output = apply_fixed_priority(
                            mean_output, state, priority_agent
                        )
                    if previous_mean_output is None or previous_state is None:
                        consistency_target = build_trajectory_consistency_target(
                            mean_output, state, state, 0.0,
                            torch.zeros_like(previous_done),
                            measured_executor_timing=measured_executor_timing,
                        )
                    else:
                        consistency_target = build_trajectory_consistency_target(
                            previous_mean_output, previous_state, state,
                            low_steps * float(task.dt), ~previous_done,
                            measured_executor_timing=measured_executor_timing,
                        )
                _, valid, safe = task.install_external_coord(output)
            reward, done, diagnostics = _macro_step(player, low_steps, collision_coef)
            reward = (
                reward
                - invalid_coef * (~valid).float()
                - unsafe_coef * (~safe).float()
            )
            state_rollout.append(state.clone())
            if c2_random_priority or c2_learned_priority:
                priority_rollout.append(priority_agent.detach().clone())
            if c2_learned_priority:
                priority_choice_rollout.append(
                    priority_choice.detach().clone()
                )
            if c2_consistency_coef > 0.0:
                consistency_rollout.append(consistency_target)
                previous_mean_output = {
                    key: mean_output[key].detach().clone()
                    for key in ("path_world", "speed", "pickup_dwell")
                }
                previous_state = state.clone()
                previous_done = done.detach().clone()
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)
            invalid_sum += float((~valid).float().mean())
            unsafe_sum += float((~safe).float().mean())
            for key in diagnostic_sum:
                diagnostic_sum[key] += diagnostics[key]

        with torch.no_grad():
            _, next_value, _ = policy.distribution(task.coord_state())
        reward_tensor = torch.stack(rewards)
        done_tensor = torch.stack(dones).float()
        value_tensor = torch.stack(values)
        advantage = torch.zeros_like(reward_tensor)
        gae = torch.zeros(task.num_envs, device=device)
        bootstrap = next_value
        for step in reversed(range(horizon)):
            mask = 1.0 - done_tensor[step]
            delta = reward_tensor[step] + gamma * bootstrap * mask - value_tensor[step]
            gae = delta + gamma * gae_lambda * mask * gae
            advantage[step] = gae
            bootstrap = value_tensor[step]
        returns = advantage + value_tensor
        flat_advantage = advantage.flatten()
        flat_advantage = (flat_advantage - flat_advantage.mean()) / flat_advantage.std().clamp(min=1e-6)
        update = _ppo_update(
            policy,
            optimizer,
            _flatten_states(state_rollout),
            _flatten_consistency_targets(consistency_rollout),
            _flatten_priority_agents(priority_rollout),
            _flatten_masks(priority_choice_rollout),
            _flatten_masks(initial_plan_rollout),
            torch.cat(actions),
            torch.cat(log_probs),
            returns.flatten(),
            flat_advantage,
            ppo_epochs,
            minibatch,
            _env_float("COORD_CLIP", 0.2),
            _env_float("COORD_VALUE_COEF", 0.5),
            _env_float("COORD_ENTROPY_COEF", 1e-4),
            _env_float(
                "COORD_AUX_COEF", 1.0 if model_kind == "c2" else (0.0 if model_kind == "simple" else 0.01)
            ),
            model_kind,
            _env_flag("COORD_C2_PEAK_COLLISION"),
            _env_float("COORD_C2_HUMAN_CLEARANCE", 1.0),
            _env_flag("COORD_C2_ARRIVAL_GAP"),
            _env_float("COORD_C2_ARRIVAL_TIME_MARGIN", 1.0),
            _env_float("COORD_C2_ARRIVAL_GAP_COEF", 2.0),
            _env_flag("COORD_C2_FIXED_YIELD_A1"),
            _env_flag("COORD_C2_BIDIRECTIONAL_TARGET_GAP"),
            _env_flag("COORD_C2_STATE_ORDERED_TARGET_GAP"),
            _env_float("COORD_C2_FIXED_YIELD_SPEED", 0.0),
            _env_flag("COORD_C2_PATH_COLLISION_ONLY"),
            _env_int("COORD_C2_COLLISION_FOCUS_STEPS", 0),
            _env_float("COORD_C2_COLLISION_TIME_UNCERTAINTY", 0.0),
            _env_flag("COORD_C2_PROXIMITY_COLLISION"),
            _env_float("COORD_C2_PROXIMITY_APPROACH_BETA", 1.0),
            _env_float("COORD_C2_PROXIMITY_MARGIN", 0.0),
            _env_float("COORD_C2_FUTURE_COLLISION_COEF", 50.0),
            _env_float("COORD_C2_SPEED_EFFICIENCY_COEF", 0.0),
            _env_float("COORD_C2_PATH_RESIDUAL_COEF", 0.0),
            _env_float("COORD_C2_PATH_SMOOTH_COEF", 0.0),
            measured_executor_timing,
            c2_consistency_coef,
            c2_random_priority,
            c2_learned_priority,
            _env_float("COORD_C2_SPEED_SMOOTH_COEF", 0.5),
            _env_float("COORD_C2_UNNECESSARY_SLOW_COEF", 0.1),
            _env_float("COORD_C2_EXTRA_DELAY_COEF", 0.0),
            _env_float("COORD_C2_DETOUR_DELAY_COEF", 0.0),
            _env_float("COORD_C2_DUAL_DELAY_COEF", 0.0),
            _env_float("COORD_C2_EXPLICIT_GAP_COEF", 0.0),
            _env_float("COORD_C2_EXPLICIT_ANCHOR_COEF", 0.0),
            _env_float("COORD_C2_EXPLICIT_POST_COEF", 0.0),
            _env_float("COORD_C2_INITIAL_PLAN_WEIGHT", 1.0),
        )
        metrics = {
            "iteration": iteration,
            "reward": float(reward_tensor.mean()),
            "invalid_rate": invalid_sum / horizon,
            "unsafe_rate": unsafe_sum / horizon,
            **{key: value / horizon for key, value in diagnostic_sum.items()},
            **update,
            "random_priority": float(c2_random_priority),
            "learned_priority": float(c2_learned_priority),
            "priority_agent1_fraction": (
                float(torch.cat(priority_rollout).float().mean())
                if priority_rollout else 0.0
            ),
            "priority_choice_fraction": (
                float(torch.cat(priority_choice_rollout).float().mean())
                if priority_choice_rollout else 0.0
            ),
            "action_std": float(policy.action_log_std.exp().mean().detach()),
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        print("COORD_TRAIN " + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()), flush=True)
        if iteration % save_every == 0 or iteration == iterations:
            if model_kind == "simple":
                save_fn, prefix = save_simple_checkpoint, "coord_b0"
            elif model_kind == "c2":
                save_fn, prefix = save_c2_checkpoint, "coord_c2"
            else:
                save_fn, prefix = save_checkpoint, "coord_c1"
            save_fn(
                output_dir / f"{prefix}_{iteration:06d}.pth",
                policy.coordinator,
                step=iteration,
                metrics=metrics,
                optimizer=optimizer,
                extras={
                    "action_log_std": policy.action_log_std.detach().cpu(),
                    "random_priority": c2_random_priority,
                    "learned_priority": c2_learned_priority,
                },
            )
            save_fn(
                output_dir / f"{prefix}_latest.pth",
                policy.coordinator,
                step=iteration,
                metrics=metrics,
                optimizer=optimizer,
                extras={
                    "action_log_std": policy.action_log_std.detach().cpu(),
                    "random_priority": c2_random_priority,
                    "learned_priority": c2_learned_priority,
                },
            )


if __name__ == "__main__":
    main()

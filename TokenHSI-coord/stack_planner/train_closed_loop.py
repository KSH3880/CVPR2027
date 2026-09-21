"""Closed-loop PPO for the stack planner with the stack agent frozen."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, List

# Isaac Gym requires this import before torch.
from isaacgym import gymapi as _gymapi  # noqa: F401

import torch
import torch.nn.functional as F

COORD_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = COORD_ROOT.parent
MASTEER_ROOT = WORKSPACE / "TokenHSI-masteer"
sys.path.insert(0, str(MASTEER_ROOT / "tokenhsi"))
sys.path.insert(0, str(COORD_ROOT))

import run as tokenhsi_run  # noqa: E402
import utils.parse_task as task_registry  # noqa: E402
from coordinator.schema import AGENTS, STATE_KEYS, CoordinatorState  # noqa: E402
from stack_planner.checkpoint import load_stack_checkpoint, save_stack_checkpoint  # noqa: E402
from stack_planner.branching import TaskBranchSnapshot  # noqa: E402
from stack_planner.consistency import (  # noqa: E402
    build_stack_consistency_target, stack_trajectory_consistency_loss,
)
from stack_planner.constraints import (  # noqa: E402
    free_path_validity_details, ordered_box_goal_visit,
)
from stack_planner.env_adapter import HumanoidMAStackPlannerTrain  # noqa: E402
from stack_planner.history import (  # noqa: E402
    StackHistoryBuffer, flatten_observations,
)
from stack_planner.model import StackPlannerConfig, StackTrajectoryPlanner  # noqa: E402
from stack_planner.policy import StackPlannerActorCritic  # noqa: E402
from stack_planner.reward import (  # noqa: E402
    StackIntervalCosts, StackPhysicalState, StackRewardConfig,
    compute_stack_planner_reward,
)
from utils.config import get_args, load_cfg, set_np_formatting, set_seed  # noqa: E402

task_registry.HumanoidMAStackPlannerTrain = HumanoidMAStackPlannerTrain


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _flatten_consistency_targets(targets):
    if not targets:
        return None
    return {
        key: torch.cat([target[key] for target in targets], dim=0)
        for key in targets[0]
    }


def _clone_physical(state: StackPhysicalState) -> StackPhysicalState:
    return StackPhysicalState(**{
        item.name: getattr(state, item.name).detach().clone()
        for item in fields(StackPhysicalState)
    })


def _masked_update(target: StackPhysicalState, source: StackPhysicalState, mask):
    for item in fields(StackPhysicalState):
        old = getattr(target, item.name)
        new = getattr(source, item.name)
        shape = (mask.shape[0],) + (1,) * (old.ndim - 1)
        setattr(target, item.name, torch.where(mask.reshape(shape), new, old))


def _refresh_obs(player):
    player.env.task._compute_observations()
    return torch.clamp(
        player.env.task.obs_buf, -player.env.clip_obs, player.env.clip_obs
    ).to(player.device)


@torch.no_grad()
def _macro_step(player, before, valid, safe, low_steps, reward_config):
    task = player.env.task
    n = task.num_envs
    collision = torch.zeros(n, device=task.device)
    disturbance = torch.zeros(n, device=task.device)
    fall = torch.zeros(n, device=task.device)
    active = torch.ones(n, dtype=torch.bool, device=task.device)
    done_env = torch.zeros_like(active)
    after = _clone_physical(before)
    last_bottom_position = task.planner_bottom_position().clone()
    obs = _refresh_obs(player)
    player.get_batch_size(obs, 1)
    elapsed_steps = torch.zeros(n, device=task.device)
    path_error_sum = torch.zeros(n, device=task.device)
    path_samples = torch.zeros(n, device=task.device)
    collision_steps = torch.zeros(n, device=task.device)
    collision_names = ("agent_agent", "agent_box", "held_box_body")
    collision_component_cost = {
        name: torch.zeros(n, device=task.device) for name in collision_names
    }
    collision_component_steps = {
        name: torch.zeros(n, device=task.device) for name in collision_names
    }
    postplace_steps = torch.zeros(n, device=task.device)
    postplace_motion = torch.zeros(n, device=task.device)
    postplace_angular_motion = torch.zeros(n, device=task.device)
    for _ in range(low_steps):
        action = player.get_action({"obs": obs}, is_determenistic=True)
        obs, _, done_rows, _ = player.env.step(action)
        current = _clone_physical(task.planner_physical_state())
        step_collision_terms = task.planner_collision_terms()
        step_collision = step_collision_terms["total"]
        bottom_position = task.planner_bottom_position()
        step_disturbance = (
            bottom_position - last_bottom_position
        ).norm(dim=-1) * task._planner_bottom_stable_seen.float()
        collision += torch.where(active, step_collision, torch.zeros_like(step_collision))
        collision_steps += active.float() * (step_collision > 0.0).float()
        for name in collision_names:
            component = step_collision_terms[name]
            collision_component_cost[name] += torch.where(
                active, component, torch.zeros_like(component),
            )
            collision_component_steps[name] += (
                active.float() * (component > 0.0).float()
            )
        disturbance += torch.where(active, step_disturbance, torch.zeros_like(step_disturbance))
        fall = torch.maximum(fall, task.planner_fall().float() * active.float())
        elapsed_steps += active.float()
        owned = task.planner_active_rows()
        owned &= active[:, None]
        lateral = task._lat_root.reshape(n, AGENTS).abs()
        path_error_sum += (lateral * owned.float()).sum(dim=-1)
        path_samples += owned.sum(dim=-1)
        postplace = (
            (task._stack_phase >= task.A1_RETREAT)
            & (task._stack_phase < task.DONE)
            & ~task._carry_rehearsal
            & active
        )
        postplace_steps += postplace.float()
        postplace_motion += (
            current.bottom_linear_velocity.norm(dim=-1)
            * float(task.dt) * postplace.float()
        )
        postplace_angular_motion += (
            current.bottom_angular_velocity.norm(dim=-1)
            * float(task.dt) * postplace.float()
        )
        _masked_update(after, current, active)
        last_bottom_position = torch.where(
            active[:, None], bottom_position, last_bottom_position
        )
        just_done = done_rows.reshape(n, AGENTS).any(dim=1) & active
        if just_done.any():
            done_env |= just_done
            active &= ~just_done
            row_ids = torch.nonzero(
                just_done.repeat_interleave(AGENTS), as_tuple=False
            ).squeeze(-1)
            obs = player.env.reset(row_ids)
    denom = elapsed_steps.clamp(min=1.0)
    interval = StackIntervalCosts(
        collision=collision / denom,
        bottom_disturbance=disturbance,
        humanoid_fall=fall.clamp(0, 1),
        elapsed_seconds=elapsed_steps * float(task.dt),
        invalid_plan=(~valid).float(),
        unsafe_plan=(valid & ~safe).float(),
    )
    diagnostics = {
        "path_error_sum": path_error_sum,
        "path_samples": path_samples,
        "collision_steps": collision_steps,
        "executed_steps": elapsed_steps,
        "collision_cost_sum": interval.collision,
        "fall_events": fall,
        "macro_samples": torch.ones_like(fall),
        "bottom_postplace_motion": postplace_motion,
        "bottom_postplace_angular_motion": postplace_angular_motion,
        "bottom_postplace_seconds": postplace_steps * float(task.dt),
        "bottom_postplace_intervals": (postplace_steps > 0).float(),
    }
    for name in collision_names:
        diagnostics[f"collision_{name}_steps"] = collision_component_steps[name]
        diagnostics[f"collision_{name}_cost_sum"] = (
            collision_component_cost[name] / denom
        )
    return (
        compute_stack_planner_reward(before, after, interval, reward_config),
        done_env,
        diagnostics,
    )


def _counterfactual_rollout(
    player, output, state, before, low_steps, reward_config,
    visit_penalty_coef, visit_tolerance, retreat_box_penalty_coef,
    endpoint_penalty_coef, turn_penalty_coef,
):
    """Execute one candidate branch and return its complete macro reward."""
    task = player.env.task
    with torch.no_grad():
        visit = ordered_box_goal_visit(
            output["path_world"],
            state.box_xyz[:, None, :, :2],
            state.goal_xy[:, None],
            tolerance=visit_tolerance,
        )
        turn = free_path_validity_details(
            output["path_world"], output["speed"], state.root_xy,
            task.planner_execution_rows(),
        )
        valid, safe = task.install_external_plan(output)
        decision = task._planner_policy_decision.clone()
        retreat_decision = (
            (task._stack_phase == task.A1_RETREAT)
            & ~task._carry_rehearsal
        ).clone()
        endpoint_change_cost = task._planner_endpoint_change_cost.clone()
        retreat_box_path_cost = task._planner_retreat_box_path_penalty.clone()
        retreat_box_min_clearance = task._planner_retreat_box_min_clearance.clone()
        retreat_box_endpoint_clearance = (
            task._planner_retreat_box_endpoint_clearance.clone()
        )
    scored_valid = valid | ~decision
    scored_safe = safe | ~decision
    terms, done, macro_diag = _macro_step(
        player, before, scored_valid, scored_safe, low_steps, reward_config,
    )
    terms["retreat_potential_delta_removed"] = torch.where(
        retreat_decision, terms["potential_delta"],
        torch.zeros_like(terms["potential_delta"]),
    )
    terms["total"] -= terms["retreat_potential_delta_removed"]
    route_penalty = visit["penalty"][:, 0].mean(dim=-1)
    terms["route_visit_penalty"] = (
        -visit_penalty_coef * route_penalty * decision.float()
    )
    terms["route_box_distance"] = visit["box_distance"][:, 0].mean(dim=-1)
    terms["route_goal_distance"] = visit["goal_distance"][:, 0].mean(dim=-1)
    turn_cost = turn["turn_excess_cost"][:, 0].mean(dim=-1)
    terms["route_turn_penalty"] = (
        -turn_penalty_coef * turn_cost * decision.float()
    )
    terms["route_max_turn_deg"] = turn["max_turn_deg"][:, 0].amax(dim=-1)
    terms["retreat_box_path_penalty"] = (
        -retreat_box_penalty_coef * retreat_box_path_cost
    )
    terms["retreat_box_min_clearance"] = retreat_box_min_clearance
    terms["retreat_box_endpoint_clearance"] = retreat_box_endpoint_clearance
    terms["total"] += (
        terms["route_visit_penalty"] + terms["route_turn_penalty"]
        + terms["retreat_box_path_penalty"]
    )
    terms["retreat_endpoint_change_penalty"] = (
        -endpoint_penalty_coef * endpoint_change_cost * decision.float()
    )
    terms["retreat_endpoint_change_cost"] = endpoint_change_cost
    terms["total"] += terms["retreat_endpoint_change_penalty"]
    invalid_decision = decision & ~valid
    analytic = (
        terms["invalid_plan_penalty"]
        + terms["unsafe_plan_penalty"]
        + terms["route_visit_penalty"]
        + terms["route_turn_penalty"]
        + terms["retreat_box_path_penalty"]
        + terms["retreat_endpoint_change_penalty"]
    )
    terms["total"] = torch.where(invalid_decision, analytic, terms["total"])
    return terms, done, macro_diag, decision, invalid_decision


def _make_player(args, cfg, cfg_train):
    tokenhsi_run.args = args
    tokenhsi_run.cfg = cfg
    tokenhsi_run.cfg_train = cfg_train
    runner = tokenhsi_run.build_alg_runner(tokenhsi_run.RLGPUAlgoObserver())
    runner.load(cfg_train)
    runner.reset()
    player = runner.create_player()
    if args.checkpoint == "Base":
        raise ValueError("--checkpoint must be the frozen sequential-stack policy")
    player.restore(args.checkpoint)
    player.model.eval()
    for parameter in player.model.parameters():
        parameter.requires_grad_(False)
    if player.env.task.__class__.__name__ != "HumanoidMAStackPlannerTrain":
        raise ValueError("--task must be HumanoidMAStackPlannerTrain")
    return player


def _ppo_update(policy, optimizer, states, actions, old_log_prob, returns,
                advantages, decision_mask, candidate_targets,
                consistency_targets, epochs, minibatch,
                clip_ratio, value_coef, entropy_coef, consistency_coef,
                diversity_coef, diversity_margin, smoothness_coef,
                speed_smoothness_coef,
                evaluator_coef):
    total = actions.shape[0]
    sums = {
        "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
        "consistency": 0.0, "consistency_position": 0.0,
        "consistency_valid_fraction": 0.0,
        "candidate_diversity_loss": 0.0,
        "candidate_path_distance": 0.0,
        "path_smoothness_loss": 0.0,
        "speed_smoothness_loss": 0.0,
        "candidate_evaluator_loss": 0.0,
        "candidate_evaluator_accuracy": 0.0,
    }
    updates = 0
    for _ in range(epochs):
        for index in torch.randperm(total, device=actions.device).split(minibatch):
            state = states.index(index)
            log_prob, entropy, value, decoded = policy.evaluate(
                state, actions[index]
            )
            ratio = (log_prob - old_log_prob[index]).exp()
            objective = torch.minimum(
                ratio * advantages[index],
                ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[index],
            )
            decided = decision_mask[index]
            if decided.any():
                policy_loss = -objective[decided].mean()
                entropy_mean = entropy[decided].mean()
            else:
                # Keep a differentiable zero so value/consistency can still
                # train on continuation states in an all-option minibatch.
                policy_loss = value.sum() * 0.0
                entropy_mean = entropy.sum() * 0.0
            value_loss = (value - returns[index]).square().mean()
            if consistency_coef > 0.0 and consistency_targets is not None:
                target = {
                    key: item[index]
                    for key, item in consistency_targets.items()
                }
                consistency = stack_trajectory_consistency_loss(
                    policy.planner(state), state, target,
                )
            else:
                zero = policy_loss.new_zeros(())
                consistency = {
                    "total": zero, "position": zero,
                    "valid_fraction": zero,
                }
            diversity = policy.diversity(state, margin=diversity_margin)
            evaluator_active = decoded["candidate_logits"].shape[-1] > 1
            evaluator_per_item = F.cross_entropy(
                decoded["candidate_logits"], candidate_targets[index],
                reduction="none",
            )
            if evaluator_active and decided.any():
                evaluator_loss = evaluator_per_item[decided].mean()
                evaluator_accuracy = (
                    decoded["candidate_logits"][decided].argmax(dim=-1)
                    == candidate_targets[index][decided]
                ).float().mean()
            else:
                evaluator_loss = evaluator_per_item.sum() * 0.0
                evaluator_accuracy = evaluator_per_item.new_zeros(())
            loss = (
                policy_loss + value_coef * value_loss
                - entropy_coef * entropy_mean
                + consistency_coef * consistency["total"]
                + diversity_coef * diversity["loss"]
                + smoothness_coef * diversity["smoothness_loss"]
                + speed_smoothness_coef * diversity["speed_smoothness_loss"]
                + evaluator_coef * evaluator_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            sums["policy_loss"] += float(policy_loss.detach())
            sums["value_loss"] += float(value_loss.detach())
            sums["entropy"] += float(entropy_mean.detach())
            sums["consistency"] += float(consistency["total"].detach())
            sums["consistency_position"] += float(consistency["position"].detach())
            sums["consistency_valid_fraction"] += float(
                consistency["valid_fraction"].detach()
            )
            sums["candidate_diversity_loss"] += float(
                diversity["loss"].detach()
            )
            sums["candidate_path_distance"] += float(
                diversity["distance"].detach()
            )
            sums["path_smoothness_loss"] += float(
                diversity["smoothness_loss"].detach()
            )
            sums["speed_smoothness_loss"] += float(
                diversity["speed_smoothness_loss"].detach()
            )
            sums["candidate_evaluator_loss"] += float(
                evaluator_loss.detach()
            )
            sums["candidate_evaluator_accuracy"] += float(
                evaluator_accuracy.detach()
            )
            updates += 1
    return {key: value / max(updates, 1) for key, value in sums.items()}


def main():
    set_np_formatting()
    args = get_args()
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

    init = os.environ.get("STACK_PLANNER_INIT", "")
    requested_history_steps = _env_int("STACK_PLANNER_HISTORY_STEPS", 4)
    requested_candidates = _env_int("STACK_PLANNER_CANDIDATES", 1)
    requested_delta_scale = _env_float("STACK_PLANNER_DELTA_SCALE", 0.5)
    if requested_candidates < 1:
        raise ValueError("STACK_PLANNER_CANDIDATES must be positive")
    if requested_delta_scale <= 0:
        raise ValueError("STACK_PLANNER_DELTA_SCALE must be positive")
    payload = None
    if init:
        planner, payload = load_stack_checkpoint(init, device)
        if planner.config.history_steps != requested_history_steps:
            raise ValueError(
                "STACK_PLANNER_HISTORY_STEPS must match init checkpoint: "
                f"requested={requested_history_steps} "
                f"checkpoint={planner.config.history_steps}"
            )
        if planner.config.candidates != requested_candidates:
            raise ValueError(
                "STACK_PLANNER_CANDIDATES must match init checkpoint: "
                f"requested={requested_candidates} "
                f"checkpoint={planner.config.candidates}"
            )
        if planner.config.delta_scale != requested_delta_scale:
            raise ValueError(
                "STACK_PLANNER_DELTA_SCALE must match init checkpoint: "
                f"requested={requested_delta_scale} "
                f"checkpoint={planner.config.delta_scale}"
            )
    else:
        planner = StackTrajectoryPlanner(StackPlannerConfig(
            candidates=requested_candidates, history_steps=requested_history_steps,
            delta_scale=requested_delta_scale,
        )).to(device)
    history = StackHistoryBuffer(
        task.num_envs, planner.config.history_steps, device,
    )
    point_std = _env_float("STACK_PLANNER_DELTA_STD", 0.12)
    endpoint_std = _env_float("STACK_PLANNER_ENDPOINT_STD", 0.20)
    anchor_std = _env_float("STACK_PLANNER_ANCHOR_STD", 0.03)
    speed_std = _env_float("STACK_PLANNER_SPEED_STD", 0.20)
    policy = StackPlannerActorCritic(
        planner, point_std=point_std, endpoint_std=endpoint_std,
        anchor_std=anchor_std, speed_std=speed_std,
    ).to(device)
    if payload and payload.get("extras", {}).get("action_log_std") is not None:
        loaded_log_std = payload["extras"]["action_log_std"].to(device)
        if loaded_log_std.shape != policy.action_log_std.shape:
            raise ValueError("checkpoint action_log_std shape mismatch")
        # Resuming an older straight-path run must not silently restore its
        # 0.03 curve exploration. Preserve any larger learned std, while the
        # configured initialization acts as a per-dimension resume floor.
        policy.action_log_std.data.copy_(torch.maximum(
            loaded_log_std, policy.action_log_std.detach()
        ))
    optimizer = torch.optim.Adam(policy.parameters(), lr=_env_float("STACK_PLANNER_LR", 3e-4))
    if payload and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    policy.train()

    iterations = _env_int("STACK_PLANNER_ITERS", 200)
    horizon = _env_int("STACK_PLANNER_HORIZON", 32)
    low_steps = _env_int("STACK_PLANNER_LOW_STEPS", 30)
    gamma = _env_float("STACK_PLANNER_GAMMA", 0.99)
    consistency_coef = _env_float("STACK_PLANNER_CONSISTENCY_COEF", 1.0)
    diversity_coef = _env_float("STACK_PLANNER_DIVERSITY_COEF", 0.05)
    diversity_margin = _env_float("STACK_PLANNER_DIVERSITY_MARGIN", 0.25)
    smoothness_coef = _env_float("STACK_PLANNER_SMOOTHNESS_COEF", 10.0)
    speed_smoothness_coef = _env_float(
        "STACK_PLANNER_SPEED_SMOOTHNESS_COEF", 1.0
    )
    evaluator_coef = _env_float("STACK_PLANNER_EVALUATOR_COEF", 1.0)
    retreat_path_scale = _env_float("STACK_PLANNER_RETREAT_PATH_CONSISTENCY_SCALE", 0.05)
    endpoint_penalty_coef = _env_float("STACK_PLANNER_RETREAT_ENDPOINT_PENALTY", 1.0)
    endpoint_tolerance = _env_float("STACK_PLANNER_RETREAT_ENDPOINT_TOLERANCE", 0.10)
    if not 0 <= retreat_path_scale <= 1 or endpoint_penalty_coef < 0 or endpoint_tolerance < 0:
        raise ValueError("invalid retreat consistency/endpoint settings")
    if (consistency_coef < 0 or diversity_coef < 0 or smoothness_coef < 0
            or speed_smoothness_coef < 0
            or diversity_margin < 0 or evaluator_coef < 0):
        raise ValueError("planner consistency/diversity settings must be non-negative")
    visit_penalty_coef = _env_float("STACK_PLANNER_VISIT_PENALTY", 10.0)
    visit_tolerance = _env_float("STACK_PLANNER_VISIT_TOLERANCE", 0.15)
    turn_penalty_coef = _env_float("STACK_PLANNER_TURN_PENALTY", 2.0)
    retreat_box_penalty_coef = _env_float(
        "STACK_PLANNER_RETREAT_BOX_PENALTY", 10.0
    )
    bottom_disturbance_weight = _env_float(
        "STACK_PLANNER_BOTTOM_DISTURBANCE_WEIGHT", 10.0
    )
    if (visit_penalty_coef < 0 or visit_tolerance < 0 or turn_penalty_coef < 0
            or retreat_box_penalty_coef < 0 or bottom_disturbance_weight < 0):
        raise ValueError("stack planner path-shaping coefficients must be non-negative")
    reward_config = StackRewardConfig(
        bottom_disturbance_weight=bottom_disturbance_weight,
    )
    output_dir = Path(os.environ.get(
        "STACK_PLANNER_OUTPUT", str(WORKSPACE / "runs/stack_planner/default")
    )).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    save_every = _env_int("STACK_PLANNER_SAVE_EVERY", 5)
    ppo_epochs = _env_int("STACK_PLANNER_PPO_EPOCHS", 3)
    minibatch = _env_int("STACK_PLANNER_MINIBATCH", 512)
    if ppo_epochs <= 0 or minibatch <= 0:
        raise ValueError("PPO epochs and minibatch must be positive")
    transitions = task.num_envs * horizon * planner.config.candidates
    optimizer_steps = ppo_epochs * math.ceil(transitions / minibatch)

    # Task construction already performs a complete reset and creates the
    # initial observation.  A second full-vector reset here is redundant and
    # triggers an Isaac Gym GPU-pipeline fault on this stack-task build.
    _refresh_obs(player)
    print(f"[stack-planner-train] envs={task.num_envs} horizon={horizon} "
          f"low_steps={low_steps} consistency={consistency_coef:g} "
          f"delta_std={point_std:g} endpoint_std={endpoint_std:g} "
          f"speed_std={speed_std:g} "
          f"history_steps={planner.config.history_steps} "
          f"candidates={planner.config.candidates} "
          f"delta_scale={planner.config.delta_scale:g} "
          f"diversity={diversity_coef:g}/{diversity_margin:g}m "
          f"smoothness={smoothness_coef:g}/{speed_smoothness_coef:g} "
          f"evaluator_coef={evaluator_coef:g} "
          f"full_candidate_rollout={planner.config.candidates > 1} "
          f"visit_penalty={visit_penalty_coef:g} visit_tol={visit_tolerance:g} "
          f"turn_penalty={turn_penalty_coef:g} "
          f"retreat_box_penalty={retreat_box_penalty_coef:g} "
          f"bottom_disturbance_weight={bottom_disturbance_weight:g} "
          f"a2_stable_delay={task._planner_a2_stable_delay:g}s "
          f"minibatch={minibatch} optimizer_steps={optimizer_steps} "
          f"iterations={iterations} frozen={args.checkpoint}",
          flush=True)
    first = 1 if payload is None else int(payload.get("step", 0)) + 1
    previous_mean_output = None
    previous_state = None
    previous_done = torch.ones(task.num_envs, device=device, dtype=torch.bool)
    for iteration in range(first, first + iterations):
        states = []
        consistency_targets = []
        actions: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        candidate_returns: List[torch.Tensor] = []
        candidate_advantages: List[torch.Tensor] = []
        candidate_targets: List[torch.Tensor] = []
        rewards: List[torch.Tensor] = []
        dones: List[torch.Tensor] = []
        decision_masks: List[torch.Tensor] = []
        diag: Dict[str, float] = {}
        planner_diag: Dict[str, float] = {}
        candidate_counts = torch.zeros(
            planner.config.candidates, device=device, dtype=torch.long,
        )
        candidate_score_sums = torch.zeros(
            planner.config.candidates, device=device,
        )
        for _ in range(horizon):
            state = task.planner_state()
            observation = history.observe(
                state, reset_mask=previous_done, commit=True,
            )
            before = _clone_physical(task.planner_physical_state())
            retreat_at_decision = (
                (task._stack_phase >= task.A1_RETREAT)
                & (task._stack_phase < task.DONE)
                & ~task._carry_rehearsal
            ).clone()
            with torch.no_grad():
                if consistency_coef > 0.0:
                    if previous_mean_output is None or previous_state is None:
                        consistency_target = build_stack_consistency_target(
                            policy.mean_output(observation), state, state, 0.0,
                            torch.zeros_like(previous_done),
                        )
                    else:
                        consistency_target = build_stack_consistency_target(
                            previous_mean_output, previous_state, state,
                            low_steps * float(task.dt), ~previous_done,
                        )
                outputs, all_actions, all_log_probs, value = policy.sample_all(
                    observation
                )

            # Every head starts from an identical simulator/task snapshot.
            # The best per-environment branch is merged back afterwards, so
            # the next planning decision follows an actually executed result.
            base_snapshot = TaskBranchSnapshot.capture(task)
            best_snapshot = base_snapshot.clone()
            best_score = torch.full(
                (task.num_envs,), -torch.inf, device=device,
            )
            branch_terms = []
            branch_dones = []
            branch_diagnostics = []
            branch_decisions = []
            branch_invalid = []
            branch_scores = []
            for candidate in range(planner.config.candidates):
                base_snapshot.restore(task)
                candidate_output = {
                    "path_local": outputs["path_local"][:, candidate:candidate + 1],
                    "path_world": outputs["path_world"][:, candidate:candidate + 1],
                    "speed": outputs["speed"][:, candidate:candidate + 1],
                }
                terms, done, macro_diag, decision, invalid = (
                    _counterfactual_rollout(
                        player, candidate_output, state, before, low_steps,
                        reward_config, visit_penalty_coef, visit_tolerance,
                        retreat_box_penalty_coef, endpoint_penalty_coef,
                        turn_penalty_coef,
                    )
                )
                with torch.no_grad():
                    next_state = task.planner_state()
                    plan_update = decision & ~invalid
                    next_plan_world = torch.where(
                        plan_update[:, None, None, None],
                        outputs["path_world"][:, candidate],
                        observation.previous_path_world,
                    )
                    next_plan_valid = (
                        observation.previous_path_valid | plan_update
                    )
                    next_observation = history.observe(
                        next_state, reset_mask=done, commit=False,
                        previous_path_world=next_plan_world,
                        previous_path_valid=next_plan_valid,
                    )
                    next_value = policy.value(next_observation)
                    continuation = ~(done | invalid)
                    score = terms["total"] + (
                        gamma * next_value * continuation.float()
                    )
                candidate_snapshot = TaskBranchSnapshot.capture(task)
                better = score > best_score
                best_snapshot.update_where(candidate_snapshot, better)
                best_score = torch.where(better, score, best_score)
                branch_terms.append(terms)
                branch_dones.append(done)
                branch_diagnostics.append(macro_diag)
                branch_decisions.append(decision)
                branch_invalid.append(invalid)
                branch_scores.append(score)

            scores = torch.stack(branch_scores)             # [K,B]
            candidate_score_sums += scores.mean(dim=1)
            best_candidate = scores.argmax(dim=0)           # [B]
            env_index = torch.arange(task.num_envs, device=device)
            best_snapshot.restore(task)
            _refresh_obs(player)

            def select_branch(items, key):
                stacked = torch.stack([item[key] for item in items])
                return stacked[best_candidate, env_index]

            terms = {
                key: select_branch(branch_terms, key)
                for key in branch_terms[0]
            }
            done = torch.stack(branch_dones)[best_candidate, env_index]
            decision_stack = torch.stack(branch_decisions)
            decision = decision_stack[best_candidate, env_index]
            invalid = torch.stack(branch_invalid)[best_candidate, env_index]
            macro_diag = {
                key: select_branch(branch_diagnostics, key)
                for key in branch_diagnostics[0]
            }
            if decision.any():
                candidate_counts += torch.bincount(
                    best_candidate[decision],
                    minlength=planner.config.candidates,
                )

            if consistency_coef > 0.0:
                consistency_target["valid"] &= decision[:, None, None, None]
                scale = torch.ones_like(consistency_target["position"][..., 0])
                scale[retreat_at_decision, :, 0] = retreat_path_scale
                consistency_target["position_scale"] = scale

            # Store every physically evaluated candidate. The evaluator target
            # is the best same-scene branch; continuous PPO credit remains on
            # the head/path that produced each individual score.
            for candidate in range(planner.config.candidates):
                states.append(observation.clone())
                actions.append(all_actions[:, candidate])
                log_probs.append(all_log_probs[:, candidate])
                candidate_returns.append(best_score.detach())
                candidate_advantages.append(
                    (scores[candidate] - value).detach()
                )
                decision_masks.append(branch_decisions[candidate])
                candidate_targets.append(best_candidate)
                if consistency_coef > 0.0:
                    consistency_targets.append({
                        key: item.clone()
                        for key, item in consistency_target.items()
                    })
            rewards.append(terms["total"])
            dones.append(done)
            selected_path_world = outputs["path_world"][
                env_index, best_candidate
            ]
            commit_mask = decision & ~invalid & ~done
            history.commit_path(
                selected_path_world,
                update_mask=commit_mask,
            )
            if consistency_coef > 0.0:
                selected_sample_path = outputs["path_world"][
                    env_index, best_candidate
                ][:, None]
                if previous_mean_output is None:
                    previous_mean_output = {
                        "path_world": selected_sample_path.detach().clone(),
                    }
                    previous_state = state.clone()
                else:
                    path_mask = commit_mask.reshape(-1, 1, 1, 1, 1)
                    previous_mean_output["path_world"] = torch.where(
                        path_mask, selected_sample_path,
                        previous_mean_output["path_world"],
                    ).detach()
                    for key in STATE_KEYS:
                        old = getattr(previous_state, key)
                        new = getattr(state, key)
                        state_mask = commit_mask.reshape(
                            -1, *((1,) * (old.ndim - 1))
                        )
                        setattr(
                            previous_state, key,
                            torch.where(state_mask, new, old).detach(),
                        )
            previous_done = done.detach().clone()
            for key, tensor in terms.items():
                diag[key] = diag.get(key, 0.0) + float(tensor.mean())
            for key, tensor in macro_diag.items():
                planner_diag[key] = (
                    planner_diag.get(key, 0.0) + float(tensor.sum())
                )
        reward_t = torch.stack(rewards)
        done_t = torch.stack(dones).float()
        returns = torch.cat(candidate_returns)
        flat_decision = torch.cat(decision_masks)
        flat_adv = torch.cat(candidate_advantages)
        decided_adv = flat_adv[flat_decision]
        if decided_adv.numel() > 1:
            normalized = (
                (flat_adv - decided_adv.mean())
                / decided_adv.std().clamp(min=1e-6)
            )
            flat_adv = torch.where(
                flat_decision, normalized, torch.zeros_like(normalized)
            )
        else:
            flat_adv = torch.zeros_like(flat_adv)
        update = _ppo_update(
            policy, optimizer, flatten_observations(states), torch.cat(actions),
            torch.cat(log_probs), returns, flat_adv, flat_decision,
            torch.cat(candidate_targets),
            _flatten_consistency_targets(consistency_targets),
            ppo_epochs,
            minibatch,
            _env_float("STACK_PLANNER_CLIP", 0.2),
            _env_float("STACK_PLANNER_VALUE_COEF", 0.5),
            _env_float("STACK_PLANNER_ENTROPY_COEF", 1e-4),
            consistency_coef,
            diversity_coef,
            diversity_margin,
            smoothness_coef,
            speed_smoothness_coef,
            evaluator_coef,
        )
        def ratio(numerator, denominator):
            return planner_diag.get(numerator, 0.0) / max(
                planner_diag.get(denominator, 0.0), 1e-8
            )

        planner_metrics = {
            "path_mae": ratio("path_error_sum", "path_samples"),
            "fall_ratio": ratio("fall_events", "macro_samples"),
            "collision_ratio": ratio("collision_steps", "executed_steps"),
            "collision_cost": ratio("collision_cost_sum", "macro_samples"),
            "bottom_postplace_linear_speed": ratio(
                "bottom_postplace_motion", "bottom_postplace_seconds"
            ),
            "bottom_postplace_angular_speed": ratio(
                "bottom_postplace_angular_motion", "bottom_postplace_seconds"
            ),
            "bottom_postplace_motion_per_interval": ratio(
                "bottom_postplace_motion", "bottom_postplace_intervals"
            ),
            "bottom_postplace_exposure": ratio(
                "bottom_postplace_seconds", "executed_steps"
            ) / float(task.dt),
        }
        for collision_name in (
            "agent_agent", "agent_box", "held_box_body",
        ):
            planner_metrics[f"collision_{collision_name}_ratio"] = ratio(
                f"collision_{collision_name}_steps", "executed_steps",
            )
            planner_metrics[f"collision_{collision_name}_cost"] = ratio(
                f"collision_{collision_name}_cost_sum", "macro_samples",
            )
        metrics = {
            "iteration": iteration,
            "reward": float(reward_t.mean()),
            "done_rate": float(done_t.mean()),
            **{key: value / horizon for key, value in diag.items()},
            **update,
            "consistency_coef": consistency_coef,
            "visit_penalty_coef": visit_penalty_coef,
            "visit_tolerance": visit_tolerance,
            "turn_penalty_coef": turn_penalty_coef,
            "retreat_box_penalty_coef": retreat_box_penalty_coef,
            "bottom_disturbance_weight": bottom_disturbance_weight,
            "planner_decision_rate": float(flat_decision.float().mean()),
            "retreat_path_consistency_scale": retreat_path_scale,
            "retreat_endpoint_penalty_coef": endpoint_penalty_coef,
            "retreat_endpoint_tolerance": endpoint_tolerance,
            "delta_std": point_std,
            "endpoint_std": endpoint_std,
            "anchor_std": anchor_std,
            "speed_std": speed_std,
            "history_steps": planner.config.history_steps,
            "candidates": planner.config.candidates,
            "candidate_evaluator_active": float(
                planner.config.candidates > 1
            ),
            "delta_scale": planner.config.delta_scale,
            "diversity_coef": diversity_coef,
            "diversity_margin": diversity_margin,
            "smoothness_coef": smoothness_coef,
            "speed_smoothness_coef": speed_smoothness_coef,
            "full_candidate_rollout": float(planner.config.candidates > 1),
            **{
                f"candidate_usage_{candidate}": float(
                    candidate_counts[candidate].float()
                    / candidate_counts.sum().clamp(min=1)
                )
                for candidate in range(planner.config.candidates)
            },
            **{
                f"candidate_score_{candidate}": float(
                    candidate_score_sums[candidate] / horizon
                )
                for candidate in range(planner.config.candidates)
            },
            **planner_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        print("[stack-planner-train] " + " ".join(
            f"{key}={value:.5g}" for key, value in metrics.items()
            if key in {"iteration", "reward", "done_rate", "potential_delta",
                       "retreat_potential_delta_removed",
                       "humanoid_fall_penalty", "route_visit_penalty",
                       "bottom_disturbance_penalty",
                       "route_turn_penalty", "route_max_turn_deg",
                       "retreat_box_path_penalty",
                       "retreat_box_endpoint_clearance",
                       "path_mae", "fall_ratio", "collision_ratio",
                       "bottom_postplace_linear_speed",
                       "policy_loss", "value_loss",
                       "candidate_path_distance",
                       "path_smoothness_loss", "collision_cost",
                       "collision_agent_agent_ratio",
                       "collision_agent_box_ratio",
                       "collision_held_box_body_ratio",
                       "collision_agent_agent_cost",
                       "collision_agent_box_cost",
                       "collision_held_box_body_cost",
                       "speed_smoothness_loss",
                       "candidate_evaluator_loss",
                       "candidate_evaluator_accuracy"}
        ), flush=True)
        if iteration % save_every == 0 or iteration == first + iterations - 1:
            save_stack_checkpoint(
                output_dir / f"planner_{iteration:06d}.pth", policy.planner,
                step=iteration, metrics=metrics, optimizer=optimizer,
                extras={"action_log_std": policy.action_log_std.detach().cpu(),
                        "frozen_executor": str(Path(args.checkpoint).resolve()),
                        "consistency_coef": consistency_coef,
                        "commit_steps": low_steps,
                        "retreat_path_consistency_scale": retreat_path_scale,
                        "delta_std": point_std,
                        "endpoint_std": endpoint_std,
                        "anchor_std": anchor_std,
                        "speed_std": speed_std,
                        "history_steps": planner.config.history_steps,
                        "candidates": planner.config.candidates,
                        "delta_scale": planner.config.delta_scale,
                        "full_candidate_rollout": planner.config.candidates > 1,
                        "evaluator_coef": evaluator_coef,
                        "diversity_coef": diversity_coef,
                        "diversity_margin": diversity_margin,
                        "smoothness_coef": smoothness_coef,
                        "speed_smoothness_coef": speed_smoothness_coef,
                        "retreat_endpoint_penalty_coef": endpoint_penalty_coef,
                        "retreat_endpoint_tolerance": endpoint_tolerance,
                        "visit_penalty_coef": visit_penalty_coef,
                        "visit_tolerance": visit_tolerance,
                        "turn_penalty_coef": turn_penalty_coef,
                        "retreat_box_penalty_coef": retreat_box_penalty_coef,
                        "bottom_disturbance_weight": bottom_disturbance_weight},
            )


if __name__ == "__main__":
    main()

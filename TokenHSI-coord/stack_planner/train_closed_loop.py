"""Closed-loop PPO for the stack planner with the stack agent frozen."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, List

# Isaac Gym requires this import before torch.
from isaacgym import gymapi as _gymapi  # noqa: F401

import torch

COORD_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = COORD_ROOT.parent
MASTEER_ROOT = WORKSPACE / "TokenHSI-masteer"
sys.path.insert(0, str(MASTEER_ROOT / "tokenhsi"))
sys.path.insert(0, str(COORD_ROOT))

import run as tokenhsi_run  # noqa: E402
import utils.parse_task as task_registry  # noqa: E402
from coordinator.schema import AGENTS, STATE_KEYS, CoordinatorState  # noqa: E402
from stack_planner.checkpoint import load_stack_checkpoint, save_stack_checkpoint  # noqa: E402
from stack_planner.consistency import (  # noqa: E402
    build_stack_consistency_target, stack_trajectory_consistency_loss,
)
from stack_planner.constraints import ordered_box_goal_visit  # noqa: E402
from stack_planner.env_adapter import HumanoidMAStackPlannerTrain  # noqa: E402
from stack_planner.model import StackPlannerConfig, StackTrajectoryPlanner  # noqa: E402
from stack_planner.policy import StackPlannerActorCritic  # noqa: E402
from stack_planner.reward import (  # noqa: E402
    StackIntervalCosts, StackPhysicalState, compute_stack_planner_reward,
)
from utils.config import get_args, load_cfg, set_np_formatting, set_seed  # noqa: E402

task_registry.HumanoidMAStackPlannerTrain = HumanoidMAStackPlannerTrain


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _flatten_states(states: List[CoordinatorState]) -> CoordinatorState:
    return CoordinatorState(**{
        key: torch.cat([getattr(state, key) for state in states], dim=0)
        for key in STATE_KEYS
    })


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
def _macro_step(player, before, valid, safe, low_steps):
    task = player.env.task
    n = task.num_envs
    collision = torch.zeros(n, device=task.device)
    disturbance = torch.zeros(n, device=task.device)
    fall = torch.zeros(n, device=task.device)
    active = torch.ones(n, dtype=torch.bool, device=task.device)
    done_env = torch.zeros_like(active)
    after = _clone_physical(before)
    last_bottom_error = before.bottom_position_error.clone()
    obs = _refresh_obs(player)
    player.get_batch_size(obs, 1)
    elapsed_steps = torch.zeros(n, device=task.device)
    for _ in range(low_steps):
        action = player.get_action({"obs": obs}, is_determenistic=True)
        obs, _, done_rows, _ = player.env.step(action)
        current = _clone_physical(task.planner_physical_state())
        step_collision = task.planner_collision_cost()
        step_disturbance = (
            current.bottom_position_error - last_bottom_error
        ).norm(dim=-1)
        collision += torch.where(active, step_collision, torch.zeros_like(step_collision))
        disturbance += torch.where(active, step_disturbance, torch.zeros_like(step_disturbance))
        fall = torch.maximum(fall, task.planner_fall().float() * active.float())
        elapsed_steps += active.float()
        _masked_update(after, current, active)
        last_bottom_error = torch.where(
            active[:, None], current.bottom_position_error, last_bottom_error
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
    return compute_stack_planner_reward(before, after, interval), done_env


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
                advantages, consistency_targets, epochs, minibatch,
                clip_ratio, value_coef, entropy_coef, consistency_coef):
    total = actions.shape[0]
    sums = {
        "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
        "consistency": 0.0, "consistency_position": 0.0,
        "consistency_valid_fraction": 0.0,
    }
    updates = 0
    for _ in range(epochs):
        for index in torch.randperm(total, device=actions.device).split(minibatch):
            state = states.index(index)
            log_prob, entropy, value, _ = policy.evaluate(state, actions[index])
            ratio = (log_prob - old_log_prob[index]).exp()
            objective = torch.minimum(
                ratio * advantages[index],
                ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[index],
            )
            policy_loss = -objective.mean()
            value_loss = (value - returns[index]).square().mean()
            entropy_mean = entropy.mean()
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
            loss = (
                policy_loss + value_coef * value_loss
                - entropy_coef * entropy_mean
                + consistency_coef * consistency["total"]
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
    payload = None
    if init:
        planner, payload = load_stack_checkpoint(init, device)
    else:
        planner = StackTrajectoryPlanner(StackPlannerConfig(candidates=1)).to(device)
    policy = StackPlannerActorCritic(planner).to(device)
    if payload and payload.get("extras", {}).get("action_log_std") is not None:
        policy.action_log_std.data.copy_(payload["extras"]["action_log_std"].to(device))
    optimizer = torch.optim.Adam(policy.parameters(), lr=_env_float("STACK_PLANNER_LR", 3e-4))
    if payload and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    policy.train()

    iterations = _env_int("STACK_PLANNER_ITERS", 200)
    horizon = _env_int("STACK_PLANNER_HORIZON", 32)
    low_steps = _env_int("STACK_PLANNER_LOW_STEPS", 30)
    gamma = _env_float("STACK_PLANNER_GAMMA", 0.99)
    gae_lambda = _env_float("STACK_PLANNER_GAE", 0.95)
    consistency_coef = _env_float("STACK_PLANNER_CONSISTENCY_COEF", 1.0)
    if consistency_coef < 0:
        raise ValueError("STACK_PLANNER_CONSISTENCY_COEF must be non-negative")
    visit_penalty_coef = _env_float("STACK_PLANNER_VISIT_PENALTY", 10.0)
    visit_tolerance = _env_float("STACK_PLANNER_VISIT_TOLERANCE", 0.15)
    if visit_penalty_coef < 0 or visit_tolerance < 0:
        raise ValueError("stack planner path-shaping coefficients must be non-negative")
    output_dir = Path(os.environ.get(
        "STACK_PLANNER_OUTPUT", str(WORKSPACE / "runs/stack_planner/default")
    )).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    save_every = _env_int("STACK_PLANNER_SAVE_EVERY", 10)

    # Task construction already performs a complete reset and creates the
    # initial observation.  A second full-vector reset here is redundant and
    # triggers an Isaac Gym GPU-pipeline fault on this stack-task build.
    _refresh_obs(player)
    print(f"[stack-planner-train] envs={task.num_envs} horizon={horizon} "
          f"low_steps={low_steps} consistency={consistency_coef:g} "
          f"visit_penalty={visit_penalty_coef:g} visit_tol={visit_tolerance:g} "
          f"iterations={iterations} frozen={args.checkpoint}",
          flush=True)
    first = 1 if payload is None else int(payload.get("step", 0)) + 1
    previous_mean_output = None
    previous_state = None
    previous_done = torch.ones(task.num_envs, device=device, dtype=torch.bool)
    for iteration in range(first, first + iterations):
        states: List[CoordinatorState] = []
        consistency_targets = []
        actions: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        rewards: List[torch.Tensor] = []
        dones: List[torch.Tensor] = []
        diag: Dict[str, float] = {}
        for _ in range(horizon):
            state = task.planner_state()
            before = _clone_physical(task.planner_physical_state())
            with torch.no_grad():
                if consistency_coef > 0.0:
                    mean_output = policy.planner(state)
                    if previous_mean_output is None or previous_state is None:
                        consistency_target = build_stack_consistency_target(
                            mean_output, state, state, 0.0,
                            torch.zeros_like(previous_done),
                        )
                    else:
                        consistency_target = build_stack_consistency_target(
                            previous_mean_output, previous_state, state,
                            low_steps * float(task.dt), ~previous_done,
                        )
                output, action, log_prob, value = policy.act(state)
                visit = ordered_box_goal_visit(
                    output["path_world"],
                    state.box_xyz[:, None, :, :2].expand(
                        -1, output["path_world"].shape[1], -1, -1
                    ),
                    state.goal_xy[:, None].expand(
                        -1, output["path_world"].shape[1], -1, -1
                    ),
                    tolerance=visit_tolerance,
                )
                valid, safe = task.install_external_plan(output)
            terms, done = _macro_step(player, before, valid, safe, low_steps)
            route_penalty = visit["penalty"][:, 0].mean(dim=-1)
            terms["route_visit_penalty"] = -visit_penalty_coef * route_penalty
            terms["route_box_distance"] = visit["box_distance"][:, 0].mean(dim=-1)
            terms["route_goal_distance"] = visit["goal_distance"][:, 0].mean(dim=-1)
            terms["total"] = terms["total"] + terms["route_visit_penalty"]
            states.append(state.clone())
            if consistency_coef > 0.0:
                consistency_targets.append(consistency_target)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(terms["total"])
            dones.append(done)
            if consistency_coef > 0.0:
                previous_mean_output = {
                    key: mean_output[key].detach().clone()
                    for key in ("path_world",)
                }
                previous_state = state.clone()
                previous_done = done.detach().clone()
            for key, tensor in terms.items():
                diag[key] = diag.get(key, 0.0) + float(tensor.mean())

        with torch.no_grad():
            _, next_value, _ = policy.distribution(task.planner_state())
        reward_t = torch.stack(rewards)
        done_t = torch.stack(dones).float()
        value_t = torch.stack(values)
        advantage = torch.zeros_like(reward_t)
        gae = torch.zeros(task.num_envs, device=device)
        bootstrap = next_value
        for step in reversed(range(horizon)):
            mask = 1.0 - done_t[step]
            delta = reward_t[step] + gamma * bootstrap * mask - value_t[step]
            gae = delta + gamma * gae_lambda * mask * gae
            advantage[step] = gae
            bootstrap = value_t[step]
        returns = advantage + value_t
        flat_adv = advantage.flatten()
        flat_adv = (flat_adv - flat_adv.mean()) / flat_adv.std().clamp(min=1e-6)
        update = _ppo_update(
            policy, optimizer, _flatten_states(states), torch.cat(actions),
            torch.cat(log_probs), returns.flatten(), flat_adv,
            _flatten_consistency_targets(consistency_targets),
            _env_int("STACK_PLANNER_PPO_EPOCHS", 3),
            _env_int("STACK_PLANNER_MINIBATCH", 512),
            _env_float("STACK_PLANNER_CLIP", 0.2),
            _env_float("STACK_PLANNER_VALUE_COEF", 0.5),
            _env_float("STACK_PLANNER_ENTROPY_COEF", 1e-4),
            consistency_coef,
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
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        print("[stack-planner-train] " + " ".join(
            f"{key}={value:.5g}" for key, value in metrics.items()
            if key in {"iteration", "reward", "done_rate", "potential_delta",
                       "humanoid_fall_penalty", "route_visit_penalty",
                       "policy_loss", "value_loss"}
        ), flush=True)
        if iteration % save_every == 0 or iteration == first + iterations - 1:
            save_stack_checkpoint(
                output_dir / f"planner_{iteration:06d}.pth", policy.planner,
                step=iteration, metrics=metrics, optimizer=optimizer,
                extras={"action_log_std": policy.action_log_std.detach().cpu(),
                        "frozen_executor": str(Path(args.checkpoint).resolve()),
                        "consistency_coef": consistency_coef,
                        "commit_steps": low_steps,
                        "visit_penalty_coef": visit_penalty_coef,
                        "visit_tolerance": visit_tolerance},
            )


if __name__ == "__main__":
    main()

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
from coordinator.schema import AGENTS  # noqa: E402
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
def _macro_step(player, low_steps: int, collision_coef: float):
    task = player.env.task
    n = task.num_envs
    start_distance = task.planner_task_distance()
    reward_sum = torch.zeros(n, device=task.device)
    collision_sum = torch.zeros(n, device=task.device)
    collision_steps = torch.zeros(n, device=task.device)
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
        collision_steps += active.float() * (collision["total"] > 0).float()
        for name in component_sum:
            component_sum[name] += torch.where(
                active, collision[name], torch.zeros_like(collision[name]),
            )
        executed += active.float()
        end_distance = torch.where(active, distance, end_distance)
        just_done = done_rows.reshape(n, AGENTS).any(dim=1) & active
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
        reward_sum / denom + 2.0 * progress
        - collision_coef * collision_mean - 0.01
    )
    diagnostics = {
        "base_reward": reward_sum.sum(),
        "progress": progress.sum(),
        "collision_cost": collision_sum.sum(),
        "collision_steps": collision_steps.sum(),
        "executed_steps": executed.sum(),
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
                speed_smoothness_coef):
    total = actions.shape[0]
    sums = {
        "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
        "path_smoothness_loss": 0.0, "speed_smoothness_loss": 0.0,
    }
    updates = 0
    for _ in range(epochs):
        for index in torch.randperm(total, device=actions.device).split(minibatch):
            observation = observations.index(index)
            log_prob, entropy, value, _ = policy.evaluate(
                observation, actions[index],
            )
            ratio = (log_prob - old_log_prob[index]).exp()
            objective = torch.minimum(
                ratio * advantages[index],
                ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
                * advantages[index],
            )
            policy_loss = -objective.mean()
            value_loss = F.mse_loss(value, returns[index])
            entropy_mean = entropy.mean()
            regularization = policy.diversity(observation)
            loss = (
                policy_loss + value_coef * value_loss
                - entropy_coef * entropy_mean
                + smoothness_coef * regularization["smoothness_loss"]
                + speed_smoothness_coef
                * regularization["speed_smoothness_loss"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            sums["policy_loss"] += float(policy_loss.detach())
            sums["value_loss"] += float(value_loss.detach())
            sums["entropy"] += float(entropy_mean.detach())
            sums["path_smoothness_loss"] += float(
                regularization["smoothness_loss"].detach()
            )
            sums["speed_smoothness_loss"] += float(
                regularization["speed_smoothness_loss"].detach()
            )
            updates += 1
    return {key: value / max(updates, 1) for key, value in sums.items()}


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
    path_update_alpha = _env_float("CARRY_PLANNER_PATH_UPDATE_ALPHA", 0.25)
    init = os.environ.get("CARRY_PLANNER_INIT", "")
    payload = None
    if init:
        planner, payload = load_stack_checkpoint(init, device)
        expected = (
            history_steps, delta_scale, delta_scale,
            path_update_alpha, True,
        )
        actual = (
            planner.config.history_steps, planner.config.delta_scale,
            planner.config.retreat_delta_scale,
            planner.config.path_update_alpha, planner.config.plain_carry,
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
        )).to(device)
    history = StackHistoryBuffer(task.num_envs, history_steps, device)
    policy = StackPlannerActorCritic(
        planner,
        point_std=_env_float("CARRY_PLANNER_DELTA_STD", 0.12),
        endpoint_std=_env_float("CARRY_PLANNER_ENDPOINT_STD", 0.03),
        anchor_std=_env_float("CARRY_PLANNER_ANCHOR_STD", 0.03),
        speed_std=_env_float("CARRY_PLANNER_SPEED_STD", 0.20),
    ).to(device)
    if payload and payload.get("extras", {}).get("action_log_std") is not None:
        policy.action_log_std.data.copy_(
            payload["extras"]["action_log_std"].to(device)
        )
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=_env_float("CARRY_PLANNER_LR", 3e-4),
    )
    if payload and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    policy.train()

    iterations = _env_int("CARRY_PLANNER_ITERS", 200)
    horizon = _env_int("CARRY_PLANNER_HORIZON", 32)
    low_steps = _env_int("CARRY_PLANNER_LOW_STEPS", 6)
    gamma = _env_float("CARRY_PLANNER_GAMMA", 0.99)
    collision_coef = _env_float("CARRY_PLANNER_COLLISION_COEF", 5.0)
    ppo_epochs = _env_int("CARRY_PLANNER_PPO_EPOCHS", 3)
    minibatch = _env_int("CARRY_PLANNER_MINIBATCH", 512)
    smoothness_coef = _env_float("CARRY_PLANNER_SMOOTHNESS_COEF", 10.0)
    speed_smoothness_coef = _env_float(
        "CARRY_PLANNER_SPEED_SMOOTHNESS_COEF", 1.0,
    )
    if min(iterations, horizon, low_steps, ppo_epochs, minibatch) <= 0:
        raise ValueError("iteration/horizon/step/minibatch values must be positive")
    if collision_coef < 0 or smoothness_coef < 0 or speed_smoothness_coef < 0:
        raise ValueError("reward and regularization coefficients must be non-negative")

    output_dir = Path(os.environ.get(
        "CARRY_PLANNER_OUTPUT", str(WORKSPACE / "runs/carry_planner/default"),
    )).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    save_every = _env_int("CARRY_PLANNER_SAVE_EVERY", 5)
    _refresh_obs(player)
    print(
        f"[carry-planner-train] envs={task.num_envs} horizon={horizon} "
        f"low_steps={low_steps} history={history_steps} delta={delta_scale:g} "
        f"collision_coef={collision_coef:g} "
        f"converge_prob={task._carry_converge_prob:g} "
        f"goal_margin={task._carry_goal_margin:g} frozen={args.checkpoint}",
        flush=True,
    )

    first = 1 if payload is None else int(payload.get("step", 0)) + 1
    previous_done = torch.ones(task.num_envs, dtype=torch.bool, device=device)
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
                }
                valid = task.install_external_plan(candidate_output)
                reward, done, macro_diag = _macro_step(
                    player, low_steps, collision_coef,
                )
                mean_path = output["mean_path_world"][:, 0]
                commit = valid & ~done
                history.commit_path(
                    mean_path, update_mask=commit,
                    base_path_world=output["base_path_world"],
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
        update = _ppo_update(
            policy, optimizer, flatten_observations(observations),
            torch.cat(actions), torch.cat(log_probs), torch.cat(returns),
            flat_advantage, ppo_epochs, minibatch,
            _env_float("CARRY_PLANNER_CLIP", 0.2),
            _env_float("CARRY_PLANNER_VALUE_COEF", 0.5),
            _env_float("CARRY_PLANNER_ENTROPY_COEF", 1e-4),
            smoothness_coef, speed_smoothness_coef,
        )

        def ratio(numerator, denominator):
            return diag.get(numerator, 0.0) / max(
                diag.get(denominator, 0.0), 1e-8,
            )

        metrics = {
            "iteration": iteration,
            "reward": float(torch.stack(rewards).mean()),
            "done_rate": float(torch.stack(dones).float().mean()),
            "progress": ratio("progress", "samples"),
            "base_reward": ratio("base_reward", "executed_steps"),
            "collision_cost": ratio("collision_cost", "executed_steps"),
            "collision_ratio": ratio("collision_steps", "executed_steps"),
            "collision_agent_agent_cost": ratio(
                "collision_agent_agent_cost", "executed_steps",
            ),
            "collision_agent_box_cost": ratio(
                "collision_agent_box_cost", "executed_steps",
            ),
            "collision_box_box_cost": ratio(
                "collision_box_box_cost", "executed_steps",
            ),
            "converge_fraction": float(
                task._carry_converge_layout.float().mean()
            ),
            **update,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(
            "[carry-planner-train] " + " ".join(
                f"{key}={value:.5g}" for key, value in metrics.items()
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
                    "collision_coef": collision_coef,
                    "commit_steps": low_steps,
                    "converge_probability": task._carry_converge_prob,
                    "goal_margin": task._carry_goal_margin,
                },
            )


if __name__ == "__main__":
    main()

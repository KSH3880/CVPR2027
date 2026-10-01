"""Train, evaluate or probe fixed-route Carry speed control."""

import os
import json
import sys
from pathlib import Path

from isaacgym import gymapi as _gymapi  # import before torch
import numpy as np
import torch
import torch.nn.functional as F

COORD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COORD / "tokenhsi"))
import utils.parse_task as registry
from utils.config import get_args, load_cfg, set_seed, set_np_formatting
from carry_planner.train_closed_loop import _make_player, _macro_step
from carry_planner.episode_collision import EpisodeCollisionTracker
from stack_planner.history import StackHistoryBuffer, flatten_observations, project_path_progress
from carry_speed.env import HumanoidMACarrySpeedTrain
from carry_speed.control import action_speeds, controlled_mask, generalized_advantages, rule_speeds
from carry_speed.model import CarrySpeedPolicy

registry.HumanoidMACarrySpeedTrain = HumanoidMACarrySpeedTrain


def observe(history, task, reset, commit):
    observation = history.observe(task.planner_state(), reset_mask=reset, commit=commit)
    path = task.speed_paths.detach().clone()
    valid = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
    observation.base_path_world = path
    observation.previous_path_world = path
    observation.base_path_valid = valid
    observation.previous_path_valid = valid
    minimum = torch.where(reset[:, None], torch.zeros_like(history.path_progress), history.path_progress)
    observation.path_progress = project_path_progress(path, observation.state.root_xy, minimum)
    if commit:
        history.commit_path(path, base_path_world=path)
        history.path_progress.copy_(observation.path_progress)
    return observation


def ppo_update(policy, optimizer, observations, priorities, actions, old_log_prob, advantages, returns):
    advantage = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp(min=1e-6)
    epochs = int(os.environ.get("CARRY_SPEED_PPO_EPOCHS", "3"))
    minibatch = int(os.environ.get("CARRY_SPEED_MINIBATCH", "512"))
    target_kl = float(os.environ.get("CARRY_SPEED_TARGET_KL", "0.02"))
    clip = float(os.environ.get("CARRY_SPEED_CLIP", "0.2"))
    entropy_coef = float(os.environ.get("CARRY_SPEED_ENTROPY_COEF", "0.003"))
    sums = dict(policy=0., value=0., entropy=0., approx_kl=0.)
    updates = 0
    stop = False
    for _ in range(epochs):
        for index in torch.randperm(len(actions), device=actions.device).split(minibatch):
            log_prob, entropy, value = policy.evaluate(observations.index(index), priorities[index], actions[index])
            log_ratio = log_prob - old_log_prob[index]
            ratio = log_ratio.exp()
            kl = ((ratio - 1) - log_ratio).mean()
            if float(kl.detach()) > target_kl:
                stop = True
                break
            policy_loss = -torch.minimum(ratio * advantage[index], ratio.clamp(1-clip, 1+clip) * advantage[index]).mean()
            value_loss = F.mse_loss(value, returns[index])
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            for name, scalar in (("policy", policy_loss), ("value", value_loss), ("entropy", entropy.mean()), ("approx_kl", kl)):
                sums[name] += float(scalar.detach())
            updates += 1
        if stop:
            break
    return {**{key: value / max(updates, 1) for key, value in sums.items()}, "updates": updates, "kl_early_stop": int(stop)}


def main():
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, _ = load_cfg(args)
    seed = set_seed(cfg_train["params"].get("seed", 0), False)
    cfg_train["params"]["seed"] = seed
    cfg_train["params"]["config"]["seed"] = seed
    cfg_train["params"]["config"]["train_dir"] = args.output_path
    if args.motion_file:
        cfg["env"]["motion_file"] = args.motion_file
    mode = os.environ.get("CARRY_SPEED_MODE", "train")
    if mode not in ("train", "policy", "nominal", "rule", "probe"):
        raise ValueError("mode must be train, policy, nominal, rule or probe")
    out = Path(os.environ["CARRY_SPEED_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    player = _make_player(args, cfg, cfg_train)
    task = player.env.task
    player.env.reset()
    device = torch.device(player.device)
    task._carry_timeout_deadline.fill_(task.max_episode_length)
    checkpoint = os.environ.get("CARRY_SPEED_INIT", "")
    policy, payload = (CarrySpeedPolicy.load(checkpoint, device) if checkpoint else (CarrySpeedPolicy().to(device), None))
    if mode == "policy" and payload is None:
        raise ValueError("policy evaluation requires CARRY_SPEED_INIT")
    if payload and Path(payload["executor"]).resolve() != Path(args.checkpoint).resolve():
        raise ValueError("speed policy checkpoint was trained with a different frozen executor")
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(os.environ.get("CARRY_SPEED_LR", "0.0003")))
    if mode == "train" and payload:
        optimizer.load_state_dict(payload["optimizer"])
    policy.train(mode == "train")
    history = StackHistoryBuffer(task.num_envs, policy.config.history_steps, device)
    reset = torch.ones(task.num_envs, dtype=torch.bool, device=device)
    tracker = EpisodeCollisionTracker(task.num_envs, device)
    iterations = int(os.environ.get("CARRY_SPEED_ITERS", "200" if mode == "train" else "2"))
    horizon = int(os.environ.get("CARRY_SPEED_HORIZON", "32"))
    low_steps = int(os.environ.get("CARRY_SPEED_LOW_STEPS", "6"))
    gamma = float(os.environ.get("CARRY_SPEED_GAMMA", "0.99"))
    lam = float(os.environ.get("CARRY_SPEED_GAE_LAMBDA", "0.95"))
    if min(iterations, horizon, low_steps) < 1:
        raise ValueError("iterations, horizon and low_steps must be positive")
    traces = []
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(out / "tensorboard"))
    except ImportError:
        pass
    first = int(payload["step"]) + 1 if payload and mode == "train" else 1
    totals = dict(completed=0., successes=0., proxy_collided=0., proxy_steps=0., executed_steps=0.)
    print(f"[carry-speed] mode={mode} envs={task.num_envs} seed={seed} executor={args.checkpoint} "
          f"fixed_routes=True random_priority=True speed_choices=0,.375,.75,1.125,1.5 free_prob={task._speed_free_probability}", flush=True)
    for iteration in range(first, first + iterations):
        observations, priorities, actions, probabilities, rewards, dones, values, next_values = ([] for _ in range(8))
        diag = dict(collision_episodes=0., completed_episodes=0., collision_steps=0., executed_steps=0., collision_cost=0.)
        stop_requested = stop_sent = stopped = held_n = controlled_n = moving_n = 0.
        speed_at_stop = stopped_at_stop = held_at_stop = reward_total = 0.
        completed_before, success_before = task.speed_completed, task.speed_successes
        for _ in range(horizon):
            observation = observe(history, task, reset, True)
            priority = task.speed_priority.clone()
            with torch.no_grad():
                action, log_prob, value = policy.act(observation, priority, deterministic=mode != "train")
                speed = action_speeds(action, priority)
                if mode == "nominal":
                    speed.fill_(1.5)
                elif mode == "rule":
                    speed = rule_speeds(observation.state.root_xy, task.speed_center, observation.state.goal_xy, priority)
                    speed = torch.where(task.speed_free[:, None], torch.full_like(speed, 1.5), speed)
                elif mode == "probe":
                    age = task.progress_buf.float() * task.dt
                    wait = (age >= 0.5) & (age < float(os.environ.get("CARRY_SPEED_PROBE_STOP_END", "2.5")))
                    speed.fill_(1.5)
                    speed = torch.where(wait[:, None] & controlled_mask(priority), torch.zeros_like(speed), speed)
                task.set_requested_speeds(speed)
                previous_success = task.speed_success_total.clone()
                reward, done, macro = _macro_step(player, low_steps,
                    float(os.environ.get("CARRY_SPEED_COLLISION_COEF", "10")),
                    float(os.environ.get("CARRY_SPEED_PROGRESS_COEF", "2")), tracker)
                reward += 6.0 * (task.speed_success_total - previous_success)
                next_observation = observe(history, task, done, False)
                _, next_value = policy.distribution(next_observation, task.speed_priority)
                state = next_observation.state
                mask = controlled_mask(priority) & ~done[:, None]
                sent = task._coord_cmd_speed.reshape(task.num_envs, 2)
                actual = state.root_vel_xy.norm(dim=-1)
                zero_sent = (sent < 0.05) & mask
                speed_at_stop += float(actual[zero_sent].sum())
                stopped_at_stop += float(((actual < 0.2) & zero_sent).sum())
                held_at_stop += float(((state.held > 0.5) & zero_sent).sum())
                reward_total += float(reward.sum())
                controlled_n += float(mask.sum())
                stop_requested += float(((speed < 0.05) & mask).sum())
                stop_sent += float(((sent < 0.05) & mask).sum())
                stopped += float(((actual < 0.2) & mask).sum())
                moving_n += float(((actual > 0.5) & mask).sum())
                held_n += float(((state.held > 0.5) & mask).sum())
                if mode != "train":
                    traces.append({key: tensor.detach().cpu().numpy() for key, tensor in {
                        "requested": speed, "sent": sent, "actual_speed": actual, "root": state.root_xy,
                        "box": state.box_xyz, "held": state.held, "done": done, "priority": priority,
                        "age": task.progress_buf.float() * task.dt, "free": task.speed_free,
                        "route": task.speed_paths,
                    }.items()})
            if mode == "train":
                observations.append(observation.clone())
                priorities.append(priority)
                actions.append(action)
                probabilities.append(log_prob)
                rewards.append(reward)
                dones.append(done)
                values.append(value)
                next_values.append(next_value)
            for key in diag:
                diag[key] += float(macro[key])
            reset = done.clone()
        update = {}
        if mode == "train":
            advantage, returns = generalized_advantages(torch.stack(rewards), torch.stack(dones), torch.stack(values), torch.stack(next_values), gamma, lam)
            update = ppo_update(policy, optimizer, flatten_observations(observations), torch.cat(priorities),
                                torch.cat(actions), torch.cat(probabilities), advantage.flatten(), returns.flatten())
        metrics = dict(iteration=iteration, mode=mode, completed_episodes=diag["completed_episodes"],
                       successful_deliveries=task.speed_successes-success_before,
                       proxy_episode_fraction=diag["collision_episodes"]/max(diag["completed_episodes"], 1),
                       proxy_step_fraction=diag["collision_steps"]/max(diag["executed_steps"], 1),
                       collision_cost=diag["collision_cost"]/max(diag["executed_steps"], 1),
                       requested_stop_fraction=stop_requested/max(controlled_n, 1),
                       sent_stop_fraction=stop_sent/max(controlled_n, 1),
                       sent_stop_samples=stop_sent,
                       actual_speed_when_sent_stop=speed_at_stop/max(stop_sent, 1),
                       stopped_when_sent_stop=stopped_at_stop/max(stop_sent, 1),
                       held_when_sent_stop=held_at_stop/max(stop_sent, 1),
                       reward_mean=reward_total/(horizon*task.num_envs),
                       actual_stopped_fraction=stopped/max(controlled_n, 1),
                       actual_moving_fraction=moving_n/max(controlled_n, 1),
                       held_fraction=held_n/max(controlled_n, 1), **update)
        totals["completed"] += task.speed_completed-completed_before
        totals["successes"] += task.speed_successes-success_before
        totals["proxy_collided"] += diag["collision_episodes"]
        totals["proxy_steps"] += diag["collision_steps"]
        totals["executed_steps"] += diag["executed_steps"]
        with (out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        if writer:
            for key, scalar in metrics.items():
                if isinstance(scalar, (float, int)) and key != "iteration":
                    writer.add_scalar("Speed/" + key, scalar, iteration)
            writer.flush()
        print("CARRY_SPEED " + json.dumps(metrics), flush=True)
        save_every = int(os.environ.get("CARRY_SPEED_SAVE_EVERY", "5"))
        if mode == "train" and (iteration % save_every == 0 or iteration == first + iterations - 1):
            policy.save(out / f"speed_{iteration:06d}.pth", optimizer, iteration, args.checkpoint, metrics)
    summary = {**totals, "mode": mode, "executor": args.checkpoint, "seed": seed,
               "delivery_fraction": totals["successes"]/max(totals["completed"], 1),
               "proxy_episode_fraction": totals["proxy_collided"]/max(totals["completed"], 1),
               "proxy_step_fraction": totals["proxy_steps"]/max(totals["executed_steps"], 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if traces:
        np.savez_compressed(out / "trace.npz", **{key: np.stack([item[key] for item in traces]) for key in traces[0]},
                            fixed_routes=task.speed_paths.detach().cpu().numpy())
    if writer:
        writer.close()
    print("CARRY_SPEED_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

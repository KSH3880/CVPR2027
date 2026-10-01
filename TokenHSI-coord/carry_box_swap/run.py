"""Compare frozen original Carry with and without mid-episode reassignment."""

import os
import sys
import json
from pathlib import Path
from isaacgym import gymapi  # must precede torch
import numpy as np
import torch

COORD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COORD / "tokenhsi"))
import run as tokenhsi_run
import utils.parse_task as registry
from utils.config import get_args, load_cfg, set_seed
from carry_box_swap.env import HumanoidCarryBoxSwapProbe
from carry_box_swap.assignment import assignment_rows
from carry_planner.held_state import observed_box_held

registry.HumanoidCarryBoxSwapProbe = HumanoidCarryBoxSwapProbe
if os.environ.get("CARRY_BOX_SWAP_EXECUTOR", "stage1") == "ms18":
    from carry_box_swap.ms18_env import HumanoidMACarryBoxSwapMS18
    registry.HumanoidMACarryBoxSwapMS18 = HumanoidMACarryBoxSwapMS18


def snapshot(task):
    root = task.humanoid_rows(task._humanoid_root_states).reshape(task.num_envs, 2, -1)
    box = task.humanoid_rows(task._box_states).reshape(task.num_envs, 2, -1)
    bodies = task.humanoid_rows(task._rigid_body_pos).reshape(task.num_envs, 2, -1, 3)
    hands = bodies[:, :, task._key_body_ids[[0, 1]]].mean(2)
    size = task._box_lib._box_size.reshape(task.num_envs, 2, 3)
    physical_held = observed_box_held(root[..., :2], box[..., :3], size[..., 2], hands)
    index = assignment_rows(task.box_assignment)
    assigned_box = box.reshape(-1, 13)[index].reshape_as(box)
    assigned_size = size.reshape(-1, 3)[index].reshape_as(size)
    assigned_held = observed_box_held(root[..., :2], assigned_box[..., :3], assigned_size[..., 2], hands)
    goal = task._box_tar_pos.reshape(task.num_envs, 2, 3)
    result = dict(root=root.detach().clone(), box=box.detach().clone(), goal=goal.detach().clone(),
                physical_held=physical_held, assigned_held=assigned_held,
                assignment=task.box_assignment.clone(), min_body_distance=task.agent_min_dist().reshape(task.num_envs, 2)[:, 0])
    if hasattr(task, "swap_paths"):
        result["path"] = task.swap_paths.detach().clone()
    return result


def main():
    args = get_args()
    cfg, cfg_train, _ = load_cfg(args)
    seed = set_seed(cfg_train["params"].get("seed", 0), False)
    cfg_train["params"]["seed"] = seed
    cfg_train["params"]["config"]["seed"] = seed
    cfg_train["params"]["config"]["train_dir"] = args.output_path
    cfg["env"]["motion_file"] = args.motion_file
    tokenhsi_run.args, tokenhsi_run.cfg, tokenhsi_run.cfg_train = args, cfg, cfg_train
    runner = tokenhsi_run.build_alg_runner(tokenhsi_run.RLGPUAlgoObserver())
    runner.load(cfg_train); runner.reset()
    player = runner.create_player(); player.restore(args.checkpoint)
    player.model.eval().requires_grad_(False)
    player.env.reset()
    task = player.env.task
    out = Path(os.environ["CARRY_BOX_SWAP_OUTPUT"])
    skill = os.environ["CARRY_BOX_SWAP_SKILL"]
    n = task.num_envs
    enabled = torch.arange(n, device=task.device) % 2 == 1
    active = torch.ones(n, device=task.device, dtype=torch.bool)
    swapped = torch.zeros_like(active)
    ever_held = torch.zeros(n, 2, device=task.device, dtype=torch.bool)
    delivered = torch.zeros_like(ever_held)
    collided = torch.zeros_like(active)
    fallen = torch.zeros_like(active)
    swap_tick = torch.full((n,), -1, device=task.device, dtype=torch.long)
    success_count = torch.zeros(n, 2, device=task.device, dtype=torch.long)
    trace, events = [], []
    steps = int(os.environ.get("CARRY_BOX_SWAP_STEPS", "600"))
    switch_after = int(os.environ.get("CARRY_BOX_SWAP_AFTER", "30" if skill == "carryWith" else "6"))
    print(f"[box-swap] executor={os.environ.get('CARRY_BOX_SWAP_EXECUTOR', 'stage1')} frozen={args.checkpoint} skill={skill} groups=even-control,odd-swap "
          f"after={switch_after} goals_follow_box={task.goals_follow_box}", flush=True)
    for tick in range(steps):
        before = snapshot(task)
        eligible = before["physical_held"].all(-1) if skill == "carryWith" else ~before["physical_held"].any(-1)
        change = enabled & active & ~swapped & eligible & (tick >= switch_after)
        if change.any():
            ids = change.nonzero(as_tuple=False).flatten()
            original_box = task._box_states.clone()
            original_root = task._humanoid_root_states.clone()
            physical_alias = task._box_states
            old_obs = task._compute_task_obs(ids).clone()
            task.set_box_assignment(ids, torch.tensor([1, 0], device=task.device).expand(len(ids), -1))
            new_obs = task._compute_task_obs(ids)
            assert task._box_states is physical_alias
            assert torch.equal(task._box_states, original_box)
            assert torch.equal(task._humanoid_root_states, original_root)
            swapped[ids] = True; swap_tick[ids] = tick
            ever_held[ids] = False; success_count[ids] = 0; delivered[ids] = False
            event = dict(tick=tick, envs=ids.tolist(), physical_state_unchanged=True,
                         observation_change_norm=float((new_obs-old_obs).norm()))
            if hasattr(task, "swap_paths"):
                event["path_change_norm"] = float((task.swap_paths[ids]-before["path"][ids]).norm())
                state = task.planner_state(ids)
                path = task.swap_paths[ids]
                assert torch.allclose(path[:, :, 0], state.root_xy, atol=1e-5)
                assert torch.allclose(path[:, :, -1], state.goal_xy, atol=1e-5)
                walking = state.held < .5
                assert torch.allclose(path[:, :, 16][walking], state.box_xyz[..., :2][walking], atol=1e-5)
                event["assigned_route_anchors_verified"] = True
            events.append(event); print("BOX_SWAP_EVENT " + json.dumps(event), flush=True)
        task._compute_observations()
        obs = torch.clamp(task.obs_buf, -player.env.clip_obs, player.env.clip_obs).to(player.device)
        player.get_batch_size(obs, 1)
        with torch.no_grad(): action = player.get_action({"obs": obs}, is_determenistic=True)
        _, _, done, _ = player.env.step(action)
        now = snapshot(task)
        ever_held |= now["assigned_held"] & active[:, None]
        index = assignment_rows(task.box_assignment)
        box = now["box"].reshape(-1, 13)[index].reshape(n, 2, 13)
        goal = (now["goal"].reshape(-1, 3)[index].reshape(n, 2, 3)
                if task.goals_follow_box else now["goal"])
        placed = ((box[..., :3]-goal).norm(dim=-1) < .3) & (box[..., 7:10].norm(dim=-1) < .2) & ~now["assigned_held"]
        success_count = torch.where(placed & active[:, None], success_count+1, torch.zeros_like(success_count))
        delivered |= (success_count >= 10) & active[:, None]
        collided |= (now["min_body_distance"] < .3) & active
        done_env = done.reshape(n, 2).any(-1)
        fallen |= task._terminate_buf.bool() & active
        trace.append({key: value.cpu().numpy() for key, value in {**now, "active": active.clone(), "swapped": swapped.clone(), "done": done_env}.items()})
        active &= ~done_env
        if tick % 60 == 0:
            print(f"BOX_SWAP_PROGRESS tick={tick} active={int(active.sum())} swapped={int(swapped.sum())}", flush=True)
        if not active.any(): break
    groups = {}
    for name, mask in (("control", ~enabled), ("swap", enabled)):
        count = int(mask.sum())
        groups[name] = dict(environments=count, swap_triggered=int(swapped[mask].sum()),
            both_assigned_held_ever=int(ever_held[mask].all(-1).sum()),
            both_delivered=int(delivered[mask].all(-1).sum()),
            collision_episodes=int(collided[mask].sum()), fallen_episodes=int(fallen[mask].sum()))
    mask = enabled & swapped
    groups["triggered_swap_only"] = dict(environments=int(mask.sum()),
        both_assigned_held_ever=int(ever_held[mask].all(-1).sum()),
        both_delivered=int(delivered[mask].all(-1).sum()),
        collision_episodes=int(collided[mask].sum()), fallen_episodes=int(fallen[mask].sum()))
    summary = dict(checkpoint=str(Path(args.checkpoint).resolve()), skill=skill, seed=seed,
        goals_follow_box=task.goals_follow_box, executor=os.environ.get("CARRY_BOX_SWAP_EXECUTOR", "stage1"), steps=len(trace), groups=groups, events=events,
        swap_tick=swap_tick.cpu().tolist(),
        note="held/delivery/body-distance geometry proxies; first episodes only; n=1 setup probe")
    np.savez_compressed(out / "trace.npz", **{key: np.stack([item[key] for item in trace]) for key in trace[0]})
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("BOX_SWAP_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__": main()

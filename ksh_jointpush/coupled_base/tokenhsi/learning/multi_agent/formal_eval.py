"""First-episode formal scoring; no automatic reset or reward-success reuse."""
import json
from pathlib import Path
import numpy as np
import torch
from utils.evaluation_protocol import HORIZON, TASK_NAMES, spatial_success, predecessors
from utils.task_role_spec import task_graph_packet


def freeze_fallen(task, newly_fallen):
    # Keep the physical actor and its contacts; remove ALL joint motor control.
    for env, agent in newly_fallen.nonzero(as_tuple=False).cpu().tolist():
        handle = task.humanoid_handles_all[env][agent]
        props = task.gym.get_actor_dof_properties(task.envs[env], handle)
        props['stiffness'][:] = 0
        props['damping'][:] = 0
        props['driveMode'][:] = 0  # gymapi.DOF_MODE_NONE
        task.gym.set_actor_dof_properties(task.envs[env], handle, props)


@torch.no_grad()
def run_formal_eval(player):
    task = player.env.task
    args = task.cfg['args']
    player._reward_debug = False
    path = Path(args.output_path)
    path.mkdir(parents=True, exist_ok=True)
    n, m = task.num_envs, task.num_agents
    horizon=task.cfg['env']['episodeLength']
    box_values=getattr(task,'_evaluation_box_values',task._logical_box_values)
    obs = player.env_reset()
    player.get_batch_size(obs['obs'], 1)
    packet = task_graph_packet(task.relation_runtime.graph, n).reshape(n, m, 5).long().clone()
    types = packet[..., 1]
    parent = predecessors(task.relation_runtime.graph, n, task.device)
    dependent = parent >= 0
    batch = torch.arange(n, device=task.device)[:, None]
    alive = packet[..., 0].bool()
    ever = torch.zeros(n, m, 2, dtype=torch.bool, device=task.device)
    simultaneous = torch.zeros_like(ever)
    fall_step = torch.full((n, m), -1, device=task.device, dtype=torch.long)
    sizes = box_values(task._box_size).clone()
    goals = task._task_goal_positions().clone()
    groups=(obs["obs"][:,-m:].long().clone() if getattr(task,"_stage2_push",False)
            else torch.arange(m,device=task.device)[None].expand(n,-1))
    initial = dict(groups=groups.cpu(),packet=packet.cpu(), predecessor=parent.cpu(), sizes=sizes.cpu(), goals=goals.cpu(),
                   roots=task._humanoid_root_states.cpu().clone(), boxes=box_values(task._box_states).cpu().clone(),
                   dof_pos=task._dof_pos.cpu().clone(), dof_vel=task._dof_vel.cpu().clone(),
                   logical_box_order=task._logical_box_order.cpu().clone())
    torch.save(initial, path/'initial.pt')
    roots_trace, boxes_trace, valid_trace = [], [], []
    for step in range(1, horizon+1):
        action = player.get_action(obs, True)
        action = action.reshape(n, m, -1) * alive[..., None]
        obs, _, done, info = player.env_step(player.env, action.flatten(0, 1))
        if not isinstance(obs, dict):
            obs = {"obs": obs}
        roots = task._humanoid_root_states[..., :3]
        boxes = box_values(task._box_states)[..., :7]
        current = spatial_success(packet, roots, boxes, sizes, goals) & alive[..., None]
        ever |= current
        simultaneous |= current & current[batch, parent.clamp_min(0)] & dependent[..., None]
        roots_trace.append(roots.cpu().numpy().copy())
        boxes_trace.append(boxes.cpu().numpy().copy())
        valid_trace.append(alive.cpu().numpy().copy())
        fallen = task._terminate_buf.reshape(n, m).bool() & alive
        fall_step[fallen] = step
        freeze_fallen(task, fallen)
        alive &= ~fallen
        if step % 100 == 0:
            print(f'[formal eval] step={step}/{horizon} active={alive.sum().item()}/{n*m}', flush=True)
        if not alive.any():
            break
    both_ever = ever & ever[batch, parent.clamp_min(0)] & dependent[..., None]
    scores = {}
    for criterion, idx in [('point', 0), ('region', 1)]:
        per_task = {}
        for kind, name in enumerate(TASK_NAMES):
            mask = (types == kind) & packet[..., 0].bool()
            pair_mask = mask & dependent
            count = int(mask.sum())
            pair_count = int(pair_mask.sum())
            per_task[name] = dict(trials=count, success=int(ever[..., idx][mask].sum()),
                rate=float(ever[..., idx][mask].double().mean()) if count else None,
                pair_trials=pair_count,
                follower_individual_success=int(ever[..., idx][pair_mask].sum()),
                both_ever_success=int(both_ever[..., idx][pair_mask].sum()),
                simultaneous_success=int(simultaneous[..., idx][pair_mask].sum()))
        scores[criterion] = per_task
    data = dict(protocol=args.eval_protocol, single_task=args.eval_single_task or None,
                experiment=Path(args.cfg_train).stem, seed=args.seed, num_envs=n, num_agents=m, num_objects=task.num_objects,
                horizon=horizon, simulated_steps=step, deterministic=True, control_dt=task.dt, scores=scores,
                falls=int((fall_step >= 0).sum()), fall_rate=float((fall_step >= 0).double().mean()),
                source_config=args.resolved_env_config)
    (path/'result.json').write_text(json.dumps(data, indent=2))
    np.savez_compressed(path/'trace.pending.npz', roots=np.stack(roots_trace), boxes=np.stack(boxes_trace),
                        valid=np.stack(valid_trace), packet=packet.cpu().numpy(), sizes=sizes.cpu().numpy(),
                        goals=goals.cpu().numpy(), predecessor=parent.cpu().numpy(),groups=groups.cpu().numpy(),
                        fall_step=fall_step.cpu().numpy(), ever=ever.cpu().numpy(),
                        both_ever=both_ever.cpu().numpy(), simultaneous=simultaneous.cpu().numpy())
    (path/'trace.pending.npz').replace(path/'trace.npz')
    print('[formal eval result]', json.dumps(data), flush=True)
    return data

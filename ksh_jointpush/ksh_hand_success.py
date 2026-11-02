"""Evaluation-only search for goal arrivals with sustained eight-hand pushing."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from ksh_jointpush.ksh_group_success import scene_snapshot, snapshot
from ksh_jointpush.ksh_success_cases import first_hits
from ksh_jointpush.ksh_live_control import write_json


def face_distances(hands, boxes, sizes):
    """3-D distance of each hand center to the finite rear face, without goal saturation."""
    delta = hands - boxes[..., None, :3]
    xyz = -boxes[..., None, 3:6].expand_as(delta)
    uv = torch.cross(xyz, delta, dim=-1)
    local = delta + 2 * (boxes[..., None, 6:7] * uv + torch.cross(xyz, uv, dim=-1))
    dx = (local[..., 0].abs() - sizes[..., None, 0] / 2).clamp_min(0)
    dz = (local[..., 2].abs() - sizes[..., None, 2] / 2).clamp_min(0)
    distance = (dx.square() + dz.square() + (local[..., 1]+sizes[..., None, 1]/2).square()).sqrt()
    return torch.where(local[..., 1] <= 0, distance, torch.full_like(distance, 1000.))


def longest_run(mask):
    best = current = 0
    for value in np.asarray(mask, bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def assess(distances, push_hits, success, box_positions, goal, dt, tolerance=.08,
           window_seconds=2., near_fraction=.9, force_fraction=.5, progress_min=.25,
           gap_seconds=.2):
    """Only genuinely new official arrivals can pass; failed candidates remain labeled."""
    distances = np.asarray(distances)
    success = np.asarray(success, bool)
    if distances.shape != (len(success), 4, 2):
        raise ValueError('Expected exactly four people and two hands each')
    if not np.isfinite(distances).all() or not np.isfinite(box_positions).all():
        raise ValueError('Nonfinite hand or box trajectory')
    hits = first_hits(success)
    arrival = int(hits.max()) if (hits >= 0).all() else -1
    near = (distances <= tolerance).all(axis=(1, 2))
    physical = np.asarray(push_hits, bool).all(axis=-1)
    window = max(1, int(round(window_seconds/dt)))
    max_gap = int(round(gap_seconds/dt))
    goal_distance = np.linalg.norm(np.asarray(box_positions)-np.asarray(goal), axis=-1)
    best = None
    best_key = None
    for end in range(window, arrival+1):
        start = end-window+1  # frame0 initialization is never counted
        near_part = near[start:end+1]
        force_part = physical[start:end+1]
        value = dict(start_frame=start, end_frame=end,
            all_eight_near_fraction=float(near_part.mean()),
            all_four_physical_push_fraction=float(force_part.mean()),
            progress_m=float(goal_distance[start-1]-goal_distance[end]),
            longest_detachment_s=float(longest_run(~near_part)*dt))
        qualifies = (value['all_eight_near_fraction'] >= near_fraction-1e-9
                     and value['all_four_physical_push_fraction'] >= force_fraction-1e-9
                     and value['progress_m'] >= progress_min
                     and longest_run(~near_part) <= max_gap)
        # Prefer a passing window; a single high-nearness static pose cannot outrank it.
        key = (qualifies, value['all_eight_near_fraction'],
               value['all_four_physical_push_fraction'], value['progress_m'])
        if best_key is None or key > best_key:
            best_key, best = key, value
    arrival_near = bool(near[arrival]) if arrival >= 0 else False
    after = near[arrival:min(len(near), arrival+int(round(1/dt))+1)] if arrival >= 0 else []
    passed = bool(arrival_near and best_key is not None and best_key[0])
    return dict(passed=passed, first_hit_frames=hits.tolist(), arrival_frame=arrival,
                all_eight_near_at_arrival=arrival_near,
                max_hand_distance_at_arrival_m=float(distances[arrival].max()) if arrival >= 0 else None,
                all_eight_near_fraction_after_arrival_1s=float(np.mean(after)) if len(after) else 0.,
                after_arrival_observed_s=float(len(after)*dt),
                longest_all_eight_near_s=float(longest_run(near[1:])*dt),
                best_window=best)


def candidate_key(metrics):
    window = metrics['best_window'] or {}
    return (metrics['passed'], metrics['all_eight_near_at_arrival'],
            window.get('all_eight_near_fraction', 0.),
            window.get('all_four_physical_push_fraction', 0.),
            metrics['all_eight_near_fraction_after_arrival_1s'],
            window.get('progress_m', 0.))


def install():
    from learning.multi_agent.ma_players import MAPlayerContinuous

    @torch.no_grad()
    def run(player):
        task = player.env.task
        if task.viewer is not None or task._task_graph_preset != 'all_coupled' or task._eval_push_rsi:
            raise ValueError('Hand search requires headless all_coupled LOCO evaluation')
        output = Path(task.cfg['args'].output_path).resolve()
        output.mkdir(parents=True, exist_ok=True)
        cases_dir = output/'cases'
        candidates_dir = output/'candidates'
        cases_dir.mkdir(exist_ok=True)
        candidates_dir.mkdir(exist_ok=True)
        quota = int(os.environ.get('KSH_HAND_COUNT', '10'))
        rounds = int(os.environ.get('KSH_SUCCESS_ROUNDS', '64'))
        n, m, horizon = task.num_envs, task.num_agents, task.max_episode_length
        actors = task._root_states.shape[0]//n
        criteria = dict(tolerance_m=.08, window_seconds=2., all_eight_near_fraction=.9,
                        all_four_physical_push_fraction=.5, progress_min_m=.25,
                        max_detachment_seconds=.2, all_eight_near_at_arrival=True,
                        physical_force_thresholds=task._joint['contact'])
        cases, best_candidates = [], []
        trials = goal_trials = 0
        best_key = None
        diagnostic = (output/'diagnostics.jsonl').open('a', buffering=1)
        def report(state='running'):
            result = dict(state=state, cases=cases, best_candidates=best_candidates,
                completed_scene_trials=trials, all_four_goal_trials=goal_trials,
                requested_cases=quota, max_rounds=rounds, criteria=criteria,
                deterministic=True, init_mode='loco', category='4', policy_batch=n,
                checkpoint=str(task.cfg['args'].checkpoint), training_started=False)
            write_json(cases_dir/'cases.json', result)
            write_json(output/'status.json', result)
            return result
        report()
        for repeat in range(rounds):
            obs = player.env_reset()
            player.get_batch_size(obs['obs'], n)
            assert (task._groups == task._groups[:, :1]).all()
            initial = snapshot(task)
            group = task._groups.cpu().numpy().copy()
            goals = task._tar_pos.cpu().numpy().copy()
            assigned = task._agent_box_assignment.cpu().numpy().copy()
            roots = np.empty((horizon+1,n,actors,13), np.float32)
            dofs = np.empty((horizon+1,n,m,task.num_dof,2), np.float32)
            bodies = np.empty((horizon+1,n,m,task.num_bodies,3), np.float32)
            distances = np.empty((horizon+1,n,m,2), np.float32)
            physical = np.zeros((horizon+1,n,m), bool)
            flags = np.zeros((horizon+1,n,m), bool)
            box_pos = np.empty((horizon+1,n,3), np.float32)
            alive = np.ones(n, bool)
            resets = []
            def record(frame, initial_frame=False):
                roots[frame] = task._root_states.view(n,actors,13).cpu().numpy()
                dofs[frame] = task._dof_state.view(n,m,task.num_dof,2).cpu().numpy()
                body = task._kinematic_humanoid_rigid_body_states[..., :3] if initial_frame else task._rigid_body_pos
                bodies[frame] = body.cpu().numpy()
                hands = body[:,:,task._key_body_ids[:2]]
                boxes = task._assigned_box_values(task._box_states)
                sizes = task._assigned_box_values(task._box_size)
                distances[frame] = face_distances(hands, boxes, sizes).cpu().numpy()
                box_pos[frame] = boxes[:,0,:3].cpu().numpy()
                flags[frame] = task.relation_runtime.done.cpu().numpy()
                if not initial_frame:
                    physical[frame] = task._push_hits.cpu().numpy()
            record(0, True)
            for frame in range(1,horizon+1):
                obs = player.env_reset(resets)
                action = player.get_action(obs, True)
                obs, _, done, _ = player.env_step(player.env, action)
                record(frame)
                ending = done.reshape(n,m).bool().any(-1).cpu().numpy()
                finished = np.flatnonzero(ending & alive)
                for env_id in finished:
                    trials += 1
                    rows = slice(0,frame+1)
                    metrics = assess(distances[rows,env_id], physical[rows,env_id],
                                     flags[rows,env_id], box_pos[rows,env_id], goals[env_id,0], float(task.dt))
                    hit_goal = metrics['arrival_frame'] >= 0
                    goal_trials += int(hit_goal)
                    meta = dict(category='4', repeat=repeat, env_id=int(env_id), frames=frame+1,
                        groups=group[env_id].tolist(), first_hit_frames=metrics['first_hit_frames'],
                        simultaneous=bool(flags[rows,env_id].all(-1).any()),
                        final_all=bool(flags[frame,env_id].all()), checkpoint=str(task.cfg['args'].checkpoint),
                        policy_batch=n, criteria=criteria, hand_quality=metrics)
                    diagnostic.write(json.dumps(meta)+'\n')
                    key = candidate_key(metrics)
                    improved = hit_goal and (best_key is None or key > best_key)
                    accepted = metrics['passed'] and len(cases) < quota
                    if accepted or improved:
                        directory = cases_dir if accepted else candidates_dir
                        name = ('case_'+str(len(cases)+1).zfill(3)) if accepted else ('candidate_'+str(trials).zfill(6))
                        path = directory/(name+'.pt')
                        saved = scene_snapshot(initial, env_id)
                        saved['success'] = torch.from_numpy(flags[rows,env_id].copy())
                        torch.save(saved, path)
                        trajectory = directory/(name+'.npz')
                        np.savez_compressed(trajectory, roots=roots[rows,env_id], dofs=dofs[rows,env_id],
                            bodies=bodies[rows,env_id], success=flags[rows,env_id],
                            hand_face_distance=distances[rows,env_id], physical_push=physical[rows,env_id],
                            box_positions=box_pos[rows,env_id], goals=goals[env_id],
                            box_sizes=initial['tensors']['_box_size'][env_id].numpy(),
                            groups=group[env_id], assignments=assigned[env_id],
                            origin=initial['tensors']['_env_origins'][env_id].numpy())
                        meta.update(file=str(path), trajectory=str(trajectory), dt=float(task.dt))
                        path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
                        if accepted:
                            cases.append(meta)
                            print('KSH_HAND_ACCEPTED',json.dumps(meta),flush=True)
                        if improved:
                            best_key = key
                            best_candidates.append(meta)
                            print('KSH_HAND_BEST_CANDIDATE',json.dumps(meta),flush=True)
                        report()
                alive[finished] = False
                resets = torch.as_tensor(np.flatnonzero(ending),device=player.device,dtype=torch.long)*m
                if frame % 300 == 0:
                    print('KSH_HAND_PROGRESS',repeat,frame,'strict',len(cases),'goal',goal_trials,'trials',trials,flush=True)
                if not alive.any() or len(cases) >= quota:
                    break
            report()
            if len(cases) >= quota:
                break
        diagnostic.close()
        result = report('complete' if len(cases)>=quota else 'budget_exhausted')
        print('KSH_HAND_SEARCH_DONE', json.dumps(result),flush=True)
        return result
    MAPlayerContinuous.run = MAPlayerContinuous.run_eval = run

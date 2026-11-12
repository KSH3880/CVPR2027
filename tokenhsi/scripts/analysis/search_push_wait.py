"""Find unforced two-person PUSH waiting examples; preserve initial states and poses."""
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tokenhsi'))
import isaacgym  # Must precede torch.
import numpy as np
import torch
from learning.multi_agent import viewer_diagnostics
from utils.task_role_spec import task_graph_packet


def first_run(mask, length):
    indices = np.flatnonzero(np.convolve(mask.astype(int), np.ones(length, int), 'valid') == length)
    return int(indices[0]) if len(indices) else -1


@torch.no_grad()
def search(player):
    task = player.env.task
    args = task.cfg['args']
    out = Path(args.output_path).resolve()
    out.relative_to(ROOT / 'output_etc/check')
    out.mkdir(parents=True, exist_ok=True)
    n, m = task.num_envs, task.num_agents
    assert m == 2 and args.task_graph == 'all_coupled'
    task._relation_cfg['diagnostics']['enabled'] = False
    obs = player.env_reset()
    player.get_batch_size(obs['obs'], 1)
    rows = torch.arange(n, device=task.device)
    ids = rows
    physical = task._agent_box_assignment[:, 0].clone()
    assert (task._agent_box_assignment == physical[:, None]).all()
    initial = dict(seed=args.seed, num_envs=n, num_agents=m, num_objects=task.num_objects,
                   checkpoint=str(Path(args.checkpoint).resolve()),
                   checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                   config=task.cfg, observation=obs['obs'].cpu().clone(),
                   roots=task._humanoid_root_states.cpu().clone(),
                   dof=task._dof_state.cpu().clone(), boxes=task._box_states.cpu().clone(),
                   sizes=task._box_size.cpu().clone(), goals=task._tar_pos.cpu().clone(),
                   center_goals=task._center_tar_pos.cpu().clone(),
                   groups=task._groups.cpu().clone(), regions=task._regions.cpu().clone(),
                   templates=task._templates.cpu().clone(), physical_indices=physical.cpu(),
                   logical_box_order=task._logical_box_order.cpu().clone(),
                   assignment=task._agent_box_assignment.cpu().clone(),
                   packet=task_graph_packet(task.relation_runtime.graph, n).cpu().clone(),
                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                   numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                   env_origins=task._env_origins.cpu().clone(), dt=float(task.dt))
    if hasattr(task,'_push_fixed_metadata'):
        initial['fixed_layout']=task._push_fixed_metadata
        (out/'fixed_layout.json').write_text(json.dumps(initial['fixed_layout'],indent=2)+'\n')
    torch.save(initial, out / 'initial.pt')
    if os.environ.get('PUSH_PREFILTER_ONLY') == '1':
        from prefilter_push_initial import select
        report = select(initial)
        temporary=out/'initial_prefilter.json.tmp'
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(out/'initial_prefilter.json')
        print('[initial distance/facing only]', args.seed, len(report['env_ids']), 'matches; no policy action', flush=True)
        return report
    traces = {k: [] for k in ('roots', 'dof', 'boxes', 'on_face', 'distance', 'force_contact', 'alive', 'native_success')}
    alive = torch.ones(n, dtype=torch.bool, device=task.device)
    compare_output = os.environ.get('COMPARE_OURS_OUTPUT', '')
    if compare_output:
        sys.path.insert(0, '/home/hwanhee/ksh/stage1_2_base_teacher/eval')
        from table2_metrics import surface_distance
        traces['surface_distance'] = []

    def capture():
        distance, _, hands, assigned, sizes, face = task._contact_geometry(ids, task._rigid_body_pos)
        for key, value in dict(roots=task._humanoid_root_states,
                dof=task._dof_state.view(n, -1, 2), boxes=task._box_states[rows, physical],
                on_face=face['on_face'], distance=distance, force_contact=task._push_hits,
                alive=alive, native_success=task.relation_runtime.done).items():
            traces[key].append(value.cpu().clone())
        if compare_output:
            traces['surface_distance'].append(surface_distance(hands.double(), assigned[:, 0, :7].double(), sizes[:, 0].double()).cpu())

    capture()
    started = time.time()
    for step in range(1, task.max_episode_length + 1):
        action = player.get_action(obs, True)
        obs, _, done, _ = player.env_step(player.env, action)
        if not isinstance(obs, dict):
            obs = {'obs': obs}
        alive &= ~task._terminate_buf.reshape(n, m).bool().any(-1)
        capture()
        if step % 50 == 0:
            status = dict(seed=args.seed, step=step, horizon=task.max_episode_length,
                          alive=int(alive.sum()), elapsed_seconds=time.time() - started)
            (out / 'progress.json').write_text(json.dumps(status, indent=2) + '\n')
            print('[push wait]', status, flush=True)
    trace = {k: torch.stack(v) for k, v in traces.items()}
    boxes = trace['boxes'][..., :3].numpy()
    goals = initial['center_goals'][:, 0, :3].numpy()
    errors = np.linalg.norm(boxes - goals[None], axis=-1)
    face = trace['on_face'].numpy()
    live = trace['alive'].numpy()
    candidates = []
    for env in range(n):
        arrivals = [first_run(face[:, env, person] & live[:, env], 3) for person in range(2)]
        if min(arrivals) < 0:
            continue
        early = int(np.argmin(arrivals))
        a, b = sorted(arrivals)
        both = first_run(face[:, env].all(-1) & live[:, env], 3)
        success = first_run((errors[:, env] <= .2) & live[:, env], 15)
        if both < 0 or success < both:
            continue
        motion = float(np.linalg.norm(boxes[a:b+1, env, :2] - boxes[a, env, :2], axis=-1).max())
        contact_rate = float(face[both:success+1, env].all(-1).mean())
        early_contact = float(face[a:b+1, env, early].mean())
        initial_error = float(errors[0, env])
        # Explicit post-hoc evidence thresholds; no policy/physics changes.
        qualifies = b-a >= 15 and motion <= .15 and early_contact >= .7 and contact_rate >= .7 and initial_error >= 1.
        candidates.append(dict(seed=args.seed, env_id=env, first_person=early,
            first_contact_step=a, second_contact_step=b, joint_contact_step=both,
            wait_steps=b-a, wait_seconds=(b-a)*float(task.dt),
            wait_box_displacement_max_m=motion, waiting_person_contact_fraction=early_contact,
            joint_contact_fraction_until_success=contact_rate, success_step=success,
            initial_goal_distance_m=initial_error, minimum_goal_distance_m=float(errors[:, env][live[:, env]].min()),
            qualifies=qualifies))
    candidates.sort(key=lambda c: (c['qualifies'], c['wait_steps'] if c['wait_box_displacement_max_m'] <= .15 else -1,
                                   c['joint_contact_fraction_until_success']), reverse=True)
    np.savez_compressed(out / 'metrics.npz', on_face=face, force_contact=trace['force_contact'].numpy(),
                        boxes=boxes, goal_errors=errors, alive=live, goals=goals,
                        native_success=trace['native_success'].numpy())
    for rank, candidate in enumerate(candidates[:5]):
        env = candidate['env_id']
        torch.save(dict(provenance=candidate, initial_file=str(out / 'initial.pt'),
                        trajectory={k: v[:, env].clone() for k, v in trace.items()}), out / f'candidate_{rank}.pt')
    report = dict(seed=args.seed, checkpoint=initial['checkpoint'], checkpoint_sha256=initial['checkpoint_sha256'],
                  num_envs=n, num_agents=m, horizon=task.max_episode_length, deterministic=True,
                  criteria=dict(sustained_arrival_frames=3, minimum_wait_frames=15, maximum_wait_box_motion_m=.15,
                                minimum_early_contact_fraction=.7, minimum_joint_contact_fraction=.7,
                                goal_tolerance_m=.2, sustained_success_frames=15, minimum_initial_goal_distance_m=1.),
                  contact_definition='Both hands on assigned pushing face, native gate_face_tolerance; force hits separately saved',
                  comparison_status='Ours only; TokenHSI*/FSM not yet executed; seed alone does not guarantee identical scene',
                  qualifying_count=sum(c['qualifies'] for c in candidates), candidates=candidates,
                  elapsed_seconds=time.time()-started)
    (out / 'candidates.json').write_text(json.dumps(report, indent=2)+'\n')
    if compare_output:
        target = Path(compare_output).resolve()
        target.relative_to(ROOT/'output_etc/check')
        target.mkdir(parents=True, exist_ok=True)
        env_ids = sorted(c['env_id'] for c in candidates if c['qualifies'])
        saved = {k: trace[k][:, env_ids].clone() for k in ('roots', 'dof', 'boxes', 'on_face', 'surface_distance', 'force_contact')}
        saved['alive'] = trace['alive'][:, env_ids, None].expand(-1, -1, 2).clone()
        for key in ('frozen', 'released', 'arrived'):
            saved[key] = torch.zeros_like(saved['on_face'])
        verification = dict(method='ours', seed=args.seed, env_ids=env_ids, horizon=task.max_episode_length,
            initial_exact={'reference_created_here': True}, policy={'checkpoint': initial['checkpoint'], 'sha256': initial['checkpoint_sha256']},
            num_agents=2, num_envs=n, source_initial=str(out/'initial.pt'), elapsed_seconds=time.time()-started)
        torch.save(dict(trajectory=saved, env_ids=env_ids, verification=verification), target/'rollout.pt')
        (target/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print('[push wait complete]', args.seed, 'qualifying', report['qualifying_count'], flush=True)
    return report


viewer_diagnostics.run_viewer_diagnostics = search
if __name__ == '__main__':
    runpy.run_path(str(ROOT / 'tokenhsi/run.py'), run_name='__main__')

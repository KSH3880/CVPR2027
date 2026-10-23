"""Non-training comparison of PUSH geometry, force height and sampled-pose PD.

Force probes use an external resultant equivalent to equal forward forces at
the two symmetric hand points. They test mechanics, not a learned policy.
PD probes apply normal joint targets from the current shoulder-retargeted RSI.
All overrides are local to this process; configs and RSI banks are untouched.
"""
from isaacgym import gymapi, gymtorch
import copy
import json
import os
from pathlib import Path
import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.door_reference import rotate
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from env.tasks.multi_agent.open_push_reward import push_hand_point_distance

torch.set_num_threads(1)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
args = get_args()
assert args.test and args.eval and args.num_envs == 2 and args.headless
base, train, _ = load_cfg(args)
base['args'] = args
base['env']['motion_file'] = args.motion_file
base['env']['enableTrackInitState'] = True
bank = json.loads(Path(base['env']['openPushRSI']['bank_file']).read_text())
out = Path(args.output_path)
out.mkdir(parents=True, exist_ok=True)
reports = []


def measure(task, initial, size):
    state = task._box_states[:, 2]
    q = state[:, 3:7]
    tilt = torch.acos((1 - 2 * (q[:, 0]**2 + q[:, 1]**2)).clamp(-1, 1))
    # For a forward push the two lower front/rear edges capture the pivot.
    edge = state.new_tensor([[0., -size[1]/2, -size[2]/2],
                             [0., size[1]/2, -size[2]/2]])
    world = state[:, None, :3] + rotate(q[:, None].expand(-1, 2, -1),
                                       edge[None].expand(2, -1, -1))
    low = world[:, :, 2].argmin(-1)
    slip = world[torch.arange(2, device=state.device), low, 1] - (initial[:, 1] + edge[low, 1])
    return state[:, 1]-initial[:, 1], tilt, slip


for shape, size, heights in [('tall', [.8, .6, 1.3], [1.05, .8, .6, .4]),
                             ('cube', [1.3, 1.3, 1.3], [1.05, .8])]:
    cfg = copy.deepcopy(base)
    cfg['env']['pushBox']['size'] = size
    cfg['env']['box']['build']['baseSize'] = size
    # Build in-memory geometric proposals only; do not rewrite screened banks.
    cfg['_rsi_build'] = True
    sim = parse_sim_params(args, cfg, train)
    sim.physx.num_threads = 1
    set_seed(42)
    task = HumanoidMAOpenPush(cfg, sim, args.physics_engine, args.device, args.device_id, True)
    ids = torch.arange(2, device=task.device)
    task._task_graph_preset = 'push'
    props = task.gym.get_actor_rigid_body_properties(task.envs[0], task._box_handles[2])
    report = dict(shape=shape, size=size, actual_mass_kg=sum(p.mass for p in props),
                  force_probes=[], sampled_pose_pd=[])
    try:
        nbody = task._rigid_body_state.view(2, -1, 13).shape[1]
        box_body = 2*task.num_bodies + 2*task._door_body_count
        for height in heights:
            for total_force in [80., 120., 160., 200.]:
                task._reset_envs(ids)
                # Keep both humans away throughout the mechanics-only probe.
                task._humanoid_root_states[:, :, 0] -= 10.
                task._kinematic_humanoid_rigid_body_states[:, :, :, 0] -= 10.
                task._reset_env_tensors(ids)
                initial = task._box_states[:, 2, :3].clone()
                forces = torch.zeros(2, nbody, 3, device=task.device)
                positions = torch.zeros_like(forces)
                max_tilt = torch.zeros(2, device=task.device)
                max_slide = torch.zeros_like(max_tilt)
                max_dy = torch.zeros_like(max_tilt)
                for _ in range(60):
                    task.pre_physics_step(torch.zeros(4, 32, device=task.device))
                    for _ in range(task.control_freq_inv):
                        # Only env 0 receives force. Env 1 is the zero-force control.
                        state = task._box_states[0, 2]
                        q = state[3:7]
                        local = state.new_tensor([0., -size[1]/2, height-size[2]/2])
                        positions[0, box_body] = state[:3] + rotate(q[None], local[None])[0]
                        forces[0, box_body] = rotate(q[None], state.new_tensor([[0., total_force, 0.]]))[0]
                        task.gym.apply_rigid_body_force_at_pos_tensors(task.sim,
                            gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(positions), gymapi.ENV_SPACE)
                        task.gym.simulate(task.sim)
                        task.gym.fetch_results(task.sim, True)
                        task._refresh_sim_tensors()
                        dy, tilt, slip = measure(task, initial, size)
                        max_tilt = torch.maximum(max_tilt, tilt)
                        max_dy = torch.maximum(max_dy, dy)
                        max_slide = torch.maximum(max_slide, torch.where(tilt<=.35, slip, 0.))
                row = dict(height_m=height, total_forward_force_N=total_force,
                           max_center_forward_m=max_dy.tolist(), max_tilt_rad=max_tilt.tolist(),
                           max_bottom_edge_slide_while_upright_m=max_slide.tolist(),
                           final_forward_m=(task._box_states[:, 2, 1]-initial[:, 1]).tolist())
                report['force_probes'].append(row)
                print('FORCE_PROBE', shape, json.dumps(row), flush=True)
        # Same source frames in each case, with the current RSI shoulder retarget.
        selected = []
        for mid in [0, 1]:
            clip = [r for r in bank['candidates']['push'] if r['motion_id']==mid]
            selected.extend(clip[round(i*(len(clip)-1)/5)] for i in range(6))
        for height in ([] if os.environ.get('PHYSICS_FORCE_ONLY')=='1' else
                       ([1.05, .8] if shape=='cube' else [1.05, .8, .6])):
            task._relation_cfg['push']['hand_target_height'] = height
            for source in selected:
                task._reset_envs(ids)
                sample = {k: torch.tensor([source[k]]*2, device=task.device) for k in source}
                task._apply_joint_rsi(ids, torch.zeros_like(ids), 'push', sample)
                task._reset_task(ids); task._reset_env_tensors(ids); task._sync_door_reset_states(ids)
                task._compute_observations(ids); task._init_amp_obs(ids)
                initial = task._box_states[:, 2, :3].clone()
                maxtilt = torch.zeros(2, device=task.device); maxslide = torch.zeros_like(maxtilt)
                maxdy = torch.zeros_like(maxtilt); contacts = torch.zeros_like(maxtilt)
                first_distance = None
                for step in range(60):
                    task.pre_physics_step(torch.zeros(4, 32, device=task.device))
                    task._physics_step(); task.post_physics_step()
                    dy, tilt, slip = measure(task, initial, size)
                    maxtilt = torch.maximum(maxtilt, tilt); maxdy = torch.maximum(maxdy, dy)
                    stable = (tilt<=.35) & (task._humanoid_root_states[:, 0, 2]>.65)
                    maxslide = torch.maximum(maxslide, torch.where(stable, slip, 0.))
                    contacts += task.relation_runtime.push_contact[:, 0]
                    if first_distance is None:
                        hands = task._rigid_body_pos[:, :, task._key_body_ids[:2]]
                        first_distance = push_hand_point_distance(hands, task._box_states[:, 2:], size,
                            task._relation_cfg['push'])[:, 0].tolist()
                    if task.reset_buf.any(): break
                row = dict(**source, height_m=height, steps=step+1,
                    first_hand_target_distance_m=first_distance, paired_contact_steps=contacts.tolist(),
                    max_center_forward_m=maxdy.tolist(), max_tilt_rad=maxtilt.tolist(),
                    max_bottom_edge_slide_while_upright_m=maxslide.tolist(), reset=task.reset_buf.tolist())
                report['sampled_pose_pd'].append(row)
                print('POSE_PD', shape, json.dumps(row), flush=True)
        reports.append(report)
        (out/(shape+'.json')).write_text(json.dumps(report, indent=2)+'\n')
    finally:
        task.gym.destroy_sim(task.sim)

(out/'comparison.json').write_text(json.dumps(dict(training=False,
    notes='External-force probes isolate mechanics; pose PD uses current shoulder-retargeted RSI poses, not a learned policy; AMP experts remain unmodified.',
    cases=reports), indent=2)+'\n')

"""Non-training Door stability, humanoid-contact and observation/reward audit."""
from isaacgym import gymapi, gymtorch
import json
import math
import os
from pathlib import Path
import torch
# This audit must not occupy the server's CPU pool.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import yaml
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.door_reference import DoorReferenceBank, align_reference, multiply, rotate
from env.tasks.multi_agent.humanoid_ma_door import HumanoidMADoor

args = get_args()
assert args.task == 'HumanoidMADoor' and args.test and args.eval and args.num_envs == 2
cfg, train, _ = load_cfg(args)
cfg['args'] = args
cfg['sim']['physx']['num_threads'] = 1
args.num_threads = 1
# Load a manipulation pose solely for the scripted contact probe. There is no
# policy, optimizer, altered Door asset or direct force/torque applied to a door.
cfg['env']['motion_file'] = args.motion_file
cfg['env']['enableTrackInitState'] = True
set_seed(42)
task = HumanoidMADoor(cfg, parse_sim_params(args, cfg, train), args.physics_engine,
                      args.device, args.device_id, args.headless)
output = Path(args.output_path)
output.mkdir(parents=True, exist_ok=True)
direction = cfg['env']['door'].get('openingDirection', 'push')
opening_sign = -1 if direction == 'pull' else 1
ids = torch.arange(2, device=task.device)
results = {'cpu_limits': {'torch_threads': torch.get_num_threads(), 'physx_threads': 1, 'affinity': sorted(os.sched_getaffinity(0))}, 'device': task.device, 'opening_direction': direction, 'notes': 'Scripted humanoid pose/contact, not learned-policy success.', 'checks': {}}
traces = []

def step():
    task.pre_physics_step(torch.zeros(4, task.num_dof, device=task.device))
    for _ in range(task.control_freq_inv):
        task.gym.simulate(task.sim)
        if args.device == "cpu":
            task.gym.fetch_results(task.sim, True)
    task.gym.fetch_results(task.sim, True)
    task.post_physics_step()

def reset(angles):
    task._reset_envs(ids)
    # Keep humans far away for isolated Door and representation tests.
    task._humanoid_root_states[..., 0] -= 10.
    task._humanoid_root_states[..., 7:] = 0.
    task.reset_door_angles(ids, torch.tensor(angles, device=task.device), apply_to_sim=False)
    task._reset_env_tensors(ids)
    task._sync_door_reset_states(ids)

def door_contact_force():
    forces = gymtorch.wrap_tensor(task.gym.acquire_net_contact_force_tensor(task.sim))
    forces = forces.view(2, -1, 3)[:, task.num_agents * task.num_bodies:]
    return forces[:, :task.num_objects * task._door_body_count].view(2, 2, task._door_body_count, 3).norm(dim=-1).amax(-1)

def correspondence():
    if task._door_approach_spec and not task._door_approach_spec.get('gate_enabled', True):
        assert task.extras['door_amp_weight'].eq(1).all(), 'Disabled AMP gate must stay at one'
    angle = task.door_hinge_angle
    panel = task._door_panel_states
    q = torch.zeros_like(panel[..., 3:7]); q[..., 2] = torch.sin(opening_sign * angle / 2); q[..., 3] = torch.cos(angle / 2)
    expected = task._box_states[..., :3] + rotate(q, q.new_tensor(task.door_local_handle_offset).expand(2, 2, 3))
    handle_error = (task.door_handle_position - expected).abs().max().item()
    quaternion_error = (1 - (panel[..., 3:7] * q).sum(-1).abs()).abs().max().item()
    assert handle_error < 2e-4, ('handle FK', handle_error)
    assert quaternion_error < 2e-5, ('panel rotation', quaternion_error)
    # Public observation layout: H223*2, O35*2, G1*2, pose7*6, graph10.
    obs = task.obs_buf
    objects = obs[:, 446:516].view(2, 2, 35)
    torch.testing.assert_close(objects[..., 30], angle, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(objects[..., 31], task.door_hinge_velocity, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(objects[..., 32:35], expected.new_tensor(task.door_local_handle_offset).expand(2, 2, 3), atol=2e-4, rtol=1e-5)
    poses = obs[:, 518:560].view(2, 6, 7)
    torch.testing.assert_close(poses[:, 2:4, :3], panel[..., :3], atol=2e-4, rtol=1e-5)
    torch.testing.assert_close(poses[:, 2:4, 3:], panel[..., 3:7], atol=2e-4, rtol=1e-5)
    d = cfg['env']['relationReward']['door']
    success = angle >= d['target_angle'] - d['angle_tolerance']
    progress = (angle / d['target_angle']).clamp(0, 1)
    expected_progress_reward = torch.where(success, torch.ones_like(progress), progress) * .2
    reward = task.relation_runtime.last_result
    torch.testing.assert_close(reward['phi_raw'], progress, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(reward['progress_component'], expected_progress_reward, atol=2e-5, rtol=1e-5)
    assert torch.equal(task.relation_runtime.done, success)
    torch.testing.assert_close(reward['success_component'], success.float() * .2)
    task._compute_metrics_evaluation()
    assert torch.equal(task._success_buf.view(2, 2).bool(), success)
    terms = task.extras['reward_terms'].view(2, 2, -1)
    roots = task._humanoid_root_states[..., :3]
    root_distance = (roots[..., :2] - task.door_handle_position[..., :2]).norm(dim=-1)
    state_scale = cfg['env']['relationReward']['state_reward_weight'] / 2
    approach = state_scale / (1 + (root_distance - d['approach_delta']).clamp_min(0) / d['approach_sigma'])
    hands = task._rigid_body_pos[:, :, task._key_body_ids[:2]]
    hand_squared_distance = (hands - task.door_handle_position[:, :, None]).square().sum(-1).amin(-1)
    interaction = state_scale * torch.exp(-d['hand_distance_scale'] * hand_squared_distance)
    approach = torch.where(success, torch.full_like(approach, state_scale), approach)
    interaction = torch.where(success, torch.full_like(interaction, state_scale), interaction)
    own = approach + interaction + expected_progress_reward + success.float() * .2
    power = -cfg['env']['power_coefficient'] * (task.dof_force_tensor * task._dof_vel).abs().sum(-1)
    separation = (roots[:, 0, :2] - roots[:, 1, :2]).norm(dim=-1)
    collision = -cfg['env']['agentCollisionCoeff'] * ((cfg['env']['agentCollisionDist'] - separation).clamp_min(0) / cfg['env']['agentCollisionDist'])[:, None].expand(-1, 2)
    sharing = cfg['env']['relationReward']['task_sharing']
    total = sharing['self'] * own + sharing['teammate'] * own.flip(-1) + power + collision
    torch.testing.assert_close(terms[..., 0], approach, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(terms[..., 1], interaction, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(terms[..., 4], power, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(terms[..., 5], collision, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(terms[..., 6], total, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(task.rew_buf.view(2, 2), total, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(terms[..., 2], expected_progress_reward, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(terms[..., 3], success.float() * .2)
    assert torch.isfinite(obs).all() and torch.isfinite(task.rew_buf).all()
    return {'handle_max_error_m': handle_error, 'rotation_max_error': quaternion_error, 'total_reward_max_error': (task.rew_buf.view(2, 2) - total).abs().max().item()}

try:
    # Twenty seconds, closed and partially open, zero hinge velocity/no actuation.
    reset([[0., .6], [0., .6]])
    initial = task.door_hinge_angle.clone()
    drift = 0.; max_force = 0.; min_human_door_xy_gap = float('inf')
    for i in range(600):
        step()
        drift = max(drift, (task.door_hinge_angle - initial).abs().max().item())
        max_force = max(max_force, door_contact_force().max().item())
        # GPU contact tensors also include Door-floor/frame contacts. Prove
        # human separation geometrically rather than treating all forces as hands.
        gap = (task._rigid_body_pos[..., 0].amax(2)[:, :, None] - task._box_states[:, None, :, 0]).abs().amin().item()
        min_human_door_xy_gap = min(min_human_door_xy_gap, gap)
    assert drift < 1e-4, ('spontaneous opening', drift)
    assert min_human_door_xy_gap > 5., ('human too close for isolated check', min_human_door_xy_gap)
    results['checks']['no_contact_20s'] = {'max_drift_rad': drift, 'max_total_door_contact_force_n_including_floor': max_force, 'min_human_door_x_gap_m': min_human_door_xy_gap, **correspondence()}
    print('NO_CONTACT_OK', results['checks']['no_contact_20s'], flush=True)

    if cfg['env']['door'].get('hinge'):
        hinge = cfg['env']['door']['hinge']
        props = task.gym.get_actor_dof_properties(task.envs[0], task._box_handles[0])
        assert abs(float(props['damping'][0]) - hinge['damping']) < 1e-6
        assert abs(float(props['friction'][0]) - hinge['friction']) < 1e-6
        coast = []
        for velocity in (.25, 2.64667, 3.):
            reset([[.1, 0.], [.1, 0.]])
            task._door_dof_state[:, 0, 1] = velocity
            task._reset_env_tensors(ids)
            task._sync_door_reset_states(ids)
            peak = .1
            for _ in range(240):
                step()
                peak = max(peak, task.door_hinge_angle[:, 0].max().item())
            assert peak < 1.15, ('impulse coast reached success', velocity, peak)
            assert task.door_hinge_velocity[:, 0].abs().max() < .01
            assert task.door_hinge_angle[:, 1].abs().max() < 1e-4
            coast.append({'initial_velocity_rad_s': velocity, 'peak_angle_rad': peak,
                          'final_velocity_rad_s': task.door_hinge_velocity[:, 0].mean().item()})
        results['checks']['hinge_impulse_coast'] = {'properties': hinge, 'cases': coast}
        print('HINGE_COAST_OK', results['checks']['hinge_impulse_coast'], flush=True)

    # Threshold, saturation and each owner's binding across actual physics steps.
    samples = []
    for angles in ([[0., .5], [1.14, 1.16]], [[1.2, 1.6], [.4, .9]]):
        reset(angles); step()
        samples.append({'actual_angles': task.door_hinge_angle.tolist(), **correspondence()})
    results['checks']['state_observation_reward'] = samples
    print('STATE_OBSERVATION_REWARD_OK', samples, flush=True)

    reset([[0., 0.], [0., 0.]])
    # Commit the all-actor GPU reset before a later humanoid-only setter.
    step()
    assert task.door_hinge_angle.abs().max() < 1e-4
    spec = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/door_open_unified_amp_on.yaml').read_text())['env']['doorReference'].copy()
    spec['interaction_skills'] = ['pickUp']
    from utils.motion_lib import MotionLib
    lib = MotionLib('tokenhsi/data/dataset_door_proxy.yaml', 'pickUp', task._dof_body_ids, task._dof_offsets, task._key_body_ids.cpu().numpy(), task.device)
    bank = DoorReferenceBank({'pickUp': lib}, task._key_body_ids[:2].cpu().tolist(),
                             task._key_body_ids[2:].cpu().tolist(), spec, 0.)
    motion_id = bank.pools['pickUp'][0][:1]
    motion_time = bank.pools['pickUp'][1][:1]
    root, rot, dof, vel, ang, dof_vel, key = lib.get_motion_state(motion_id, motion_time)
    body, body_rot, body_vel, body_ang = lib.get_motion_state_max(motion_id, motion_time)
    aligned, aligned_rot, _, _, yaw = align_reference(body, body_rot, body_vel, body_ang,
        rot, task.door_handle_position[:1, 0], task._key_body_ids[:2].cpu().tolist(), .16, spec['min_forward_reach'])
    relative = aligned[0] - aligned[0, 0]
    # Identify the aligned forward hand by distance to its target handle.
    hand_ids = task._key_body_ids[:2]
    selected_hand = hand_ids[(aligned[0, hand_ids] - task.door_handle_position[0, 0]).square().sum(-1).argmin()]
    root0 = aligned[0, 0].clone()
    quat0 = aligned_rot[0, 0].clone()
    hand_offset = relative[selected_hand].clone()
    task._every_env_init_dof_pos[0, 0] = dof[0]
    actor_ids = task._humanoid_actor_ids[:1, :1].flatten().contiguous()
    body_index = task.gym.get_actor_rigid_body_index(task.envs[0], task.humanoid_handles_all[0][0], int(selected_hand), gymapi.DOMAIN_ENV)
    door_indices = {task.gym.get_actor_rigid_body_index(task.envs[0], task._box_handles[0], j, gymapi.DOMAIN_ENV) for j in range(task._door_body_count)}
    handle_index = task.gym.get_actor_rigid_body_index(task.envs[0], task._box_handles[0], task._door_handle_index, gymapi.DOMAIN_ENV)
    hand_handle_pair_steps = 0
    contact_steps = 0; hand_pair_steps = 0; max_angle = 0.; max_door_force = 0.; max_hand_force = 0.
    initial_handle_y = task.door_handle_position[0, 0, 1].item()
    for i in range(180):
        # Advance a real humanoid hand into/along its own rotating door handle.
        # Only humanoid root/joint states are set; the Door remains passive.
        angle = opening_sign * task.door_hinge_angle[0, 0].clone()
        if direction == 'pull':
            # Prescribed slow hand arc; do not chase and amplify hinge motion.
            angle = angle.new_tensor(-min(i * .007, 1.25))
        spin = quat0.new_tensor([0., 0., 0., 1.])
        spin[2] = torch.sin(angle / 2); spin[3] = torch.cos(angle / 2)
        normal = spin.new_tensor([-torch.sin(angle), torch.cos(angle), 0.])
        stand_off = max(-.08, .16 - i * .004)
        contact_target = task.door_handle_position[0, 0]
        if direction == 'pull':
            stand_off = -max(.07, .09 - i * .001)
            contact_target = task._box_states[0, 0, :3] + rotate(spin, spin.new_tensor(task.door_local_handle_offset))
        wanted_hand = contact_target - normal * stand_off
        wanted_root = wanted_hand - rotate(spin, hand_offset)
        if direction == 'push':
            wanted_root[2] = root0[2]
        task._humanoid_root_states[0, 0, :3] = wanted_root
        task._humanoid_root_states[0, 0, 3:7] = multiply(spin, quat0)
        task._humanoid_root_states[0, 0, 7:] = 0.
        task._dof_pos[0, 0] = dof[0]; task._dof_vel[0, 0] = 0.
        task.gym.set_actor_root_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task._root_states), gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
        task.gym.set_dof_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task._dof_state), gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
        step()
        force = door_contact_force()[0, 0].item()
        hand_force = task._contact_forces[0, 0, selected_hand].norm().item()
        contact_steps += int(hand_force > .1)
        max_door_force = max(max_door_force, force); max_hand_force = max(max_hand_force, hand_force)
        if args.device == 'cpu':
            pairs = task.gym.get_env_rigid_contacts(task.envs[0])
            hand_pair_steps += int(any((int(c['body0']) == body_index and int(c['body1']) in door_indices) or (int(c['body1']) == body_index and int(c['body0']) in door_indices) for c in pairs))
            hand_handle_pair_steps += int(any({int(c['body0']), int(c['body1'])} == {body_index, handle_index} for c in pairs))
        max_angle = max(max_angle, task.door_hinge_angle[0, 0].item())
        traces.append({'step': i, 'hinge_angle_rad': task.door_hinge_angle[0, 0].item(), 'door_contact_force_n': force, 'hand_contact_force_n': hand_force, 'other_door_angle': task.door_hinge_angle[0, 1].item()})
        correspondence()
        # End the pull probe after reaching the goal instead of teleporting the
        # hand against the upper joint stop and generating artificial impulses.
        if direction == 'pull' and task.door_hinge_angle[0, 0] >= 1.2:
            break
    assert contact_steps > 0 and max_angle > .1, ('humanoid failed to open Door', contact_steps, max_angle)
    if cfg['env']['door'].get('hinge'):
        assert max_angle >= 1.15, ('sustained contact failed to reach success', max_angle)
    assert task.door_hinge_angle[0, 1].abs() < 1e-4
    assert task.door_hinge_angle[1].abs().max() < 1e-4
    if args.device == 'cpu':
        assert hand_pair_steps > 0, 'No actual hand-Door contact pairs'
        if direction == 'pull':
            assert hand_handle_pair_steps > 0, 'No actual hand-handle contact pairs'
    handle_y_delta = task.door_handle_position[0, 0, 1].item() - initial_handle_y
    assert opening_sign * handle_y_delta > .05, ('wrong physical opening direction', handle_y_delta)
    results['checks']['humanoid_contact'] = {'max_angle_rad': max_angle, 'handle_y_delta_m': handle_y_delta, 'hand_force_steps': contact_steps,
        'hand_handle_pair_steps_cpu': hand_handle_pair_steps if args.device == 'cpu' else None,
        'hand_door_pair_steps_cpu': hand_pair_steps if args.device == 'cpu' else None,
        'max_door_contact_force_n': max_door_force, 'max_hand_contact_force_n': max_hand_force,
        'untouched_door_angles': [task.door_hinge_angle[0, 1].item(), *task.door_hinge_angle[1].tolist()],
        'motion_id': motion_id.item(), 'motion_time': motion_time.item(), 'hand_body_id': selected_hand.item()}
    if direction == 'pull':
        # Release contact and verify free hinge motion remains finite/in bounds.
        task._humanoid_root_states[..., 0] -= 10.
        away_ids = task._humanoid_actor_ids.flatten().contiguous()
        task.gym.set_actor_root_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task._root_states), gymtorch.unwrap_tensor(away_ids), len(away_ids))
        low = float('inf'); high = -float('inf')
        for _ in range(120):
            step()
            assert torch.isfinite(task._rigid_body_state).all()
            low = min(low, task.door_hinge_angle.min().item())
            high = max(high, task.door_hinge_angle.max().item())
            assert low >= -.001 and high <= 1.701, ('hinge limit violation', low, high)
            correspondence()
        results['checks']['release_4s'] = {'min_angle_rad': low, 'max_angle_rad': high, 'final_angles': task.door_hinge_angle.tolist()}
    results['status'] = 'passed'
    print('HUMANOID_CONTACT_OK', results['checks']['humanoid_contact'], flush=True)
finally:
    (output / 'physics_audit.json').write_text(json.dumps(results, indent=2))
    (output / 'contact_trace.json').write_text(json.dumps(traces, indent=2))
    task.gym.destroy_sim(task.sim)

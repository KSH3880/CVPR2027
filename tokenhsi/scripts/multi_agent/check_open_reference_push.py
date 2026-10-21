"""Non-training CPU audit of reference joint PD versus door closing resistance.

Roots are free after reset. No human pose teleportation or external forces are
used in reference trials. Separate handle-force trials isolate door mechanics.
All spring overrides are in memory; YAML, RSI banks and checkpoints are untouched.
"""
from isaacgym import gymapi, gymtorch
import csv
import json
import math
import os
from pathlib import Path

import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.door_hold import closing_torque
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush


def main():
    torch.set_num_threads(1)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    args = get_args()
    assert args.test and args.eval and args.headless and args.num_envs == 2
    assert args.device == 'cpu', 'CPU PhysX required for actual contact pairs'
    cfg, train, _ = load_cfg(args)
    cfg['args'] = args
    cfg['env']['motion_file'] = args.motion_file
    cfg['env']['enableTrackInitState'] = True
    sim = parse_sim_params(args, cfg, train)
    sim.physx.num_threads = 1
    set_seed(42)
    task = HumanoidMAOpenPush(cfg, sim, args.physics_engine, args.device, args.device_id, True)
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    ids = torch.arange(2, device=task.device)
    owners = torch.zeros_like(ids)
    actions = torch.zeros(4, task.num_dof, device=task.device)
    task._task_graph_preset = 'open'
    task.render = lambda *a, **k: None
    lib = task._motion_lib['doorOpen']
    source_rows = task._joint_rsi_bank.rows['doorOpen']
    selected = [source_rows[i] for i in (0, len(source_rows)//2, len(source_rows)-1)]
    springs = {
        'original': dict(stiffness=2., damping=.5, max_torque=4.),
        'moderate': dict(stiffness=6., damping=1., max_torque=12., preload_torque=2.),
        'strong_current': dict(task._door_cfg['autoClose']),
    }
    if os.environ.get('OPEN_REFERENCE_SWEEP') == '1':
        # Search is a physics probe, not a change to any training experiment.
        for k, c, preload in ((.5, .25, .5), (.5, .25, 1.), (.5, .5, 1.),
                              (1., .25, .5), (1., .5, .5), (1., .5, 1.),
                              (2., .5, .5), (2., .5, 1.), (3., 1., 1.)):
            name = f'candidate_k{k:g}_c{c:g}_p{preload:g}'
            springs[name] = dict(stiffness=k, damping=c, max_torque=6., preload_torque=preload)
    hand_ids = set(task._key_body_ids[:2].tolist())
    door_ids = {2*task.num_bodies + task._door_leaf_index,
                2*task.num_bodies + task._door_handle_index}
    body_names = task.gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles_all[0][0])
    reports, traces, force_reports = [], [], []

    def pairs():
        hands, others = set(), set()
        for contact in task.gym.get_env_rigid_contacts(task.envs[0]):
            b0, b1 = int(contact['body0']), int(contact['body1'])
            human = b0 if b1 in door_ids else b1 if b0 in door_ids else -1
            if 0 <= human < task.num_bodies:
                (hands if human in hand_ids else others).add(body_names[human])
        return hands, others

    def reset(source=None, angle=0.):
        set_seed(42)
        task._reset_envs(ids)
        if source is not None:
            row = dict(source, angle=angle)
            sample = {k: torch.tensor([v, v], device=task.device) for k, v in row.items()}
            task._apply_joint_rsi(ids, owners, 'doorOpen', sample)
            task._humanoid_root_states[:, 1, 0] -= 10.
        else:
            task._humanoid_root_states[..., 0] -= 10.
            task._humanoid_root_states[..., 7:] = 0.
            task._door_dof_state[:, :, 0] = angle
            task._door_dof_state[:, :, 1] = 0.
        task._reset_env_tensors(ids)
        task._sync_door_reset_states(ids)
        task._refresh_sim_tensors()
        task.relation_runtime.reset(ids, task._sample_tasks(ids), task._box_states[:, 2:])
        task._reset_task(ids)
        task.relation_runtime.initialize_manipulation(ids, task._box_states[:, 2:],
            task._tar_pos, task.door_hinge_angle)
        task._compute_observations(ids)
        task._init_amp_obs(ids)

    try:
        for name, spring in springs.items():
            task._door_cfg['autoClose'] = spring
            for source in selected:
                mid = int(source['motion_id'])
                length = float(lib._motion_lengths[mid])
                for start_kind, angle in [('closed', 0.), ('rsi_angle', source['angle'])]:
                    reset(source, angle)
                    initial_root = task._humanoid_root_states[0, 0, :3].clone()
                    max_angle = clean_max = angle
                    first_body = None
                    hand_steps = body_steps = clean_steps = 0
                    max_hand_force = 0.
                    clean_upright_90 = False
                    contacted_bodies = set()
                    errors = []
                    # Replay the remaining clip, then hold its final joint pose 1s.
                    steps = math.ceil((length-source['time']+1.)/task.dt)
                    trial = f'{name}_{source["time"]:.3f}_{start_kind}'
                    for step in range(steps):
                        time = min(length, source['time'] + (step+1)*task.dt)
                        state = lib.get_motion_state(torch.tensor([mid, mid]), torch.tensor([time, time]))
                        task._every_env_init_dof_pos[:, 0] = state[2]
                        task.pre_physics_step(actions)
                        task._physics_step()
                        task.post_physics_step()
                        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._dof_pos).all()
                        hands, body = pairs()
                        degrees = math.degrees(float(task.door_hinge_angle[0, 0]))
                        max_angle = max(max_angle, math.radians(degrees))
                        if body and first_body is None:
                            first_body = (step+1)*task.dt
                        if first_body is None:
                            clean_max = max(clean_max, math.radians(degrees))
                            from utils.door_hold import current_posture
                            clean_upright_90 |= degrees >= 90. and bool(current_posture(task)[0, 0])
                        hand_steps += bool(hands)
                        body_steps += bool(body)
                        clean_steps += bool(hands) and not body
                        contacted_bodies.update(body)
                        force = float(task._contact_forces[0, 0, task._key_body_ids[:2]].norm(dim=-1).max())
                        max_hand_force = max(max_hand_force, force)
                        error = float((task._dof_pos[0, 0]-state[2][0]).square().mean().sqrt())
                        errors.append(error)
                        traces.append(dict(trial=trial, seconds=(step+1)*task.dt, reference_time=time,
                            angle_deg=degrees, hand_pair=int(bool(hands)), body_pair=int(bool(body)),
                            hand_force_N=force, root_z=float(task._humanoid_root_states[0, 0, 2]),
                            joint_error_rms_rad=error, nonhand_bodies='|'.join(sorted(body))))
                        # The unrelated agent is parked away and can fall;
                        # only the tested owner's termination ends this trial.
                        if task._terminate_buf[0]:
                            break
                    reports.append(dict(trial=trial, spring=name, source=source, start_kind=start_kind,
                        initial_angle_deg=math.degrees(angle), steps=step+1,
                        max_angle_deg=math.degrees(max_angle), final_angle_deg=degrees,
                        max_angle_before_first_nonhand_contact_deg=math.degrees(clean_max),
                        first_nonhand_contact_seconds=first_body, hand_pair_steps=hand_steps,
                        nonhand_pair_steps=body_steps, hand_without_nonhand_pair_steps=clean_steps,
                        nonhand_bodies=sorted(contacted_bodies), peak_hand_net_force_N=max_hand_force,
                        reached_90_upright_before_nonhand_contact=clean_upright_90,
                        mean_joint_error_rms_rad=sum(errors)/len(errors),
                        final_root_height_m=float(task._humanoid_root_states[0, 0, 2]),
                        root_displacement_m=float((task._humanoid_root_states[0, 0, :3]-initial_root).norm()),
                        terminated=bool(task._terminate_buf[0])))
                    print('REFERENCE_PD', json.dumps(reports[-1]), flush=True)
            # Isolated mechanics: a tangential force follows the handle; people
            # are moved away, and env1 receives no external force as a control.
            for force_N in (20., 40., 60.):
                reset()
                nbody = task._rigid_body_state.view(2, -1, 13).shape[1]
                forces = torch.zeros(2, nbody, 3)
                positions = torch.zeros_like(forces)
                maximum = torch.zeros(2)
                for step in range(120):
                    task.pre_physics_step(actions)
                    for _ in range(task.control_freq_inv):
                        a = task.door_hinge_angle[0, 0]
                        forces[0, 2*task.num_bodies+task._door_leaf_index] = torch.tensor([-torch.sin(a), torch.cos(a), 0.])*force_N
                        positions[0, 2*task.num_bodies+task._door_leaf_index] = task.door_handle_position[0, 0]
                        task._closing_forces[:, -2:] = closing_torque(task.door_hinge_angle, task.door_hinge_velocity, spring)
                        task.gym.set_dof_actuation_force_tensor(task.sim, gymtorch.unwrap_tensor(task._closing_forces))
                        task.gym.apply_rigid_body_force_at_pos_tensors(task.sim,
                            gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(positions), gymapi.ENV_SPACE)
                        task.gym.simulate(task.sim)
                        task.gym.fetch_results(task.sim, True)
                        task._refresh_sim_tensors()
                        maximum = torch.maximum(maximum, task.door_hinge_angle[:, 0])
                force_reports.append(dict(spring=name, force_N=force_N,
                    max_angle_deg=(maximum*180/math.pi).tolist(),
                    final_angle_deg=(task.door_hinge_angle[:, 0]*180/math.pi).tolist()))
                print('HANDLE_FORCE', json.dumps(force_reports[-1]), flush=True)
        result = dict(training=False, device=task.device, springs=springs,
            reference_pd=reports, handle_force=force_reports,
            hand_body_indices=sorted(hand_ids), door_body_indices=sorted(door_ids),
            body_names=body_names, motion_duration_seconds=float(lib._motion_lengths[0]),
            notes=['Reference trials use original joint targets and free floating roots; no pose teleportation or external assistance after reset.',
                   'PD tracking failure is not proof that a learned policy cannot push.',
                   'CPU contacts identify actual human/door pairs at control boundaries; short contacts between boundaries can be missed.',
                   'Before-first-body angles conservatively exclude all subsequent progression, even after body contact ends.',
                   'Net hand force includes all contacts; it is not a pair-specific tangential force.',
                   'The clip end pose is held for 1s, not a learned 5s support policy.',
                   'Handle-force trials impose an ideal tangential force, not a humanoid strength measurement. Env1 is the zero-force control.'])
        (out/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
        with (out/'reference_traces.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(traces[0]))
            writer.writeheader()
            writer.writerows(traces)
        write_report(out, result, traces)
    finally:
        task.gym.destroy_sim(task.sim)


def write_report(out, result, traces):
    """Keep the interpretation and a reproducible standalone plot with the audit."""
    lines = ['# OPEN reference push audit', '',
        'CPU PhysX, 2 environments; evaluated owner is env0/agent0. No training or checkpoint.', '',
        'The reference joint targets are replayed with the normal asset PD gains. The root is free after RSI alignment. '
        'Starts use three screened reference times and either a closed door or its RSI proxy angle. '
        'Other people are parked away. A trial ends when the tested owner falls.', '',
        '## Reference PD', '',
        '| Spring | Reference time (s) | Start | Initial (deg) | Max (deg) | Max before first non-hand contact (deg) | First non-hand contact (s) | Fell |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |']
    for r in result['reference_pd']:
        first = '-' if r['first_nonhand_contact_seconds'] is None else f'{r["first_nonhand_contact_seconds"]:.2f}'
        lines.append(f'| {r["spring"]} | {r["source"]["time"]:.3f} | {r["start_kind"]} | '
            f'{r["initial_angle_deg"]:.1f} | {r["max_angle_deg"]:.1f} | '
            f'{r["max_angle_before_first_nonhand_contact_deg"]:.1f} | {first} | {r["terminated"]} |')
    lines += ['', '## Ideal handle-force control', '',
        'Force follows the handle tangent continuously for 4s. Humans are kept away. '
        'This tests mechanical openability; it does not measure humanoid hand strength.', '',
        '| Spring | Applied force (N) | Max (deg) | Final (deg) | Unforced control max (deg) |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in result['handle_force']:
        lines.append(f'| {r["spring"]} | {r["force_N"]:.0f} | {r["max_angle_deg"][0]:.1f} | '
            f'{r["final_angle_deg"][0]:.1f} | {r["max_angle_deg"][1]:.6f} |')
    lines += ['', '## Limits', ''] + [f'- {n}' for n in result['notes']]
    lines += ['', 'All three spring settings must be compared. Failure with the original resistance '
        'means balancing/reference alignment is also a limitation; lowering resistance alone is not validated as a solution.', '',
        'Files: `audit.json`, `reference_traces.csv`, `comparison.png`.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    colors = dict(original='#2a9d8f', moderate='#e9a52b', strong_current='#d1495b')
    if 'candidate_k1_c0.5_p1' in result['springs']:
        colors['candidate_k1_c0.5_p1'] = '#3366cc'
    times = sorted({r['source']['time'] for r in result['reference_pd']})
    for i, kind in enumerate(('closed', 'rsi_angle')):
        for j, time in enumerate(times):
            ax = axes[i, j]
            for spring, color in colors.items():
                report = next(r for r in result['reference_pd'] if r['spring']==spring and
                    r['start_kind']==kind and r['source']['time']==time)
                rows = [r for r in traces if r['trial']==report['trial']]
                ax.plot([r['seconds'] for r in rows], [r['angle_deg'] for r in rows], color=color, label=spring)
                body = next((r for r in rows if r['body_pair']), None)
                if body is not None:
                    ax.scatter(body['seconds'], body['angle_deg'], marker='x', color=color, s=55)
            ax.axhline(90, linestyle='--', color='gray', linewidth=1)
            ax.set_title(f'{kind}; reference start {time:.2f}s')
            ax.set_xlabel('Replay seconds (ends on fall)')
            ax.grid(alpha=.2)
        axes[i, 0].set_ylabel('Door angle (degrees)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Original AMP joint PD, free root: x = first non-hand door contact')
    fig.tight_layout()
    fig.savefig(out/'comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()

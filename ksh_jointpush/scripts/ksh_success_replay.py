"""Replay recorded successful episodes, without policy or physical resimulation."""
from pathlib import Path
import argparse
import json
import time
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import isaacgym
from isaacgym import gymapi, gymtorch
import numpy as np
import torch
from ksh_jointpush.ksh_success_cases import first_hits


def load_cases(directory):
    report = json.loads((directory / 'cases.json').read_text())
    cases = []
    for metadata in report['cases']:
        file = Path(metadata['file'])
        if not file.is_absolute():
            file = directory / file.name
        metadata = dict(metadata, file=str(file))
        with np.load(file, allow_pickle=False) as source:
            data = {key: source[key].copy() for key in source.files}
        hits = first_hits(data['success'])
        if not (hits >= 0).any() or hits.tolist() != metadata['first_hit_frames']:
            raise ValueError('Case has no verified later arrival')
        metadata['ever_agents'] = np.flatnonzero(hits >= 0).tolist()
        metadata['final_agents'] = np.flatnonzero(data['success'][-1]).tolist()
        for key in ('roots', 'dofs', 'bodies', 'targets', 'box_sizes'):
            if not np.isfinite(data[key]).all():
                raise ValueError('Nonfinite recorded state: ' + key)
        cases.append((metadata, data))
    if not cases:
        raise ValueError('No success cases')
    cases.sort(key=lambda case: (len(case[0].get('final_agents', [])),
        len(case[0].get('ever_agents', []))), reverse=True)
    return cases


def goal_lines(gym, viewer, env, targets, flags, origin):
    angles = np.linspace(0, 2 * np.pi, 41)
    for target, achieved in zip(targets - origin, flags):
        ring = np.column_stack((target[0] + .2 * np.cos(angles),
            target[1] + .2 * np.sin(angles), np.full_like(angles, .025)))
        sphere = []
        for a, b in ((0, 1), (0, 2), (1, 2)):
            circle = np.tile(target, (len(angles), 1))
            circle[:, a] += .12 * np.cos(angles)
            circle[:, b] += .12 * np.sin(angles)
            sphere.append(np.stack((circle[:-1], circle[1:]), 1))
        segments = np.concatenate((np.stack((ring[:-1], ring[1:]), 1),
            np.array([[[target[0], target[1], .025], target]]), *sphere)).astype(np.float32)
        color = (0., 1., .2) if achieved else (1., .8, 0.)
        gym.add_lines(viewer, env, len(segments), segments,
            np.tile(color, (len(segments), 1)).astype(np.float32))


def replay(directory, gpu=0, verify_frames=0):
    cases = load_cases(directory)
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1e-6
    params.substeps = 1
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0., 0., 0.)
    params.use_gpu_pipeline = False
    params.physx.use_gpu = False
    sim = gym.create_sim(gpu, gpu, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError('Could not create replay simulator')
    env = gym.create_env(sim, gymapi.Vec3(-30., -30., 0.), gymapi.Vec3(30., 30., 30.), 1)
    asset_path = Path(cases[0][0]['humanoid_asset'])
    options = gymapi.AssetOptions()
    options.angular_damping = .01
    options.max_angular_velocity = 100.
    options.disable_gravity = True
    options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    humanoid = gym.load_asset(sim, str(asset_path.parent), asset_path.name, options)
    if humanoid is None:
        raise RuntimeError('Could not load original humanoid asset')
    nb, nd = gym.get_asset_rigid_body_count(humanoid), gym.get_asset_dof_count(humanoid)
    palette = [(0.3, .6, 1.), (1., .4, .2), (.8, .3, .9), (.2, .9, .6)]
    blocks = []
    box_cache = {}
    for index, (metadata, data) in enumerate(cases):
        if data['dofs'].shape[1:3] != (4, nd) or data['bodies'].shape[1:3] != (4, nb):
            raise ValueError('Recorded humanoid geometry differs')
        start = gym.get_sim_actor_count(sim)
        for agent in range(4):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(1000. + 30. * index, 5. * agent, 10.)
            handle = gym.create_actor(env, humanoid, pose, f'case{index}_agent{agent}', 0, 0xFFFF)
            color = (0., 1., .2) if agent in metadata['ever_agents'] else palette[agent]
            for body in range(nb):
                gym.set_rigid_body_color(env, handle, body, gymapi.MESH_VISUAL, gymapi.Vec3(*color))
            props = gym.get_actor_dof_properties(env, handle)
            props['driveMode'] = gymapi.DOF_MODE_NONE
            props['stiffness'] = props['damping'] = 0.
            gym.set_actor_dof_properties(env, handle, props)
        for slot, size in enumerate(data['box_sizes']):
            key = tuple(float(x) for x in size)
            if key not in box_cache:
                box_cache[key] = gym.create_box(sim, *key, options)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(1000. + 30. * index, 30. + slot, 10.)
            handle = gym.create_actor(env, box_cache[key], pose, f'case{index}_box{slot}', 0, 0xFFFF)
            color = (.8, .65, .25) if slot in data['assignments'] else (.4, .4, .4)
            gym.set_rigid_body_color(env, handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color))
        blocks.append(start)
    floor_options = gymapi.AssetOptions()
    floor_options.fix_base_link = True
    floor = gym.create_box(sim, 80., 80., .05, floor_options)
    pose = gymapi.Transform()
    pose.p.z = -.025
    floor_handle = gym.create_actor(env, floor, pose, 'visual_floor', 0, 0xFFFF)
    gym.set_rigid_body_color(env, floor_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(.55, .55, .55))
    gym.prepare_sim(sim)
    roots = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dofs = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
    bodies = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
    gym.refresh_actor_root_state_tensor(sim)
    parked = roots.clone()
    parked[:, 7:] = 0.
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError('Could not create replay viewer')
    for key, name in [(gymapi.KEY_N, 'next'), (gymapi.KEY_P, 'previous'),
                      (gymapi.KEY_R, 'restart'), (gymapi.KEY_SPACE, 'pause'),
                      (gymapi.KEY_G, 'arrival'),
                      (gymapi.KEY_ESCAPE, 'quit')]:
        gym.subscribe_viewer_keyboard_event(viewer, key, name)
    output = directory / 'replay'
    output.mkdir(exist_ok=True)
    selected, frame, paused, changed, rendered = 0, 0, False, True, 0
    verified = set()
    print('KSH_SUCCESS_REPLAY_READY', len(cases), 'N=next P=previous R=restart G=arrival Space=pause Esc=quit; green humans=ever reached, green boxes/goals=current success', flush=True)
    try:
        while not gym.query_viewer_has_closed(viewer):
            started = time.monotonic()
            for event in gym.query_viewer_action_events(viewer):
                if event.value <= 0:
                    continue
                if event.action == 'quit':
                    return
                if event.action in ('next', 'previous'):
                    selected = (selected + (1 if event.action == 'next' else -1)) % len(cases)
                    frame, changed = 0, True
                elif event.action == 'restart':
                    frame = 0
                elif event.action == 'pause':
                    paused = not paused
                elif event.action == 'arrival':
                    frame = min(x for x in cases[selected][0]['first_hit_frames'] if x >= 0)
            metadata, data = cases[selected]
            origin = data['origin']
            if changed:
                agent = min(metadata['ever_agents'], key=lambda a: metadata['first_hit_frames'][a])
                members = np.flatnonzero(data['groups'] == data['groups'][agent])
                hit = metadata['first_hit_frames'][agent]
                box = 4 + int(data['assignments'][agent])
                points = np.concatenate((data['roots'][:hit + 1, members, :3].reshape(-1, 3),
                    data['roots'][:hit + 1, box, :3], data['targets'][agent:agent + 1])) - origin
                low, high = points.min(0), points.max(0)
                target = (low + high) / 2
                target[2] = .8
                distance = max(7., float(np.linalg.norm(high[:2] - low[:2])) * 1.1)
                camera = target + np.array([distance * .5, -distance, distance * .65])
                gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(*camera), gymapi.Vec3(*target))
                print('KSH_SUCCESS_REPLAY_CASE', selected + 1, json.dumps(metadata), flush=True)
                changed = False
            roots.copy_(parked)
            start = blocks[selected]
            pose = torch.from_numpy(data['roots'][frame, :28].copy())
            pose[:, :3] -= torch.from_numpy(origin)
            pose[:, 7:] = 0.
            roots[start:start + 28] = pose
            dofs.zero_()
            begin = selected * 4 * nd
            dofs[begin:begin + 4 * nd, 0] = torch.from_numpy(data['dofs'][frame, ..., 0].reshape(-1).copy())
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(roots))
            gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dofs))
            # One microsecond materializes transforms; recorded poses, not
            # policy actions or new task physics, drive the displayed motion.
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_rigid_body_state_tensor(sim)
            gym.clear_lines(viewer)
            goal_lines(gym, viewer, env, data['targets'], data['success'][frame], origin)
            for slot in np.unique(data['assignments']):
                achieved = data['success'][frame, data['assignments'] == slot].any()
                color = (0., 1., .2) if achieved else (.8, .65, .25)
                gym.set_rigid_body_color(env, blocks[selected] + 4 + int(slot), 0,
                    gymapi.MESH_VISUAL, gymapi.Vec3(*color))
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            rendered += 1
            hit = min(x for x in metadata['first_hit_frames'] if x >= 0)
            if frame == hit and selected not in verified:
                body_start = selected * (4 * nb + 24)
                actual = bodies[body_start:body_start + 4 * nb, :3].view(4, nb, 3).numpy()
                error = float(np.max(np.abs(actual - (data['bodies'][frame] - origin))))
                if error > .01:
                    raise RuntimeError('Replay body pose differs from recording: ' + str(error))
                image = output / ('case_' + str(selected + 1).zfill(3) + '_arrival.png')
                gym.write_viewer_image_to_file(viewer, str(image))
                (output / ('case_' + str(selected + 1).zfill(3) + '_verification.json')).write_text(
                    json.dumps(dict(frame=frame, body_max_abs_error=error,
                        official_success=data['success'][frame].tolist(),
                        image=str(image), policy_steps_during_replay=0), indent=2) + '\n')
                verified.add(selected)
                print('KSH_SUCCESS_REPLAY_ARRIVAL_VERIFIED', selected + 1, frame, error, flush=True)
            if verify_frames:
                if rendered >= verify_frames:
                    return
                frame = hit if rendered == 1 else (frame + 1) % len(data['roots'])
            elif not paused:
                frame = (frame + 1) % len(data['roots'])
            time.sleep(max(0., metadata['dt'] - (time.monotonic() - started)))
    finally:
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--verify-frames', type=int, default=0)
    args = parser.parse_args()
    replay(args.directory.resolve(), args.gpu, args.verify_frames)

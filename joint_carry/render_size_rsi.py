import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tokenhsi'))
from isaacgym import gymapi, gymtorch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import yaml

from joint_carry.artifacts import contract_digest
from joint_carry.size_candidates import make_candidates
from tokenhsi.utils.grasp_geometry import ANCHOR_INSET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=27)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--source', choices=('single', 'stage2'), default='single')
    parser.add_argument('--inset', type=float, default=ANCHOR_INSET)
    parser.add_argument('--directory', default='joint_carry/stage2_size_rsi/inset15_single')
    parser.add_argument('--sizes', nargs='+')
    args = parser.parse_args()
    output = Path(args.output)
    if Path('output_etc/check').resolve() not in output.resolve().parents:
        raise ValueError('RSI previews belong under output_etc/check')
    output.mkdir(parents=True, exist_ok=True)
    sizes = list(itertools.product((.2, .4, .6), (.4, .8, 1.2), (.2, .4, .6)))[:args.limit]
    if args.sizes:
        sizes = [tuple(float(value) for value in size.split(',')) for size in args.sizes]
    labels = ['source'] + ['_'.join(str(round(v * 20)) for v in size) for size in sizes]
    physical_sizes = [[.52, .8, .4]] + [list(size) for size in sizes]
    candidates, accepted = {}, {}
    for skill in ('pickUp', 'carryWith', 'putDown'):
        with np.load(Path('joint_carry/source_rsi') / skill / 'cooperative_rsi_candidate.npz', allow_pickle=False) as archive:
            candidates[skill] = [dict(archive)]
        accepted[skill] = [None]
        for label, size in zip(labels[1:], sizes):
            data, _ = make_candidates(skill, size, args.source, args.inset)
            path = Path(args.directory) / label / (skill + '.npz')
            metadata = json.loads(path.with_suffix('.json').read_text())
            if metadata['contract_digest'] != contract_digest(args.source, args.inset):
                raise ValueError('Stale screened RSI: ' + str(path))
            with np.load(path, allow_pickle=False) as archive:
                selected = archive['source_frame']
                if not np.allclose(data['dof_position'][selected], archive['dof_position'], atol=1e-5):
                    raise ValueError('Preview differs from screened joint states')
                if not np.allclose(data['body_position'][selected], archive['body_state'][..., :3], atol=1e-5):
                    raise ValueError('Preview differs from screened body positions')
            mask = np.zeros(len(data['time']), dtype=bool)
            mask[selected] = True
            candidates[skill].append(data)
            accepted[skill].append(mask)
        print('Prepared', skill, len(sizes), 'corrected sizes', flush=True)

    cfg = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_v2_fixed.yaml').read_text())
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.up_axis = gymapi.UP_AXIS_Z
    params.dt = 1 / 60
    params.gravity = gymapi.Vec3(0, 0, 0)
    params.use_gpu_pipeline = False
    params.physx.use_gpu = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError('Could not create native RSI preview renderer')
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)
    gym.set_light_parameters(sim, 0, gymapi.Vec3(.8, .8, .8), gymapi.Vec3(.45, .45, .45), gymapi.Vec3(-.4, -.5, 1))
    options = gymapi.AssetOptions()
    options.disable_gravity = True
    asset = cfg['env']['asset']
    human = gym.load_asset(sim, asset['assetRoot'], asset['assetFileName'], options)
    environments, cameras, origins = [], [], []
    width, height = 640, 480
    for index, size in enumerate(physical_sizes):
        env = gym.create_env(sim, gymapi.Vec3(-4, -4, 0), gymapi.Vec3(4, 4, 4), 6)
        environments.append(env)
        origin = gym.get_env_origin(env)
        origins.append([origin.x, origin.y, origin.z])
        for person, color in enumerate(((.20, .58, .95), (.95, .36, .20))):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(person * 3, 0, 1)
            actor = gym.create_actor(env, human, pose, 'person' + str(person), index, 0)
            for body in range(gym.get_asset_rigid_body_count(human)):
                gym.set_rigid_body_color(env, actor, body, gymapi.MESH_VISUAL, gymapi.Vec3(*color))
        box_asset = gym.create_box(sim, *size, options)
        actor = gym.create_actor(env, box_asset, gymapi.Transform(), 'payload', index, 0)
        gym.set_rigid_body_color(env, actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(.85, .69, .30))
        props = gymapi.CameraProperties()
        props.width, props.height = width, height
        props.horizontal_fov = 48.
        cameras.append([gym.create_camera_sensor(env, props) for _ in range(2)])
    gym.prepare_sim(sim)
    root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).reshape(len(labels), 3, 13)
    dof = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).reshape(len(labels), 2, 32, 2)
    origins = np.asarray(origins)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    writers, logs = [], []
    for label in labels:
        log = (output / (label + '.ffmpeg.log')).open('w')
        logs.append(log)
        writers.append(subprocess.Popen([
            'ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', '1280x480', '-r', '15', '-i', '-', '-an', '-c:v', 'libx264',
            '-threads', '1', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(output / (label + '.mp4')),
        ], stdin=subprocess.PIPE, stderr=log))
    total_frames = 0
    for skill, batch in candidates.items():
        count = len(batch[0]['time'])
        if args.frames:
            count = min(count, args.frames)
        for frame in range(0, count, 2):
            states = []
            for data in batch:
                people = np.concatenate((data['root_position'][frame], data['root_rotation_xyzw'][frame], np.zeros((2, 6))), -1)
                box = np.concatenate((data['box_position'][frame], data['box_rotation_xyzw'][frame], np.zeros(6)))
                states.append(np.concatenate((people, box[None]), 0))
            states = np.asarray(states, dtype=np.float32)
            states[..., :3] += origins[:, None]
            root[:] = torch.from_numpy(states)
            dof[..., 0] = torch.tensor(np.stack([data['dof_position'][frame] for data in batch]))
            dof[..., 1] = 0.
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root))
            gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof))
            for index, (env, cams) in enumerate(zip(environments, cameras)):
                center = states[index, 2, :3].copy()
                target = center.copy()
                target[2] = origins[index, 2] + .85
                for camera, offset in zip(cams, ((3., -1.9, 1.7), (0., -.2, 4.))):
                    position = target + np.asarray(offset)
                    gym.set_camera_location(camera, env, gymapi.Vec3(*position), gymapi.Vec3(*target))
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            for index, (env, cams, writer) in enumerate(zip(environments, cameras, writers)):
                panels = [gym.get_camera_image(sim, env, camera, gymapi.IMAGE_COLOR).reshape(height, width, 4)[..., :3] for camera in cams]
                image = Image.fromarray(np.concatenate(panels, 1))
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 1280, 68), fill=(15, 23, 34))
                size_text = ' x '.join('{:.2f}'.format(v) for v in physical_sizes[index])
                title = 'OLD STAGE2 BASELINE' if index == 0 else '{:g}cm inset'.format(args.inset*100)
                draw.text((12, 7), '{} | {} | {} m'.format(title, skill, size_text), font=font, fill='white')
                state = 'REFERENCE ONLY' if index == 0 else 'ELIGIBLE RESET FRAME' if accepted[skill][index][frame] else 'EXCLUDED FROM RESET POOL'
                color = '#b8c5d6' if index == 0 else '#72e6a2' if accepted[skill][index][frame] else '#ff8c80'
                draw.text((12, 34), 'frame {:03d}/{} | {}'.format(frame, len(batch[index]['time']) - 1, state), font=font, fill=color)
                draw.text((650, 7), 'TOP VIEW | kinematic replay', font=font, fill='#b8c5d6')
                writer.stdin.write(np.asarray(image).tobytes())
                if frame == count // 2 // 2 * 2:
                    image.save(output / (labels[index] + '_' + skill + '.jpg'))
            total_frames += 1
        print('Rendered', skill, count, 'source frames', flush=True)
    for writer, log in zip(writers, logs):
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError('Video encoder failed')
        log.close()
    record = dict(kind='native_asset_kinematic_replay', policy_rollout=False,
                  physics_simulated_during_replay=False, output_fps=15,
                  output_frames=total_frames, contract_digest=contract_digest(args.source, args.inset),
                  source_mode=args.source, anchor_inset=args.inset,
                  corrected_sizes=[list(size) for size in sizes],
                  displayed_frames='all candidates; eligibility badge uses screened source_frame indices',
                  source_video='source.mp4', corrected_videos=[label + '.mp4' for label in labels[1:]])
    (output / 'render.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Saved native RSI previews:', output, flush=True)


if __name__ == '__main__':
    main()

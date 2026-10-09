from isaacgym import gymapi, gymtorch
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

from carry_joint_preview import DOF_BODIES, DOF_SIZES, forward_kinematics
from retarget_teamhoi_reference import ASSET, ROOT, SOURCE, select_clips, foot_vertices, lowest_vertex

sys.path.insert(0, str(ROOT/'tokenhsi'))
from utils.motion_lib import MotionLib
from utils.unified_training import append_family
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry, build_amp_observations

parser = argparse.ArgumentParser()
parser.add_argument('--all', action='store_true')
parser.add_argument('--posture-first', action='store_true')
args = parser.parse_args()
OUTPUT, CLIPS = select_clips(args.all, args.posture_first)
OUT = ROOT/('output_etc/teamhoi_retarget_all_check' if args.all else 'output_etc/teamhoi_retarget_pilot_check')
if args.posture_first:
    OUT = ROOT/('output/teamhoi_retarget_posture_all_check' if args.all else 'output_etc/teamhoi_retarget_posture_pilot_check')
count_clips = len(CLIPS)
OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 20, 23, 26, 29, 32]
KEYS = [5, 8, 11, 14]
OUT.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/approach_stage2_joint_carry_mixed80_task_embedding.yaml').read_text())
baseline = yaml.safe_load((ROOT/'tokenhsi/data/dataset_loco_sit_carry_climb.yaml').read_text())['motions']['loco'][0]['file']
paths = [SOURCE/p for p in CLIPS.values()]+[OUTPUT/(kind+'.npy') for kind in CLIPS]+[ROOT/'tokenhsi/data'/baseline]
labels = ['raw_'+kind for kind in CLIPS]+['retarget_'+kind for kind in CLIPS]+['existing_loco']
wrapper = OUT/'loader.yaml'
wrapper.write_text(yaml.safe_dump({'motions':{'loco':[{'file':str(p), 'weight':1.} for p in paths]}}))
lib = MotionLib(str(wrapper), 'loco', DOF_BODIES, OFFSETS, KEYS, 'cpu')
raw0 = np.load(paths[0], allow_pickle=True).item()
names = raw0['skeleton_tree']['node_names']
parents = raw0['skeleton_tree']['parent_indices']['arr']
bodies = {b.get('name'):b for b in ET.parse(ASSET).getroot().findall('.//worldbody//body')}
offsets = np.array([np.fromstring(bodies[n].get('pos', '0 0 0'), sep=' ') for n in names])
bounds = np.deg2rad([np.fromstring(j.get('range'), sep=' ') for n in names[1:] for j in bodies[n].findall('joint')])
ctx = SimpleNamespace(dt=1/30, _num_amp_obs_steps=10, device='cpu', _local_root_obs=cfg['env']['localRootObs'],
                      _root_height_obs=cfg['env']['rootHeightObs'], _dof_obs_size=72, _dof_offsets=OFFSETS)
report = {'clips':[], 'training_connected':False, 'scope':'Geometry, actual AMP history construction, physical reset and open-loop PD; no policy training'}
states, poses, velocities, indices = [], [], [], []


def decode_motion(state):
    pos, rot, dof, vel, ang, dof_vel, key = state
    local = np.zeros((len(pos), 15, 4));local[:, :, 3] = 1;local[:, 0] = rot.numpy()
    for j, body in enumerate(DOF_BODIES):
        rv = dof[:, OFFSETS[j]:OFFSETS[j+1]].numpy()
        if DOF_SIZES[j] == 1:
            rv = np.pad(rv, ((0, 0), (1, 1)))
        local[:, body] = R.from_rotvec(rv).as_quat()
    return forward_kinematics(local, pos.numpy(), parents, offsets)


for i, path in enumerate(paths):
    raw = np.load(path, allow_pickle=True).item()
    frames = len(raw['root_translation']['arr'])
    times = torch.arange(frames)/raw['fps']
    state = lib.get_motion_state(torch.full((frames,), i, dtype=torch.long), times)
    pos, rot, dof, vel, ang, dof_vel, key = state
    physical, world_q = decode_motion(state)
    half_state = lib.get_motion_state(torch.full((frames-1,), i, dtype=torch.long), times[:-1]+.5/raw['fps'])
    half_pos, half_q = decode_motion(half_state)
    foot_floors = np.stack([physical[:, j, 2]+lowest_vertex(world_q[:, j], foot_vertices(names[j])) for j in (11, 14)], 1)
    half_floors = np.stack([half_pos[:, j, 2]+lowest_vertex(half_q[:, j], foot_vertices(names[j])) for j in (11, 14)], 1)
    foot_speed = np.linalg.norm(np.gradient(physical[:, [11, 14], :2], axis=0)*raw['fps'], axis=-1)
    near_ground = foot_floors < .035
    material_speeds = []
    for j in (11, 14):
        mesh = foot_vertices(names[j])
        points = np.einsum('fij,vj->fvi', R.from_quat(world_q[:, j]).as_matrix(), mesh)+physical[:, j, None]
        material_v = np.gradient(points, axis=0)*raw['fps']
        near = (points[:, :, 2] < .035) & (points[:, :, 2] < points[:, :, 2].min(1, keepdims=True)+.005)
        material_speeds.extend(np.linalg.norm(material_v[:, :, :2], axis=-1)[near].tolist())
    check_times = torch.linspace(.3, float(lib._motion_lengths[i]), 128)
    demo = HumanoidMACarry.build_amp_obs_demo(ctx, torch.full((128,), i, dtype=torch.long), check_times, lib)
    demo = append_family(demo, torch.zeros(len(demo), dtype=torch.long)).reshape(128, -1)
    limit_error = np.maximum(bounds[:, 0]-dof.numpy(), dof.numpy()-bounds[:, 1]).clip(0)
    half_limit_error = np.maximum(bounds[:, 0]-half_state[2].numpy(), half_state[2].numpy()-bounds[:, 1]).clip(0)
    key_error = np.linalg.norm(physical[:, KEYS]-key.numpy(), axis=-1)
    velocity_error = np.linalg.norm(vel.numpy()-np.gradient(pos.numpy(), axis=0)*raw['fps'], axis=-1)
    report['clips'].append(dict(label=labels[i], path=str(path.relative_to(ROOT)), frames=frames,
        key_position_max_error_m=float(key_error.max()),
        subframe_key_position_max_error_m=float(np.linalg.norm(half_pos[:, KEYS]-half_state[-1].numpy(), axis=-1).max()),
        floor_min_m=float(min(foot_floors.min(), half_floors.min())),
        support_foot_max_height_m=float(max(foot_floors.min(1).max(), half_floors.min(1).max())),
        joint_limit_max_violation_rad=float(max(limit_error.max(), half_limit_error.max())),
        frame_central_difference_velocity_max_error_mps=float(velocity_error.max()),
        subframe_velocity_derivative_max_error_mps=float(np.linalg.norm(half_state[3].numpy()-np.diff(pos.numpy(), axis=0)*raw['fps'], axis=-1).max()),
        dof_velocity_max_radps=float(dof_vel.abs().max()),
        near_ground_foot_origin_speed_mean_mps=float(foot_speed[near_ground].mean()),
        near_ground_material_speed_mean_mps=float(np.mean(material_speeds)),
        near_ground_material_speed_p95_mps=float(np.quantile(material_speeds, .95)),
        amp_history_shape=list(demo.shape), amp_finite=bool(torch.isfinite(demo).all())))
    states.append(torch.cat((pos, rot, vel, ang), -1));poses.append(dof);velocities.append(dof_vel)
    indices.extend([i]*frames)
(OUT/'geometry.json').write_text(json.dumps(report, indent=2)+'\n')
print('GEOMETRY_DONE', flush=True)

device = 'cuda:0'
lib_gpu = MotionLib(str(wrapper), 'loco', DOF_BODIES, OFFSETS, KEYS, device)
ctx.device = device
mid = torch.arange(len(paths), device=device).repeat_interleave(128)
torch.manual_seed(7)
sample_times = .3+torch.rand(len(mid), device=device)*(lib_gpu._motion_lengths[mid]-.3)
demo = HumanoidMACarry.build_amp_obs_demo(ctx, mid, sample_times, lib_gpu)
demo = append_family(demo, torch.zeros(len(demo), device=device, dtype=torch.long)).reshape(len(mid), -1)
report['gpu_amp'] = dict(shape=list(demo.shape), finite=bool(torch.isfinite(demo).all()), root_height_obs=ctx._root_height_obs)
N = 2048
root_data = torch.cat(states).to(device);dof_data = torch.cat(poses).to(device);vel_data = torch.cat(velocities).to(device)
indices = np.asarray(indices)
gym = gymapi.acquire_gym();params = gymapi.SimParams()
params.dt = 1/60;params.substeps = cfg['sim']['substeps'];params.up_axis = gymapi.UP_AXIS_Z
params.gravity = gymapi.Vec3(0, 0, -9.81);params.use_gpu_pipeline = True
for key, value in cfg['sim']['physx'].items():
    setattr(params.physx, key, value)
params.physx.use_gpu = True
sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
plane = gymapi.PlaneParams();plane.normal = gymapi.Vec3(0, 0, 1)
plane.static_friction = cfg['env']['plane']['staticFriction'];plane.dynamic_friction = cfg['env']['plane']['dynamicFriction']
gym.add_ground(sim, plane)
options = gymapi.AssetOptions();options.angular_damping = .01;options.max_angular_velocity = 100.
options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
asset = gym.load_asset(sim, str(ASSET.parent), ASSET.name, options)
props = gym.get_asset_dof_properties(asset);props['driveMode'] = gymapi.DOF_MODE_POS
if gym.get_asset_dof_count(asset) != 32 or gym.get_asset_rigid_body_names(asset) != names:
    raise RuntimeError('Physical asset layout differs from motion layout')
origins = []
for i in range(N):
    env = gym.create_env(sim, gymapi.Vec3(-3, -3, 0), gymapi.Vec3(3, 3, 3), 46)
    transform = gymapi.Transform();transform.p.z = 1
    actor = gym.create_actor(env, asset, transform, 'human', i, 0)
    gym.set_actor_dof_properties(env, actor, props)
    origin = gym.get_env_origin(env);origins.append([origin.x, origin.y, origin.z])
gym.prepare_sim(sim)
origins = torch.tensor(origins, device=device)
root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
dof = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(N, 32, 2)
body = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)).view(N, 15, 13)
contact = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim)).view(N, 15, 3)


def reset(state, position, velocity):
    root.copy_(state);root[:, :3] += origins
    dof[:, :, 0] = position;dof[:, :, 1] = velocity
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root))
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof))
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(position.contiguous()))


def step():
    gym.simulate(sim);gym.fetch_results(sim, True)
    gym.refresh_actor_root_state_tensor(sim);gym.refresh_dof_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim);gym.refresh_net_contact_force_tensor(sim)


screen = []
sim_fk_errors = []
for start in range(0, len(root_data), N):
    index = torch.arange(start, start+N, device=device).clamp_max(len(root_data)-1)
    state = root_data[index].clone();state[:, :2] = 0
    reset(state, dof_data[index], vel_data[index]);initial = root.clone()
    peak_dv = torch.zeros(N, device=device);peak_disp = torch.zeros_like(peak_dv)
    finite = torch.ones(N, device=device, dtype=torch.bool)
    for _ in range(6):
        step()
        peak_dv = torch.maximum(peak_dv, (root[:, 7:10]-initial[:, 7:10]).norm(dim=-1))
        peak_disp = torch.maximum(peak_disp, (root[:, :3]-initial[:, :3]).norm(dim=-1))
        finite &= torch.isfinite(root).all(-1) & torch.isfinite(dof).all((-1, -2)) & torch.isfinite(body).all((-1, -2))
    good = finite & (peak_dv <= 3) & (peak_disp <= .3)
    observed_root = root.clone().cpu();observed_root[:, :3] -= origins.cpu()
    simulated_fk, _ = decode_motion((observed_root[:, :3], observed_root[:, 3:7],
                                    dof[:, :, 0].cpu(), None, None, None, None))
    observed_body = (body[:, :, :3]-origins[:, None]).cpu().numpy()
    count = min(N, len(root_data)-start)
    sim_fk_errors.append(np.linalg.norm(simulated_fk-observed_body, axis=-1)[:count])
    screen.append(torch.stack((good, peak_dv, peak_disp, finite), -1)[:count].cpu().numpy())
screen = np.concatenate(screen)
sim_fk_errors = np.concatenate(sim_fk_errors)
for i, entry in enumerate(report['clips']):
    values = screen[indices == i]
    entry['snapshot_0_1s'] = dict(passed=int(values[:, 0].sum()), total=len(values),
                                all_finite=bool(values[:, 3].all()), max_root_dv_mps=float(values[:, 1].max()),
                                p95_root_dv_mps=float(np.quantile(values[:, 1], .95)))
    entry['simulator_body_vs_fk_max_error_m'] = float(sim_fk_errors[indices == i].max())
(OUT/'snapshot.json').write_text(json.dumps(report, indent=2)+'\n')
print('SNAPSHOT_DONE', flush=True)

ids = torch.arange(N, device=device)%len(paths)
times0 = torch.full((N,), .3, device=device)
p, q, j, v, a, jv, _ = lib_gpu.get_motion_state(ids, times0)
p[:, :2] = 0
reset(torch.cat((p, q, v, a), -1), j, jv)
fallen = torch.zeros(N, device=device, dtype=torch.bool)
first_fall = torch.full((N,), -1., device=device)
all_finite = True
track = []
for frame in range(240):
    time = torch.minimum(times0+(frame+1)/30, lib_gpu._motion_lengths[ids])
    _, _, targets, _, _, _, _ = lib_gpu.get_motion_state(ids, time)
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(targets.contiguous()))
    step();step()
    all_finite &= bool(torch.isfinite(root).all() & torch.isfinite(dof).all())
    fell = root[:, 2]-origins[:, 2] < .5
    first_fall[~fallen & fell] = (frame+1)/30;fallen |= fell
    track.append((body[:len(paths), :, :3]-origins[:len(paths), None]).cpu().numpy())
    if frame % 60 == 0:
        print('PD', frame, '/', 240, flush=True)
for i, entry in enumerate(report['clips']):
    selected = ids == i
    entry['free_root_pd_8s'] = dict(fallen_fraction=float(fallen[selected].float().mean()),
        first_fall_s=float(first_fall[selected & fallen].median()) if (selected & fallen).any() else None)
report['physics'] = dict(gpu=5, environments=N, engine='GPU PhysX / GPU pipeline', all_finite=all_finite,
                         note='Open-loop PD is not a learned balance controller. Compare with existing_loco; falling is not an AMP rejection criterion.')
manifest = json.loads((OUTPUT/'manifest.json').read_text())
for i, entry in enumerate(report['clips'][count_clips:2*count_clips]):
    source = np.load(paths[i], allow_pickle=True).item()
    converted = np.load(paths[i+count_clips], allow_pickle=True).item()
    record = manifest['clips'][i]
    sample_stride = converted['fps']//source['fps']
    entry['root_xy_max_change_m'] = float(np.abs(source['root_translation']['arr'][:, :2]-converted['root_translation']['arr'][::sample_stride, :2]).max())
    entry['checks'] = dict(
        source_hash_unchanged=hashlib.sha256(paths[i].read_bytes()).hexdigest() == record['source_sha256'],
        output_hash_matches=hashlib.sha256(paths[i+count_clips].read_bytes()).hexdigest() == record['output_sha256'],
        root_xy_preserved=entry['root_xy_max_change_m'] < 1e-6,
        root_orientation_preserved=bool(np.array_equal(source['rotation']['arr'][:, 0], converted['rotation']['arr'][::sample_stride, 0])),
        amp_history=entry['amp_finite'] and report['gpu_amp']['finite'] and entry['amp_history_shape'] == [128, 1320],
        key_geometry=entry['key_position_max_error_m'] < .002 and entry['subframe_key_position_max_error_m'] < .005,
        simulator_geometry=entry['simulator_body_vs_fk_max_error_m'] < .001,
        contact_height=entry['floor_min_m'] >= -.002 and entry['support_foot_max_height_m'] < .02,
        joint_limits=entry['joint_limit_max_violation_rad'] < 1e-4,
        root_velocity=entry['subframe_velocity_derivative_max_error_mps'] < 1e-3,
        snapshot_physics=entry['snapshot_0_1s']['passed'] == entry['frames'] and entry['snapshot_0_1s']['all_finite'])
report['interface_and_short_physics_pass'] = all(all(entry['checks'].values()) for entry in report['clips'][count_clips:2*count_clips])
report['not_certified'] = ['Natural gait quality', 'Zero foot sliding', 'Closed-loop motion tracking', 'AMP learning improvement', 'Joint RSI']
(OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')
manifest['validation_status'] = 'interface_and_short_physics_pass' if report['interface_and_short_physics_pass'] else 'needs_review'
manifest['validation_report'] = str((OUT/'report.json').relative_to(ROOT))
(OUTPUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
np.savez_compressed(OUT/'simulation.npz', screen=screen, clip_indices=indices, track=np.stack(track))
gym.destroy_sim(sim)
print('DONE', OUT/'report.json', flush=True)

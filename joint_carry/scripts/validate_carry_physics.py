"""Audit saved carry candidates with asset geometry and CPU Isaac Gym PhysX.

Runs the repository's 0.1s constant-PD-target RSI shock screen for every frame,
then free-root PD motion tracking. Never teleports actors during a rollout and
never edits input candidates or declares a policy-trained skill impossible.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from isaacgym import gymapi, gymtorch  # Must precede torch.
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tokenhsi'))
from utils.size_rsi import SCREEN, screen_states

CONFIG = Path('tokenhsi/data/cfg/multi_agent/approach_scenario_stage1_unified_size_rsi.yaml')
ASSET = Path('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml')
SKILLS = ('pickUp', 'carryWith', 'putDown')


def geometry(data):
    """Signed distances for actual asset spheres and sampled capsule centre lines.

    Feet use OBJ vertices for floor checks. Capsule centre samples are <=5mm
    apart; intersection at any sample proves overlap of the actual capsule.
    Mesh-vs-box and humanoid self-collision are left to PhysX.
    """
    xml = ET.parse(ASSET).getroot()
    meshes = {m.get('name'): ASSET.parent/m.get('file') for m in xml.findall('asset/mesh')}
    names = data['joint_names'].tolist()
    frames, agents = data['body_position'].shape[:2]
    hand_depth, other_depth, floor_depth = [np.zeros(frames) for _ in range(3)]
    worst_name = np.full(frames, '', dtype='<U60')
    all_clouds = [[] for _ in range(agents)]
    inverse = R.from_quat(data['box_rotation_xyzw']).inv()
    for body in xml.findall('.//worldbody//body'):
        if body.get('name') not in names:
            continue
        j = names.index(body.get('name'))
        for geom in body.findall('geom'):
            typ = geom.get('type', 'capsule')
            if typ == 'mesh':
                points = np.array([[float(v) for v in line.split()[1:4]]
                                   for line in meshes[geom.get('mesh')].read_text().splitlines()
                                   if line.startswith('v ')])
                radius = 0.
            elif geom.get('fromto'):
                ends = np.fromstring(geom.get('fromto'), sep=' ').reshape(2, 3)
                points = np.linspace(*ends, max(2, int(np.ceil(np.linalg.norm(ends[1]-ends[0])/.005))+1))
                radius = float(geom.get('size').split()[0])
            else:
                assert typ == 'sphere', typ
                points = np.fromstring(geom.get('pos', '0 0 0'), sep=' ')[None]
                radius = float(geom.get('size').split()[0])
            for a in range(agents):
                rot = R.from_quat(data['body_rotation_xyzw'][:, a, j])
                world = np.stack([rot.apply(point) for point in points], axis=1) + data['body_position'][:, a, j, None]
                floor_depth = np.maximum(floor_depth, np.maximum(0, radius-world[..., 2].min(1)))
                if typ == 'mesh':
                    continue
                local = np.stack([inverse.apply(world[:, k]-data['box_position']) for k in range(len(points))], axis=1)
                delta = np.abs(local)-data['box_size']/2
                sdf = np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(delta.max(-1), 0)
                depth = np.maximum(0, radius-sdf.min(1))
                if 'hand' in body.get('name'):
                    hand_depth = np.maximum(hand_depth, depth)
                else:
                    bigger = depth > other_depth
                    worst_name[bigger] = body.get('name') + '/' + geom.get('name', '')
                    other_depth = np.maximum(other_depth, depth)
                # Thinner cloud sufficient to establish inter-person penetration.
                all_clouds[a].append((world[:, ::max(1, len(points)//12)], radius))
    peer_depth = np.zeros(frames)
    if agents == 2:
        for x, rx in all_clouds[0]:
            for y, ry in all_clouds[1]:
                distance = np.linalg.norm(x[:, :, None]-y[:, None], axis=-1).min((1, 2))
                peer_depth = np.maximum(peer_depth, np.maximum(0, rx+ry-distance))
    return dict(hand_box_penetration_m=hand_depth, nonhand_box_penetration_m=other_depth,
                floor_penetration_m=floor_depth, interperson_penetration_lower_bound_m=peer_depth,
                worst_nonhand_shape=worst_name)


class Scene:
    def __init__(self, data):
        self.data = data
        self.n = data['root_position'].shape[1]
        cfg = yaml.safe_load(CONFIG.read_text())
        self.gym = gymapi.acquire_gym()
        params = gymapi.SimParams()
        params.dt = 1/60
        params.substeps = cfg['sim']['substeps']
        params.up_axis = gymapi.UP_AXIS_Z
        params.gravity = gymapi.Vec3(0, 0, -9.81)
        params.use_gpu_pipeline = False
        for key, value in cfg['sim']['physx'].items():
            setattr(params.physx, key, value)
        params.physx.use_gpu = False
        params.physx.contact_collection = gymapi.ContactCollection.CC_ALL_SUBSTEPS
        self.sim = self.gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
        assert self.sim is not None
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        plane.static_friction = cfg['env']['plane']['staticFriction']
        plane.dynamic_friction = cfg['env']['plane']['dynamicFriction']
        plane.restitution = cfg['env']['plane']['restitution']
        self.gym.add_ground(self.sim, plane)
        opt = gymapi.AssetOptions()
        opt.angular_damping = .01
        opt.max_angular_velocity = 100.
        opt.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        human = self.gym.load_asset(self.sim, str(ASSET.parent), ASSET.name, opt)
        assert self.gym.get_asset_rigid_body_names(human) == data['joint_names'].tolist()
        assert self.gym.get_asset_dof_count(human) == 32
        self.props = self.gym.get_asset_dof_properties(human)
        self.props['driveMode'] = gymapi.DOF_MODE_POS
        self.env = self.gym.create_env(self.sim, gymapi.Vec3(-8,-8,0), gymapi.Vec3(8,8,8), 1)
        self.humans = []
        for a in range(self.n):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(3*a, 0, 1)
            handle = self.gym.create_actor(self.env, human, pose, 'human'+str(a), 0, 0)
            self.gym.set_actor_dof_properties(self.env, handle, self.props)
            self.gym.enable_actor_dof_force_sensors(self.env, handle)
            self.humans.append(handle)
        opt = gymapi.AssetOptions()
        opt.angular_damping = .01
        opt.linear_damping = .01
        opt.max_angular_velocity = 100.
        opt.density = 100.
        box = self.gym.create_box(self.sim, *data['box_size'].tolist(), opt)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-3, 0, 2)
        self.box = self.gym.create_actor(self.env, box, pose, 'box', 0, 0)
        self.gym.prepare_sim(self.sim)
        self.root = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.dof = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.body = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.net_contact = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        self.box_mass = float(self.gym.get_actor_rigid_body_properties(self.env, self.box)[0].mass)
        self.reset(0)

    def reset(self, f):
        d = self.data
        roots = np.concatenate([d['root_position'][f], d['root_rotation_xyzw'][f],
                                d['body_linear_velocity'][f, :, 0], d['body_angular_velocity'][f, :, 0]], axis=-1)
        box = np.concatenate([d['box_position'][f], d['box_rotation_xyzw'][f],
                              d['box_linear_velocity'][f], d['box_angular_velocity'][f]])
        self.root.copy_(torch.from_numpy(np.concatenate([roots, box[None]]).astype(np.float32)))
        self.dof[:, 0].copy_(torch.from_numpy(d['dof_position'][f].reshape(-1).astype(np.float32)))
        self.dof[:, 1].copy_(torch.from_numpy(d['dof_velocity'][f].reshape(-1).astype(np.float32)))
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof))
        self.targets(d['dof_position'][f])
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        errors = np.linalg.norm(self.body[:self.n*15, :3].numpy().reshape(self.n,15,3)
                               - d['body_position'][f], axis=-1)
        self.reset_fk_error = float(errors.max())
        # Source motion has 3-axis elbow rotations; the physical asset only
        # represents elbow Y. MotionLib's existing 32-DOF conversion discards
        # elbow X/Z, shifting the hand centres. All other origins must match.
        represented = [j for j in range(15) if j not in (5,8)]
        assert errors[:, represented].max() < 2e-5, ('Unexpected reset FK error', errors)

    def targets(self, target):
        self.target = torch.from_numpy(np.asarray(target, dtype=np.float32).reshape(-1).copy())
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.target))

    def step(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def contacts(self):
        contacts = self.gym.get_env_rigid_contacts(self.env)
        touch = np.zeros(self.n, dtype=bool)
        peak_overlap, peak_force = 0., 0.
        for c in contacts:
            b0, b1 = int(c['body0']), int(c['body1'])
            if self.n*15 not in (b0, b1):
                continue
            other = b1 if b0 == self.n*15 else b0
            if 0 <= other < self.n*15:
                peak_overlap = max(peak_overlap, float(c['initialOverlap']))
                peak_force = max(peak_force, float(c['lambda']))
                if other % 15 in (5, 8) and float(c['lambda']) > .1:
                    touch[other//15] = True
        return touch, peak_overlap, peak_force

    def close(self):
        self.gym.destroy_sim(self.sim)


def audit(data, output):
    preview_geom = geometry(data)
    scene = Scene(data)
    count = len(data['time'])
    records = []
    reset_error = 0.
    try:
        actual_positions, actual_rotations, discrepancies = [], [], []
        for f in range(count):
            scene.reset(f)
            state = scene.body[:scene.n*15].numpy().reshape(scene.n,15,13).copy()
            actual_positions.append(state[..., :3]); actual_rotations.append(state[..., 3:7])
            discrepancies.append(scene.reset_fk_error)
        actual = dict(data, body_position=np.asarray(actual_positions),
                      body_rotation_xyzw=np.asarray(actual_rotations))
        geom = geometry(actual)
        np.savez_compressed(output/'simulator_reset_geometry.npz',
                            body_position=actual['body_position'],body_rotation_xyzw=actual['body_rotation_xyzw'],
                            preview_hand_position_error_m=np.asarray(discrepancies))
        for f in range(count):
            scene.reset(f)
            reset_error = max(reset_error, scene.reset_fk_error)
            initial = scene.root.clone()
            metrics = np.zeros(7)
            passed = True
            touches = []
            for _ in range(6):
                scene.step()
                passed &= bool(screen_states(scene.root[:scene.n], scene.root[-1:].expand(scene.n,1,13),
                                              initial[:scene.n], initial[-1:].expand(scene.n,1,13)).all())
                passed &= bool(torch.isfinite(scene.dof).all())
                touch, overlap, force = scene.contacts()
                touches.append(touch.all())
                values = [torch.linalg.norm(scene.root[:scene.n,7:10]-initial[:scene.n,7:10],dim=-1).max().item(),
                          torch.linalg.norm(scene.root[:scene.n,:3]-initial[:scene.n,:3],dim=-1).max().item(),
                          torch.linalg.norm(scene.root[-1,7:10]).item(),
                          torch.linalg.norm(scene.root[-1,:3]-initial[-1,:3]).item(),
                          overlap, force, max(0.,float(initial[-1,2]-scene.root[-1,2]))]
                metrics = np.maximum(metrics, values)
            row = dict(frame=f, time_s=float(data['time'][f]), source_rsi_allowed=bool(data['source_rsi_allowed'][f]),
                       shock_screen_pass=passed, preview_hand_position_error_m=discrepancies[f],
                       **{k: v[f].item() for k,v in geom.items()})
            row.update(zip(('root_delta_speed_max_mps','root_displacement_max_m','box_speed_max_mps',
                            'box_displacement_max_m','physx_overlap_max_m','box_contact_force_peak_N','box_drop_max_m'), metrics.tolist()))
            row['all_carriers_hand_contact_step_fraction'] = float(np.mean(touches))
            row['geometry_screen_pass'] = max(row[k] for k in geom if k != 'worst_nonhand_shape') <= .02
            row['conservative_rsi_screen_pass'] = passed and row['geometry_screen_pass'] and row['source_rsi_allowed']
            records.append(row)
        # Free-root, physically simulated reference-PD playback; targets alone
        # are updated. This controller is not a trained balancing/carry policy.
        scene.reset(0)
        trajectory = [scene.root.numpy().copy()]
        bodies = []
        contact = []
        for step in range(2*(count-1)):
            t = (step+1)/2
            lo, hi = int(t), min(int(t)+1,count-1)
            scene.targets((1-(t-lo))*data['dof_position'][lo]+(t-lo)*data['dof_position'][hi])
            scene.step()
            if step % 2 == 1:
                trajectory.append(scene.root.numpy().copy())
                bodies.append(scene.body.numpy().copy())
                contact.append(scene.contacts()[0])
        trajectory = np.asarray(trajectory)
        box_error = np.linalg.norm(trajectory[:,-1,:3]-data['box_position'],axis=-1)
        np.savez_compressed(output/'pd_rollout.npz', root_states=trajectory,
                            body_states_after_frame0=np.asarray(bodies), hand_contacts=np.asarray(contact),
                            reference_time=data['time'], box_reference_error_m=box_error)
        allowed = [r for r in records if r['source_rsi_allowed']]
        summary = dict(frames=count, source_allowed_frames=len(allowed), box_mass_kg=scene.box_mass,
                       reset_fk_max_error_m=reset_error,
                       preview_geometry_maximum={k:float(v.max()) for k,v in preview_geom.items()
                                                 if k != 'worst_nonhand_shape'},
                       shock_pass_all=sum(r['shock_screen_pass'] for r in records),
                       shock_pass_source_allowed=sum(r['shock_screen_pass'] for r in allowed),
                       geometry_pass_source_allowed=sum(r['geometry_screen_pass'] for r in allowed),
                       conservative_pass_source_allowed=sum(r['conservative_rsi_screen_pass'] for r in allowed),
                       maximum={k:max(r[k] for r in records) for k in (
                           'hand_box_penetration_m','nonhand_box_penetration_m','floor_penetration_m',
                           'interperson_penetration_lower_bound_m','root_delta_speed_max_mps','box_speed_max_mps',
                           'physx_overlap_max_m','box_contact_force_peak_N')},
                       pd_rollout=dict(duration_s=float(data['time'][-1]), box_final_error_m=float(box_error[-1]),
                                       box_max_error_m=float(box_error.max()),
                                       all_carriers_hand_contact_fraction=float(np.all(contact,axis=-1).mean()),
                                       min_root_height_m=float(trajectory[:,:scene.n,2].min())))
    finally:
        scene.close()
    with (output/'frames.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    np.savez_compressed(output/'screen_masks.npz',
                        shock_pass=np.asarray([r['shock_screen_pass'] for r in records]),
                        conservative_rsi_screen_pass=np.asarray([r['conservative_rsi_screen_pass'] for r in records]),
                        source_rsi_allowed=data['source_rsi_allowed'], rsi_ready=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def plot_report(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(14,7))
    for col,skill in enumerate(SKILLS):
        for name,color in [('original','#2477b4'),('cooperative','#d35d39')]:
            path=output/skill/name
            with (path/'frames.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            times=[float(r['time_s']) for r in rows]
            axes[0,col].plot(times,[100*float(r['nonhand_box_penetration_m']) for r in rows],
                             color=color,label=name)
            rollout=np.load(path/'pd_rollout.npz')
            axes[1,col].plot(rollout['reference_time'],rollout['box_reference_error_m'],color=color,label=name)
        axes[0,col].axhline(2,color='black',ls='--',lw=1,label='2 cm screen limit')
        axes[0,col].set_title(skill)
        axes[0,col].set_ylabel('Non-hand body / box penetration (cm)')
        axes[1,col].set_ylabel('PD rollout box position error (m)')
        for ax in axes[:,col]:
            ax.set_xlabel('Source time (s)'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Carry physics audit: actual 32-DOF geometry + free-root PhysX\n'
                 'Native joint PD only; no learned balancing controller',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.91))
    fig.savefig(output/'physics_summary.png',dpi=150)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('joint_carry/archive/carry_joint_preview_all_inset15'))
    parser.add_argument('--output',type=Path,default=Path('joint_carry/archive/carry_physics_validation'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    report=dict(engine='Isaac Gym PhysX CPU, free roots, native asset PD gains',
                dt=1/60, substeps=2, shock_seconds=.1, shock_thresholds=SCREEN,
                geometry_penetration_limit_m=.02, density_kg_m3=100.,
                input=str(args.input), input_hashes={},
                limitations=['No learned balance/carry controller. Tracking failure does not prove no controller could succeed.',
                             'Physical elbows are 1-DOF; source elbow X/Z and fixed wrist rotations are omitted by the existing MotionLib conversion. Geometry uses actual simulator FK.',
                             'Geometry screen omits mesh-vs-box/self collision; actual PhysX includes them.',
                             'CPU PhysX audit is not an exact GPU-training rollout.',
                             'No input frames or metadata were overwritten; passing screens do not certify long-term stability.'],
                results={})
    for skill in SKILLS:
        report['results'][skill]={}
        for name,file in [('original','original_joints.npz'),('cooperative','cooperative_rsi_candidate.npz')]:
            path=args.input/skill/file
            report['input_hashes'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            data=dict(np.load(path,allow_pickle=False))
            output=args.output/skill/name; output.mkdir(parents=True,exist_ok=True)
            summary=audit(data,output)
            report['results'][skill][name]=summary
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(skill,name,json.dumps(summary),flush=True)
    plot_report(report,args.output)


if __name__=='__main__':
    main()

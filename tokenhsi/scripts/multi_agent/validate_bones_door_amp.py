"""GPU6-only non-training loader/AMP/FK check and scripted reference preview.

The door angle is inferred from the hand direction for display. It is neither
measured object motion nor proof that a policy can open the passive hinge.
"""
import os
import subprocess
gpu6_uuid = subprocess.check_output(['nvidia-smi', '-i', '6', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
if os.environ.get('TOKENHSI_GPU') not in ('6', gpu6_uuid):
    raise RuntimeError('This validation must be launched with TOKENHSI_GPU=6')
from isaacgym import gymapi, gymtorch
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from isaacgym.torch_utils import quat_mul, quat_rotate
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from env.tasks.multi_agent.humanoid_ma_door import HumanoidMADoor

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
torch.set_num_threads(1)
args = get_args()
if args.num_envs != 2 or not args.test or not args.eval or args.device != 'cuda':
    raise ValueError('Requires 2 evaluation environments on GPU, without a policy')
cfg, train, _ = load_cfg(args)
cfg['args'] = args
cfg['env']['motion_file'] = args.motion_file
cfg['env']['enableTrackInitState'] = True
capture = os.environ.get('CAPTURE_BONES', '0') == '1'
if capture:
    raise RuntimeError('GPU6 offscreen graphics initialization failed on this host; use the CPU preview instead')
cfg['env']['video'] = dict(enable=capture, width=800, height=600, focus='agent0')
sim = parse_sim_params(args, cfg, train)
sim.physx.num_threads = 1
set_seed(42)
output = Path(os.environ.get('OUTPUT_PATH', 'output/door_bones_left_amp_check'))
output.mkdir(parents=True, exist_ok=True)
class ReferenceAuditDoor(HumanoidMADoor):
    def create_sim(self):
        # Vulkan graphics ordinals are physical; CUDA is remapped by the runtime.
        # Never use the inherited graphics=0 for this GPU6-only diagnostic.
        self.graphics_device_id = 6 if capture else -1
        return super().create_sim()


task = ReferenceAuditDoor(cfg, sim, args.physics_engine, args.device, args.device_id, True)
report = dict(physical_gpu=6, training=False, scripted_preview=True,
              door_angle_source='inferred from hand XY; not source object motion', checks={})
try:
    env_ids = torch.arange(2, device=task.device)
    task._reset_envs(env_ids)
    assert set(task._motion_lib) == {'loco', 'doorOpen'}
    lib = task._motion_lib['doorOpen']
    assert lib.num_motions() == 1
    motion = lib.get_motion(0)
    assert motion.fps == 30 and len(motion.tensor) == 103
    ids = torch.zeros(103, dtype=torch.long, device=task.device)
    times = torch.arange(103, device=task.device)/30
    states = lib.get_motion_state(ids, times)
    assert all(torch.isfinite(v).all() for v in states)
    assert states[2].shape == (103, 32)
    # Exercise both expert libraries and the actual task AMP observation builder.
    task._skill_disc_prob.copy_(torch.tensor([0., 1.], device=task.device))
    door_demo = task.fetch_amp_obs_demo(64).clone()
    task._skill_disc_prob.copy_(torch.tensor([1., 0.], device=task.device))
    loco_demo = task.fetch_amp_obs_demo(64).clone()
    task._skill_disc_prob.copy_(torch.tensor([.3, .7], device=task.device))
    mixed_demo = task.fetch_amp_obs_demo(64).clone()
    assert all(torch.isfinite(v).all() and v.shape == (64, task.get_num_amp_obs())
               for v in (door_demo, loco_demo, mixed_demo))
    assert not torch.equal(door_demo, loco_demo)
    from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
    builder = AMPMultiAgentBuilder()
    builder.load(train['params']['network'])
    network = builder.build('amp', actions_num=32, input_shape=(task.get_obs_size(),),
        amp_input_shape=(task.get_num_amp_obs(),), value_size=1, num_agents=2, num_objects=2,
        humanoid_obs_size=230, object_obs_size=39, goal_obs_size=6, observation_mode='clean_scene',
        scene_entity_sizes=[223,35,1], scene_kinematic_size=7, scene_arena_scale=5.,
        relation_reward_mode=cfg['env']['relationReward']['mode'],
        relation_graph_spec=cfg['env']['relationGraph'], device=task.device).to(task.device)
    with torch.no_grad():
        logits = network.eval_disc(door_demo)
    assert logits.shape == (64, 1) and torch.isfinite(logits).all()
    contract = task.get_door_experiment_contract()
    assert contract['expert_skills'] == ['loco', 'doorOpen']
    assert contract['expert_mapping'] == {'OPEN': [.3, .7]}
    report['checks'].update(loader=True, dof_shape=[103, 32], amp_shape=list(door_demo.shape),
                            amp_finite=True, discriminator_forward_finite=True,
                            discriminator_logits_shape=list(logits.shape), both_expert_pools=True, contract=contract)
    root, rotation, dof, velocity, angular, dof_velocity, key = states
    body, body_rot, body_vel, body_ang = lib.get_motion_state_max(ids, times)
    q = torch.tensor([0., 0., np.sin(np.pi/4), np.cos(np.pi/4)], device=task.device, dtype=root.dtype)
    roots = quat_rotate(q.expand_as(rotation), root)
    bodies = quat_rotate(q.expand(len(body)*15, 4), body.reshape(-1, 3)).view_as(body)
    anchor = round(1.25932*30)
    # Align at contact in XY only. Grounded Z and hand height are preserved.
    offset = task.door_handle_position[0, 0, :2]-bodies[anchor, 8, :2]
    roots[:, :2] += offset
    bodies[..., :2] += offset
    hinge = task._box_states[0, 0, :3].clone()
    delta = bodies[:, 8, :2]-hinge[:2]
    raw_angle = torch.atan2(delta[:, 1], delta[:, 0])-np.arctan2(-.1, .78)
    angles = torch.cummax(raw_angle.clamp(0, 1.7), dim=0).values
    angles[:anchor] = 0
    handle = torch.stack((.78*torch.cos(angles)+.10*torch.sin(angles),
                          .78*torch.sin(angles)-.10*torch.cos(angles), torch.ones_like(angles)), -1)+hinge
    distance = (bodies[:, 8]-handle).norm(dim=-1)
    report['door_geometry'] = dict(contact_anchor_s=anchor/30, max_inferred_open_angle_rad=angles.max().item(),
        contact_anchor_distance_m=distance[anchor].item(),
        hand_handle_distance_after_anchor_median_m=distance[anchor:].median().item(),
        hand_handle_distance_after_anchor_max_m=distance[anchor:].max().item(),
        contact_not_verified=True, clearance_not_verified=True)
    # Validate rotation/DOF conventions in isolation from door contact.
    task._box_states[..., 0] -= 10
    task._humanoid_root_states[0, 1, 0] = 15
    task._set_env_state(torch.tensor([0], device=task.device), torch.tensor([0], device=task.device),
                        roots[:1], quat_mul(q[None], rotation[:1]), dof[:1],
                        torch.zeros_like(velocity[:1]), torch.zeros_like(angular[:1]), torch.zeros_like(dof_velocity[:1]))
    task._every_env_init_dof_pos[0, 0] = dof[0]
    task._reset_env_tensors(env_ids)
    task.pre_physics_step(torch.zeros(4, 32, device=task.device))
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    task._refresh_sim_tensors()
    isolated_error = (task._rigid_body_pos[0, 0]-bodies[0]).norm(dim=-1).max().item()
    report['checks']['isolated_gym_vs_poselib_body_error_max_m'] = isolated_error
    assert isolated_error < .02, ('Isolated DOF/FK convention mismatch', isolated_error)
    task._reset_envs(env_ids)
    frames = []
    fk_errors = []
    # Non-training replay: prescribe human states and door angle, no policy.
    for f in range(0, 103, 2):
        task._set_env_state(torch.tensor([0], device=task.device), torch.tensor([0], device=task.device),
                            roots[f:f+1], quat_mul(q[None], rotation[f:f+1]), dof[f:f+1],
                            torch.zeros_like(velocity[f:f+1]), torch.zeros_like(angular[f:f+1]),
                            torch.zeros_like(dof_velocity[f:f+1]))
        task._every_env_init_dof_pos[0, 0] = dof[f]
        # Move other human out of shot; retain 2-human task contract.
        task._humanoid_root_states[0, 1, 0] = 15
        task._humanoid_root_states[0, 1, 7:] = 0
        task.reset_door_angles(env_ids[:1], torch.tensor([[angles[f].item(), 0.]], device=task.device), apply_to_sim=False)
        task._reset_env_tensors(env_ids)
        task.pre_physics_step(torch.zeros(4, 32, device=task.device))
        task.gym.simulate(task.sim)
        task.gym.fetch_results(task.sim, True)
        task._refresh_sim_tensors()
        task._compute_observations()
        assert torch.isfinite(task.obs_buf).all()
        actual = task._rigid_body_pos[0, 0]
        fk_errors.append((actual-bodies[f]).norm(dim=-1).max().item())
        if capture:
            cam = task._video_cam
            task.gym.set_camera_location(cam, task.envs[0],
                gymapi.Vec3(float(hinge[0])+2.8, float(hinge[1])-3.2, 2.3),
                gymapi.Vec3(float(hinge[0])+.35, float(hinge[1])+.15, 1.))
            task.gym.step_graphics(task.sim)
            task.gym.render_all_camera_sensors(task.sim)
            rgba = task.gym.get_camera_image(task.sim, task.envs[0], cam, gymapi.IMAGE_COLOR).reshape(600, 800, 4)
            image = Image.fromarray(rgba[:, :, :3].copy())
            ImageDraw.Draw(image).text((12, 12), f'Retarget replay {f/30:.2f}s / scripted door (NOT policy)', fill='white')
            frames.append(image)
    report['checks']['simulation_observations_finite'] = True
    report['checks']['replay_frames'] = len(fk_errors)
    report['checks']['gym_vs_poselib_body_error_max_m'] = max(fk_errors)
    # One step introduces gravity/contact differences; this also detects axis mapping errors.
    assert max(fk_errors) < .12, ('Gym vs Poselib FK mismatch', max(fk_errors))
    if frames:
        frames[0].save(output/'reference_replay.gif', save_all=True, append_images=frames[1:], duration=67, loop=0)
        selected = [frames[i] for i in (0, 19, 29, 39, 51)]
        sheet = Image.new('RGB', (800*len(selected), 600))
        for i, image in enumerate(selected):
            sheet.paste(image, (800*i, 0))
        sheet.save(output/'reference_contact_sheet.jpg')
    report['status'] = 'passed'
    print('BONES_DOOR_AMP_CHECK_OK', json.dumps(report), flush=True)
finally:
    (output/'validation_report.json').write_text(json.dumps(report, indent=2)+'\n')
    task.gym.destroy_sim(task.sim)

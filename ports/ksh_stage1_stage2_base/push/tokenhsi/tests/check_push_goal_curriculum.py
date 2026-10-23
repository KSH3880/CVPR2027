"""GPU reset/observation/physics audit of the isolated Stage1 PUSH extension."""
from isaacgym import gymapi
import json
from pathlib import Path
import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.push_point import point_geometry, world_points
from utils.stage1_push import hand_box_overlap
from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push

args = get_args()
cfg, train, _ = load_cfg(args); cfg['args'] = args; cfg['env']['motion_file'] = args.motion_file
sim = parse_sim_params(args, cfg, train); set_seed(42); torch.set_num_threads(1)
args.eval = False  # Exercise the configured 50% training RSI law without a PPO run.
task = HumanoidMAStage1Push(cfg, sim, args.physics_engine, args.device, args.device_id, True)
ids = torch.arange(task.num_envs, device=task.device)
try:
    masses = []; frictions = []
    for i, actor in enumerate(task._box_handles):
        bodies = task.gym.get_actor_rigid_body_properties(task.envs[i//4], actor)
        shapes = task.gym.get_actor_rigid_shape_properties(task.envs[i//4], actor)
        masses.append(sum(body.mass for body in bodies)); frictions.extend(shape.friction for shape in shapes)
    assert all(abs(mass-cfg['env']['pushBox']['mass']) < 1e-4 for mass in masses)
    assert all(abs(friction-.6) < 1e-6 for friction in frictions)
    interaction = samples = 0; shortest = float('inf'); longest = 0.
    initial_shortest = float('inf'); initial_longest = 0.
    native_samples=[]
    height_min=float('inf');height_max=0.
    rsi_raw_min=float('inf');rsi_raw_max=0.;rsi_placed_min=float('inf');rsi_placed_max=0.
    loco_geometry=dict(agent_samples=0,hand_overlap_count=0,hand_center_inside=0,
                       body_center_inside=0,max_hand_overlap_m=0.,min_root_rear_face_gap_m=float('inf'),
                       min_root_center_distance_m=float('inf'),max_root_center_distance_m=0.)
    for audit_index in range(32):
        task._push_goal_epoch = (0, 999, 1000, 3000)[audit_index//8]
        task._reset_envs(ids)
        target = world_points(task._box_states[:, :2], task._push_local)
        height_min=min(height_min,float(target[...,2].min()));height_max=max(height_max,float(target[...,2].max()))
        if cfg['env']['pushTask'].get('height_mode') == 'amp_reference':
            assert (target[..., 2] > 0).all() and (target[..., 2] < task._box_size[:, :2, 2]).all()
            r, a = task._push_rsi_interaction.nonzero(as_tuple=True)
            actual = task._kinematic_humanoid_rigid_body_states[r, a][:, task._key_body_ids[:2], 2].mean(-1)
            torch.testing.assert_close(target[r, a, 2], actual, atol=1e-6, rtol=0)
        else:
            torch.testing.assert_close(target[..., 2], torch.full_like(target[..., 2], cfg['env']['pushTask']['height_m']), atol=1e-6, rtol=0)
        torch.testing.assert_close(task._box_states[..., 2], task._box_size[..., 2]/2, atol=1e-6, rtol=0)
        assert torch.all(task._box_size[..., 2] == 1.3)
        if cfg['env']['pushBox'].get('tie_xy', False):
            assert torch.equal(task._box_size[..., 0], task._box_size[..., 1])
        if 'size_y_m' in cfg['env']['pushBox']:
            assert torch.all(task._box_size[..., 1] == cfg['env']['pushBox']['size_y_m'])
            assert torch.all(task._push_side == 2)
            torch.testing.assert_close(task._push_local[..., 1], -task._box_size[:, :2, 1]/2)
            assert (task._push_local[..., 0] == 0).all()
        assert task.get_scene_num_tokens() == 8 and task.get_clean_object_obs_size() == 30
        assert task.num_objects == 4 and task.obs_buf.shape[1] == 634
        raw = HumanoidMACarry._compute_clean_scene_obs(task, ids)
        torch.testing.assert_close(raw, task.obs_buf, rtol=0, atol=0)
        # For RSI starts recover reset-to-goal distances from the actual sampler's stored initial states.
        delta = task._tar_pos-task._box_states[:, :2, :3]
        from utils.door_reference import rotate
        forward = rotate(task._box_states[:, :2, 3:7], delta.new_tensor([0., 1., 0.]).expand_as(delta))
        remaining = delta.norm(dim=-1)
        initial_distance = remaining/(1-task._push_rsi_samples['box_progress'])
        from utils.push_goal_curriculum import distance_range
        lo, hi = distance_range(cfg['env'], task._push_goal_epoch)
        assert initial_distance.min() >= lo-.002 and initial_distance.max() <= hi+.002
        initial_shortest = min(initial_shortest, initial_distance.min().item())
        initial_longest = max(initial_longest, initial_distance.max().item())
        # World-position subtraction loses a few float32 ulps at distant env origins.
        # Compare displacement directly: dividing by a short RSI remainder
        # amplifies the world-coordinate subtraction error at distant origins.
        torch.testing.assert_close(delta, forward*remaining[..., None], atol=2e-4, rtol=0)
        assert remaining.min() >= .21
        shortest = min(shortest, remaining.min().item()); longest = max(longest, remaining.max().item())
        assert not task.relation_runtime.done.any()
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
        geometry=task._push_reset_geometry()
        if geometry is not None:
            poses=task._kinematic_humanoid_rigid_body_states.reshape(task.num_envs*2,-1,13)
            assert not geometry.invalid(poses[...,:3],poses[...,3:7],task._box_states[:,:2].reshape(-1,13),task._box_size[:,:2].reshape(-1,3)).any()
            assert geometry.floor_shift(poses[...,:3],poses[...,3:7]).max()<1e-4
        lr,la=(~task._push_rsi_interaction).nonzero(as_tuple=True)
        if len(lr):
            body=task._kinematic_humanoid_rigid_body_states[lr,la,:,:3]
            box=task._box_states[lr,la];size=task._box_size[lr,la]
            depth,inside=hand_box_overlap(body[:,task._key_body_ids[:2]],box,size)
            inv=box[:,3:7].clone();inv[:,:3]*=-1
            local=rotate(inv[:,None].expand(-1,body.shape[1],-1),body-box[:,None,:3])
            body_inside=(local.abs()<size[:,None]/2).all(-1)
            loco_geometry['agent_samples']+=len(lr)
            loco_geometry['hand_overlap_count']+=int((depth>1e-5).sum())
            loco_geometry['hand_center_inside']+=int(inside.sum())
            loco_geometry['body_center_inside']+=int(body_inside.sum())
            loco_geometry['max_hand_overlap_m']=max(loco_geometry['max_hand_overlap_m'],depth.max().item())
            gap=-local[:,0,1]-size[:,1]/2
            loco_geometry['min_root_rear_face_gap_m']=min(loco_geometry['min_root_rear_face_gap_m'],gap.min().item())
            root_distance=local[:,0,:2].norm(dim=-1)
            start_lo,start_hi=cfg['env']['pushTask']['loco_distance_range_m']
            assert root_distance.min()>=start_lo-.001 and root_distance.max()<=start_hi+.001
            loco_geometry['min_root_center_distance_m']=min(loco_geometry['min_root_center_distance_m'],root_distance.min().item())
            loco_geometry['max_root_center_distance_m']=max(loco_geometry['max_root_center_distance_m'],root_distance.max().item())
            assert not body_inside.any() and not inside.any() and depth.max()<=1e-5
        if cfg['env']['pushTask'].get('rsi_sampling') == 'native_size_cache':
            mask=task._push_rsi_interaction
            r,a=mask.nonzero(as_tuple=True)
            clips=task._push_rsi_samples['motion_id'][r,a];times=task._push_rsi_samples['time'][r,a]
            original=task._motion_lib['push'].get_motion_state(clips,times)
            torch.testing.assert_close(task._dof_pos[r,a],original[2],atol=1e-6,rtol=0)
            torch.testing.assert_close(task._dof_vel[r,a],original[5],atol=1e-6,rtol=0)
            if cfg['env']['pushTask'].get('rsi_max_hand_midpoint_height_m') is not None:
                raw=task._motion_lib['push'].get_motion_state_max(clips,times)[0]
                placed=task._kinematic_humanoid_rigid_body_states[r,a,:,:3]
                raw_z=raw[:,task._key_body_ids[:2],2].mean(-1)
                placed_z=placed[:,task._key_body_ids[:2],2].mean(-1)
                limit=cfg['env']['pushTask']['rsi_max_hand_midpoint_height_m']
                assert (raw_z<=limit+1e-6).all() and (placed_z<=limit+1e-6).all()
                # Every body's Z changes by the same rigid floor translation; no posture edit.
                dz=placed[...,2]-raw[...,2]
                torch.testing.assert_close(dz,dz[:,:1].expand_as(dz),atol=2e-6,rtol=0)
                rsi_raw_min=min(rsi_raw_min,float(raw_z.min()));rsi_raw_max=max(rsi_raw_max,float(raw_z.max()))
                rsi_placed_min=min(rsi_placed_min,float(placed_z.min()));rsi_placed_max=max(rsi_placed_max,float(placed_z.max()))
            native_samples.extend(dict(motion_id=int(c),time=float(t),late=bool(l)) for c,t,l in zip(clips.cpu(),times.cpu(),task._push_rsi_samples['late_requested'][r,a].cpu()))
        interaction += int(task._push_rsi_interaction.sum()); samples += len(ids)*2
    before = [x[1:].clone() for x in (task._box_states, task._tar_pos, task._humanoid_root_states, task._amp_obs_buf, task._push_local)]
    task._reset_envs(ids[:1])
    for x, y in zip(before, (task._box_states, task._tar_pos, task._humanoid_root_states, task._amp_obs_buf, task._push_local)):
        torch.testing.assert_close(x, y[1:], rtol=0, atol=0)
    goals = task._tar_pos.clone()
    contact_local = task._push_local.clone()
    for _ in range(12):
        task.pre_physics_step(torch.zeros(task.num_envs*2, task.num_actions, device=task.device))
        task._physics_step(); task.post_physics_step()
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all() and torch.isfinite(task._amp_obs_buf).all()
    torch.testing.assert_close(goals, task._tar_pos, rtol=0, atol=0)
    torch.testing.assert_close(contact_local, task._push_local, rtol=0, atol=0)
    out = Path(args.output_path); out.mkdir(parents=True, exist_ok=True)
    report = dict(objects=4,humans=2,goals=2,tokens=8,observation_width=634,object_width=30,
                  native_observation_identical=True,reset_agent_samples=samples,rsi_fraction=interaction/samples,
                  actual_remaining_distance_m=[shortest,longest],partial_reset_preserved=True,
                  reconstructed_initial_goal_distance_m=[initial_shortest,initial_longest],
                  loco_initial_geometry=loco_geometry,
                  full_collider_bounds_clear=cfg['env']['pushTask'].get('reset_geometry_filter')=='collider_bounds_v1',
                  xy_tied=cfg['env']['pushBox'].get('tie_xy', False),
                  finite_control_steps=12,goal_fixed_during_rollout=True,
                  distinct_box_xyz_sizes=len(torch.unique(task._box_size.reshape(-1,3),dim=0)),
                  box_xyz_min_m=task._box_size.reshape(-1,3).min(0).values.cpu().tolist(),
                  box_xyz_max_m=task._box_size.reshape(-1,3).max(0).values.cpu().tolist(),
                  rsi_height_selection_limit_m=cfg['env']['pushTask'].get('rsi_max_hand_midpoint_height_m'),
                  rsi_reference_height_range_m=[rsi_raw_min,rsi_raw_max] if rsi_raw_max else None,
                  rsi_placed_height_range_m=[rsi_placed_min,rsi_placed_max] if rsi_placed_max else None,
                  midpoint_target_height_mode=cfg['env']['pushTask'].get('height_mode', 'fixed'),
                  midpoint_target_initial_height_m=(None if cfg['env']['pushTask'].get('height_mode') == 'amp_reference'
                                                     else cfg['env']['pushTask']['height_m']),
                  midpoint_target_actual_height_range_m=[height_min,height_max],
                  contact_target_fixed_during_rollout=True,
                  actual_box_masses_kg=masses,actual_shape_frictions=frictions)
    (out/'audit.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=2))
    if native_samples:
        counts={i:sum(r['motion_id']==i for r in native_samples) for i in range(21)}
        assert all(counts.values())
        sampling=dict(samples=native_samples,per_motion=counts,late_fraction=sum(r['late'] for r in native_samples)/len(native_samples),
            original_dof_positions_and_velocities_identical=True,
            distinct_clip_times=len({(r['motion_id'],round(r['time'],5)) for r in native_samples}))
        (out/'sampling.json').write_text(json.dumps(sampling,indent=2)+'\n')
finally:
    task.gym.destroy_sim(task.sim)

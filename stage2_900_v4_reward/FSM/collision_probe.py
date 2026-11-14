import json
import os
from pathlib import Path

from isaacgym import gymapi, gymtorch
import numpy as np
import torch

from controller import StateFreeze


@torch.no_grad()
def run_probe(player):
    task = player.env.task
    task._design = True
    task._show_lines_flag = False
    player._amp_debug = lambda info: None
    player._reward_debug = False
    destination = Path(task.cfg['args'].output_path)
    steps = int(os.environ['FSM_STEPS'])
    task.cfg['env']['episodeLength'] = steps
    old_camera = task._update_camera
    task._update_camera = lambda: None
    canonical = None
    results = []
    modes = ('frozen_hit','free_hit','free_nohit')
    for episode in range(int(os.environ['FSM_EPISODES'])):
        mode = modes[episode % len(modes)]
        path = destination/f'episode_{episode:03d}_{mode}'
        path.mkdir(parents=True,exist_ok=True)
        player.env_reset()
        target = int((task._before_predecessor[0]>=0).nonzero()[0])
        if canonical is None:
            leader = int(task._before_predecessor[0,target])
            source = int(task._scenario_physical_object(torch.tensor([0],device=task.device),
                                                       torch.tensor([leader],device=task.device))[0])
            canonical = dict(root=task._humanoid_root_states[0,target].clone(),
                dof=task._dof_pos[0,target].clone(),body=task._kinematic_humanoid_rigid_body_states[0,target].clone(),
                source=source,size=task._box_size[0,source].clone())
        source = canonical['source']
        root = canonical['root'].clone()
        root[:2] = task._env_origins[0,:2]
        root[7:] = 0
        body = canonical['body'].clone()
        body[:,:3] += root[:3]-canonical['root'][:3]
        body[:,7:] = 0
        for person in range(task.num_agents):
            offset = root.new_tensor([0.,0.,0.] if person==target else [30.,30.,0.])
            task._humanoid_root_states[0,person] = root
            task._humanoid_root_states[0,person,:3] += offset
            task._dof_pos[0,person] = canonical['dof']
            task._dof_vel[0,person] = 0
            task._kinematic_humanoid_rigid_body_states[0,person] = body
            task._kinematic_humanoid_rigid_body_states[0,person,:,:3] += offset
        task._box_states[0,:,:2] = task._env_origins[0,:2]+1000.+12*torch.arange(task.num_physical_objects,device=task.device)[:,None]
        task._box_states[0,:,7:] = 0
        box_ids = task._box_actor_ids.flatten().contiguous()
        task.gym.set_actor_root_state_tensor_indexed(task.sim,gymtorch.unwrap_tensor(task._root_states),
                                                    gymtorch.unwrap_tensor(box_ids),len(box_ids))
        if task._enable_markers:
            task._marker_pos[:] = task._env_origins[:,None]+1000.
            marker_ids = task._marker_actor_ids.flatten().contiguous()
            task.gym.set_actor_root_state_tensor_indexed(task.sim,gymtorch.unwrap_tensor(task._root_states),
                                                        gymtorch.unwrap_tensor(marker_ids),len(marker_ids))
        controller = StateFreeze(task)
        controller.frozen[:] = True
        controller.project(submit=True)
        task._action_to_pd_targets = lambda actions: controller.saved_dof
        controller.color()
        if task.viewer is not None:
            task.gym.viewer_camera_look_at(task.viewer,task.envs[0],gymapi.Vec3(3.,-3.,2.),gymapi.Vec3(0.,0.,.9))
            handle = task._box_handles[source]
            task.gym.set_rigid_body_color(task.envs[0],handle,0,gymapi.MESH_VISUAL,gymapi.Vec3(.95,.25,.15))
        records = {key:[] for key in ('root','box','human_contact_forces','box_contact_force')}
        launch_step = 30
        print(f'[collision probe] episode={episode} mode={mode}; launch/release at step={launch_step}; speed=3m/s',flush=True)
        try:
            for frame in range(steps):
                if frame == launch_step:
                    if mode != 'frozen_hit':
                        controller.frozen[0,target] = False
                        handle = task.humanoid_handles_all[0][target]
                        if task.viewer is not None:
                            for index in range(task.num_bodies):
                                task.gym.set_rigid_body_color(task.envs[0],handle,index,gymapi.MESH_VISUAL,gymapi.Vec3(.5,.9,.2))
                    if mode != 'free_nohit':
                        projectile = task._box_states[0,source]
                        projectile[:] = 0
                        projectile[:3] = root[:3]
                        projectile[0] -= canonical['size'][0]/2+.6
                        projectile[2] += .1
                        projectile[6] = 1
                        projectile[7] = 3.
                        actors = task._box_actor_ids[0,source:source+1].contiguous()
                        task.gym.set_actor_root_state_tensor_indexed(task.sim,gymtorch.unwrap_tensor(task._root_states),
                                                                    gymtorch.unwrap_tensor(actors),len(actors))
                actions = torch.zeros(task.num_envs*task.num_agents,task.num_dof,device=task.device)
                player.env_step(player.env,actions)
                controller.frame = frame+1
                for key,value in dict(root=task._humanoid_root_states[0,target],box=task._box_states[0,source],
                    human_contact_forces=task._contact_forces[0,target],box_contact_force=task._ontop_box_contact[0,source]).items():
                    records[key].append(value.cpu().numpy().copy())
                if task.viewer is not None and os.environ.get('FSM_CAPTURE')=='1' and frame+1 in (1,30,33,36,40,50,60,120):
                    task.render()
                    task.gym.write_viewer_image_to_file(task.viewer,str(path/f'{frame+1:04d}.png'))
        finally:
            proof = controller.finish()
        raw = {key:np.asarray(value) for key,value in records.items()}
        np.savez_compressed(path/'collision_trace.npz',**raw)
        displacement = np.linalg.norm(raw['root'][:,:2]-root[:2].cpu().numpy(),axis=-1)
        result = dict(mode=mode,episode=episode,target=target,steps=steps,control_dt=task.dt,
            launch_step=launch_step,launch_speed_mps=3.,box_size_m=canonical['size'].cpu().tolist(),
            max_root_xy_displacement_m=float(displacement.max()),
            max_human_contact_force_N=float(np.linalg.norm(raw['human_contact_forces'][launch_step:],axis=-1).max()),
            max_box_contact_force_N=float(np.linalg.norm(raw['box_contact_force'][launch_step:],axis=-1).max()),
            freeze=proof,policy_actions_used=False,formal_sr_evaluation=False,
            pd_targets='same saved joint pose in all cases',initial_pose_identical=True,
            note='Contact forces include floor contacts. Compare time series and free_nohit control; not a pure collision impulse measurement.')
        if not all(np.isfinite(values).all() for values in raw.values()):
            raise ValueError('Non-finite collision probe state')
        (path/'probe.json').write_text(json.dumps(result,indent=2))
        results.append(result)
        print('[collision probe result] '+json.dumps({key:result[key] for key in
            ('mode','max_root_xy_displacement_m','max_human_contact_force_N','max_box_contact_force_N')}),flush=True)
    task._update_camera = old_camera
    (destination/'collision_summary.json').write_text(json.dumps(dict(completed=True,results=results,
        purpose='controlled collision demonstration, not SR evaluation',true_kinematic_actor=False),indent=2))
    return results

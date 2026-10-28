"""Render reset poses only: no policy action, env_step, or physics advance."""
import json,time
from pathlib import Path
import torch
from isaacgym import gymapi,gymtorch


def install():
    from learning.multi_agent.ma_players import MAPlayerContinuous
    def run(self):
        task=self.env.task
        if task.viewer is None:raise ValueError('Pose viewer requires a graphical viewer')
        gym=task.gym
        for key,name in ((gymapi.KEY_N,'pose_next'),(gymapi.KEY_R,'pose_push'),(gymapi.KEY_L,'pose_loco')):
            gym.subscribe_viewer_keyboard_event(task.viewer,key,name)
        def reset(probability):
            task._viewer_push_probability=probability
            self.env_reset()
            # GPU reset tensors reach render transforms only after a simulation
            # sync. Materialize once at 1 microsecond with zero velocity/gravity;
            # then never simulate again until the user requests another pose.
            params=gym.get_sim_params(task.sim)
            saved_dt=params.dt
            saved_gravity=gymapi.Vec3(params.gravity.x,params.gravity.y,params.gravity.z)
            params.dt=1e-6;params.gravity=gymapi.Vec3(0.,0.,0.)
            gym.set_sim_params(task.sim,params)
            task._root_states[...,7:]=0
            task._dof_state[...,1]=0
            gym.set_actor_root_state_tensor(task.sim,gymtorch.unwrap_tensor(task._root_states))
            gym.set_dof_state_tensor(task.sim,gymtorch.unwrap_tensor(task._dof_state))
            pose_targets=task._dof_pos.contiguous()
            gym.set_dof_position_target_tensor(task.sim,gymtorch.unwrap_tensor(pose_targets))
            gym.simulate(task.sim);gym.fetch_results(task.sim,True)
            task._refresh_sim_tensors()
            params.dt=saved_dt;params.gravity=saved_gravity
            gym.set_sim_params(task.sim,params)
            task._camera_follow_flag=False
            rows=torch.arange(task.num_envs,device=task.device)[:,None]
            boxes=task._box_states[rows,task._agent_box_assignment][0]
            sizes=task._box_size[rows,task._agent_box_assignment][0]
            roots=task._humanoid_root_states[0,:,:3]
            points=torch.cat((roots,boxes[:,:3]-sizes/2,boxes[:,:3]+sizes/2),0)
            lower,upper=points.amin(0),points.amax(0)
            target=(lower+upper)/2;target[2]=1.
            distance=max(8.,float((upper[:2]-lower[:2]).norm())*1.2)
            camera=target+target.new_tensor([distance*.35,-distance,distance*.65])
            gym.viewer_camera_look_at(task.viewer,None,gymapi.Vec3(*camera.cpu().tolist()),gymapi.Vec3(*target.cpu().tolist()))
            chosen=task._interaction[0]
            category='PUSH' if chosen.all() else 'LOCO' if (~chosen).all() else 'MIXED'
            balanced=task.cfg['env']['stage2JointPush'].get('coupled_rsi_selection')=='balanced_group_ternary_v1'
            controls=('N=equal-thirds' if balanced else 'N=mix50')+' R=push L=loco; mouse=camera; Esc=quit'
            print('KSH_POSE_FROZEN',dict(category=category,push_probability=probability,interaction=chosen.cpu().tolist(),controls=controls),flush=True)
            return task._root_states.clone(),task._dof_state.clone()
        roots,dofs=reset(.5)
        frames=0
        while not gym.query_viewer_has_closed(task.viewer):
            for event in gym.query_viewer_action_events(task.viewer):
                if event.value<=0:continue
                if event.action=='QUIT':return
                if event.action in ('pose_next','pose_push','pose_loco'):
                    roots,dofs=reset({'pose_next':.5,'pose_push':1.,'pose_loco':0.}[event.action])
            gym.step_graphics(task.sim)
            gym.draw_viewer(task.viewer,task.sim,True)
            frames+=1
            if frames==5:
                assert torch.equal(task._root_states,roots) and torch.equal(task._dof_state,dofs)
                out=Path(task.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
                gym.write_viewer_image_to_file(task.viewer,str(out/'frozen_pose.png'))
                (out/'pose_verification.json').write_text(json.dumps(dict(render_frames=5,root_unchanged=True,dofs_unchanged=True,physics_materialization_steps=1,materialization_dt=1e-6,physics_steps_during_view=0,policy_steps=0),indent=2))
                print('KSH_POSE_STATIC_VERIFIED',flush=True)
            time.sleep(1/30)
    MAPlayerContinuous.run=run
    MAPlayerContinuous.run_eval=run

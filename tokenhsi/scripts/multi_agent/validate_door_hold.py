"""Non-training physical regression: push contact, unsupported-hold rejection, release/closure."""
import os
from isaacgym import gymapi, gymtorch
import json
from pathlib import Path
import torch
from isaacgym.torch_utils import quat_rotate, quat_mul
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from env.tasks.multi_agent.humanoid_ma_door_hold import HumanoidMADoorHold

torch.set_num_threads(1)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
args=get_args()
assert args.task=='HumanoidMADoorHold' and args.test and args.eval and args.num_envs==2 and args.num_agents==1
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True
sim=parse_sim_params(args,cfg,train);sim.physx.num_threads=1
set_seed(42)
task=HumanoidMADoorHold(cfg,sim,args.physics_engine,args.device,args.device_id,True)
out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
report={'training':False,'num_people':1,'num_doors':1,'device':task.device,'checks':{},
        'notes':'Prescribed real humanoid poses/contact. Only environmental closing torque acts directly on door; no policy success claim.'}
trace=[]
ids=torch.arange(2,device=task.device)


def step():
    task.pre_physics_step(torch.zeros(2,32,device=task.device))
    task._physics_step()
    task.gym.fetch_results(task.sim,True);task.post_physics_step()
    assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()


def separate():
    task._humanoid_root_states[...,0]-=10
    task._humanoid_root_states[...,7:]=0
    task._reset_env_tensors(ids)


try:
    task._reset_envs(ids)
    assert task.num_agents==1 and task.num_objects==1
    assert task.get_clean_object_obs_size()==38 and task.get_obs_size()==288
    assert task.get_relation_suffix_size()==5
    lo,hi=cfg['env']['doorSpawnDistanceRange'];samples=[]
    for _ in range(5):
        task._reset_envs(ids)
        distance=(task._humanoid_root_states[...,:2]-task.door_handle_position[...,:2]).norm(dim=-1)
        assert ((distance>=lo-1e-4)&(distance<=hi+1e-4)).all()
        samples.extend(distance.flatten().tolist())
    assert max(samples)-min(samples)>.2,'Reset distances are not randomized'
    other=task._humanoid_root_states[1].clone()
    task._reset_envs(ids[:1])
    torch.testing.assert_close(task._humanoid_root_states[1],other)
    report['checks']['reset_and_expert_sampling']={'handle_distance_samples_m':samples,'range_m':[lo,hi],'partial_reset_preserves_other_human':True,
        'initial_skills':cfg['env']['skillInitProb'],'expert_probabilities':cfg['env']['skillDiscProb'],
        'open_expert_mapping':cfg['env']['ampExpertMapping']['OPEN']}
    demo=task.fetch_amp_obs_demo(64);assert demo.shape==(64,1290) and torch.isfinite(demo).all()
    from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
    builder=AMPMultiAgentBuilder();builder.load(train['params']['network'])
    network=builder.build('amp',actions_num=32,input_shape=(task.get_obs_size(),),
        amp_input_shape=(task.get_num_amp_obs(),),value_size=1,num_agents=1,num_objects=1,
        humanoid_obs_size=230,object_obs_size=39,goal_obs_size=6,observation_mode='clean_scene',
        scene_entity_sizes=[223,38,1],scene_kinematic_size=7,scene_arena_scale=5.,
        relation_reward_mode=cfg['env']['relationReward']['mode'],
        relation_graph_spec=cfg['env']['relationGraph'],device=task.device).to(task.device)
    with torch.no_grad():
        logits=network.eval_disc(demo)
    assert logits.shape==(64,1) and torch.isfinite(logits).all()
    report['checks']['amp_discriminator']={'shape':list(logits.shape),'finite':True}
    report['checks']['loader_amp_observation']={'obs_size':task.get_obs_size(),'amp_shape':list(demo.shape),'contract':task.get_door_experiment_contract()}
    # No hand: an initially open door must return toward closed without success.
    separate();task.reset_door_angles(ids,torch.tensor([[1.25],[1.25]],device=task.device),apply_to_sim=False)
    task._reset_env_tensors(ids)
    start=task.door_hinge_angle.clone()
    print('HOLD_DOOR_PROPERTIES',task.gym.get_actor_dof_properties(task.envs[0],task._box_handles[0]),flush=True)
    for i in range(round(10/task.dt)):
        step();assert not task.relation_runtime.done.any()
        if i<5:print('CLOSING_TRACE',i,task.door_hinge_angle.tolist(),task.door_hinge_velocity.tolist(),task._closing_forces[:,-1].tolist(),flush=True)
    assert task.door_hinge_angle.max()<.1,task.door_hinge_angle
    report['checks']['automatic_close_without_hands']={'start_angles':start.tolist(),'end_angles':task.door_hinge_angle.tolist(),'seconds':10}
    with torch.no_grad():
        mu,sigma=network.eval_actor(task.obs_buf)
        value=network.eval_critic(task.obs_buf)
    assert mu.shape==(2,32) and value.shape==(2,1)
    assert all(torch.isfinite(x).all() for x in (mu,sigma,value))
    report['checks']['actor_critic']={'actions_shape':list(mu.shape),'value_shape':list(value.shape),'finite':True}
    task._reset_envs(ids);separate();task.reset_door_angles(ids,torch.zeros(2,1,device=task.device),apply_to_sim=False)
    task._reset_env_tensors(ids);step()
    lib=task._motion_lib['doorOpen'];motion_ids=torch.zeros(1,dtype=torch.long,device=task.device)
    time=torch.tensor([1.7],device=task.device)
    root,rot,dof,_,_,_,_=lib.get_motion_state(motion_ids,time)
    bodies,_,_,_=lib.get_motion_state_max(motion_ids,time)
    yaw=torch.tensor([[0.,0.,2**-.5,2**-.5]],device=task.device)
    root_rot=quat_mul(yaw,rot)[0]
    hand=8  # left_hand of phys_humanoid_v3
    offset=quat_rotate(yaw,bodies[:,hand]-root)[0]
    actor=task._humanoid_actor_ids[:1,:1].flatten().contiguous()
    body_index=task.gym.get_actor_rigid_body_index(task.envs[0],task.humanoid_handles_all[0][0],hand,gymapi.DOMAIN_ENV)
    door_indices={task.gym.get_actor_rigid_body_index(task.envs[0],task._box_handles[0],j,gymapi.DOMAIN_ENV) for j in range(task._door_body_count)}
    max_elapsed=0.;contact_steps=0;pair_steps=0;max_angle=0.;invalid_posture_steps=0;legacy_elapsed=0.;max_legacy_elapsed=0.
    for i in range(round(12/task.dt)):
        a=task.door_hinge_angle[0,0].clone().clamp(max=1.65)
        spin=torch.stack((a*0,a*0,torch.sin(a/2),torch.cos(a/2)))
        normal=torch.stack((-torch.sin(a),torch.cos(a),a*0))
        stand_off=max(.055,.13-i*.002)
        handle=task._box_states[0,0,:3]+quat_rotate(spin[None],a.new_tensor(task.door_local_handle_offset)[None])[0]
        wanted_hand=handle-normal*stand_off
        wanted_root=wanted_hand-quat_rotate(spin[None],offset[None])[0]
        wanted_root[2]=root[0,2]  # Preserve the reference; foot support must be measured, not fabricated.
        # Prescribe humans, never an opening door angle/positive hinge torque.
        task._humanoid_root_states[0,0,:3]=wanted_root
        task._humanoid_root_states[0,0,3:7]=quat_mul(spin[None],root_rot[None])[0]
        task._humanoid_root_states[0,0,7:]=0
        task._dof_pos[0,0]=dof[0];task._dof_vel[0,0]=0
        task._every_env_init_dof_pos[0,0]=dof[0]
        task.gym.set_actor_root_state_tensor_indexed(task.sim,gymtorch.unwrap_tensor(task._root_states),gymtorch.unwrap_tensor(actor),len(actor))
        task.gym.set_dof_state_tensor_indexed(task.sim,gymtorch.unwrap_tensor(task._dof_state),gymtorch.unwrap_tensor(actor),len(actor))
        step()
        contact=bool(task.relation_runtime.contact[0,0]);elapsed=task.relation_runtime.elapsed[0,0].item();angle=task.door_hinge_angle[0,0].item()
        posture=bool(task.relation_runtime.posture[0,0])
        above=angle>=cfg['env']['relationReward']['door']['target_angle']
        legacy_elapsed=legacy_elapsed+task.dt if contact and above else 0.
        max_legacy_elapsed=max(max_legacy_elapsed,legacy_elapsed)
        if contact and above and not posture:
            invalid_posture_steps+=1
            assert elapsed==0. and not task.relation_runtime.done[0,0]
        contact_steps+=int(contact);max_elapsed=max(max_elapsed,elapsed);max_angle=max(max_angle,angle)
        if args.device=='cpu':
            pairs=task.gym.get_env_rigid_contacts(task.envs[0])
            pair_steps+=int(any((int(c['body0'])==body_index and int(c['body1']) in door_indices) or (int(c['body1'])==body_index and int(c['body0']) in door_indices) for c in pairs))
        trace.append(dict(step=i,angle=angle,foot_height=task._rigid_body_pos[0,0,task._posture_body_ids[2:],2].tolist(),foot_support_force=task._posture_foot_forces[0,0,:,2].tolist(),pelvis_height=task._rigid_body_pos[0,0,task._posture_body_ids[0],2].item(),contact=contact,posture=bool(task.relation_runtime.posture[0,0]),hold_seconds=elapsed,success=bool(task.relation_runtime.done[0,0]),
            hand_distance=(task._rigid_body_pos[0,0,hand]-task.door_handle_position[0,0]).norm().item(),
            hand_force=task._contact_forces[0,0,hand].tolist(),
            leaf_force=task._leaf_contact_forces[0,0,[task._door_leaf_index,task._door_handle_index]].sum(0).tolist()))
        if i%60==0:print('HOLD_TRACE',trace[-1],flush=True)
        if task.relation_runtime.done[0,0]:break
    assert max_angle>=cfg['env']['relationReward']['door']['target_angle'],('no 90-degree push opening',max_angle)
    assert invalid_posture_steps>0,'No unsupported-contact regression samples'
    assert max_legacy_elapsed>=cfg['env']['doorHold']['duration'],'No legacy angle/contact false-success example'
    assert max_elapsed<cfg['env']['doorHold']['duration'] and not task.relation_runtime.done[0,0]
    assert task.door_hinge_angle[1,0].abs()<1e-4 and not task.relation_runtime.done[1,0]
    if args.device=='cpu':assert pair_steps>0,'No real humanoid hand/door contact pair'
    report['checks']['unsupported_hold_rejected']={'max_open_angle_rad':max_angle,'max_valid_hold_seconds':max_elapsed,'max_angle_contact_only_seconds':max_legacy_elapsed,'invalid_posture_contact_steps':invalid_posture_steps,'support_contact_steps':contact_steps,'cpu_hand_door_pair_steps':pair_steps if args.device=='cpu' else None,'standing_5_second_physical_hold_verified':False}
    # Release hands; current success must clear and the passive door must close.
    released=task.door_hinge_angle[0,0].item();separate();step()
    assert not task.relation_runtime.done.any() and not task.relation_runtime.elapsed.any()
    for _ in range(round(10/task.dt)):step()
    assert task.door_hinge_angle[0,0]<.1,task.door_hinge_angle
    report['checks']['release_clears_success_and_closes']={'start_angle_rad':released,'end_angle_rad':task.door_hinge_angle[0,0].item()}
    report['status']='passed';print('DOOR_HOLD_CHECK_OK',json.dumps(report),flush=True)
except BaseException as error:
    import traceback
    report['status']='failed';report['error']=repr(error)
    traceback.print_exc()
    raise
finally:
    (out/'physics_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'contact_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    task.gym.destroy_sim(task.sim)

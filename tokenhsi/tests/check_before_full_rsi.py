"""Finite rollout, exact clip geometry, layout rules and partial-reset isolation."""
import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import isaacgym
import torch
from torch import nn
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.parse_task import parse_task
from utils.stage2_spec import check_checkpoint
from learning.multi_agent.before_network import BeforeBuilder
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from env.tasks.multi_agent.shared_support_scene import scene_infeasible
from check_before import check_at_progress


def inspect(t):
    ids=torch.arange(t.num_envs,device=t.device)
    assert not scene_infeasible(t,ids).any()
    assert not t._full_rsi_invalid.any()
    recorded=torch.zeros_like(t._full_rsi_skill,dtype=torch.bool)
    for skill,(r,a) in t._reset_ref_slots.items():
        recorded[r,a]=True
        lib=t._motion_lib[skill];clips=t._reset_ref_motion_ids[skill];times=t._reset_ref_motion_times[skill]
        reference=lib.get_motion_state_max(clips,times)[0]
        actual=t._kinematic_humanoid_rigid_body_states[r,a,:,:3]
        torch.testing.assert_close(torch.cdist(reference,reference),torch.cdist(actual,actual),atol=2e-4,rtol=2e-4)
        torch.testing.assert_close(reference[...,2],actual[...,2],atol=2e-5,rtol=1e-5)
        if skill in ('pickUp','carryWith','putDown'):
            source=t._scenario_physical_object(r,a)
            pos,_=lib.get_obj_motion_state(clips,times)
            pos[:,2]=torch.maximum(pos[:,2],t._box_size[r,source,2]/2)
            # Actor and payload must receive exactly the same XY/yaw rigid transform.
            torch.testing.assert_close((reference-pos[:,None]).norm(dim=-1),
                (actual-t._box_states[r,source,None,:3]).norm(dim=-1),atol=2e-4,rtol=2e-4)
    assert recorded.all()
    r,b=(t._before_predecessor>=0).nonzero(as_tuple=True);a=t._before_predecessor[r,b]
    modes=t._before_distance_modes[r,b];mix=t.cfg['env']['before']['distance_mixture']
    free=t._full_rsi_skill[r,a]!=7
    goals=t._scenario_goal(r[free],a[free])
    d=(t._tar_pos[r[free],goals,:2]-t._box_states[r[free],a[free],:2]).norm(dim=-1)
    bounds=torch.tensor(mix['leader_goal_distance'],device=t.device)[modes[free]]
    assert ((d>=bounds[:,0]-1e-4)&(d<=bounds[:,1]+1e-4)).all()
    free=(t._full_rsi_skill[r,b]==0)|((t._templates[r,b]==4)&((t._full_rsi_skill[r,b]==5)|(t._full_rsi_skill[r,b]==6)))
    d=(t._humanoid_root_states[r[free],b[free],:2]-t._box_states[r[free],a[free],:2]).norm(dim=-1)
    mode=modes[free];template=t._templates[r[free],b[free]]
    bounds=torch.tensor(mix['follower_distance'],device=t.device)[mode]
    bounds=torch.where((template==4)[:,None],torch.tensor(mix['ontop_follower_distance'],device=t.device)[mode],bounds)
    bounds=torch.where(((template==1)&(mode==0))[:,None],bounds.new_tensor(mix['sit_original_distance']),bounds)
    assert ((d>=bounds[:,0]-1e-4)&(d<=bounds[:,1]+1e-4)).all()
    return dict(skills=torch.bincount(t._full_rsi_skill.flatten(),minlength=8).tolist(),pairs=len(r),progress=check_at_progress(t))


def main():
    torch.set_num_threads(1);args=get_args();args.test=args.eval=True
    cfg,train,_=load_cfg(args);cfg['env']['motion_file']=args.motion_file;cfg['args']=args;set_seed(args.seed)
    task,env=parse_task(args,cfg,train['params']['config'],parse_sim_params(args,cfg,train))
    builder=BeforeBuilder();builder.load(train['params']['network'])
    net=builder.build('amp',actions_num=32,input_shape=(task.get_obs_size(),),amp_input_shape=(task.get_num_amp_obs(),),value_size=1,num_agents=task.num_agents,num_objects=task.num_objects,observation_mode='clean_scene',scene_entity_sizes=task.get_scene_entity_sizes(),scene_kinematic_size=7,scene_arena_scale=task.get_scene_arena_scale(),relation_reward_mode=task._relation_cfg['mode'],relation_graph_spec=task._relation_graph_spec,device=task.device).to(task.device)
    model=nn.Module();model.a2c_network=net
    ck=torch.load(args.checkpoint,map_location=task.device,weights_only=False);check_checkpoint(ck,task.cfg['env']);model.load_state_dict(ck['model'],strict=True);model.eval()
    rms=SceneRunningMeanStd([223,30,1],[4,8,4],normalized_sizes=[223,30,0],kinematic_size=7,extra_passthrough_size=task.get_relation_suffix_size()).to(task.device)
    rms.load_state_dict(ck['running_mean_std'],strict=True);rms.eval();weights={k:v.clone() for k,v in model.state_dict().items()}
    results=[]
    for preset,mode,distance in [('random_scenario','loco',0),('random_scenario','mix',0),('before_stack','mix',2),('all_solo','mix',0),('sit','mix',0),('climb','mix',0),('holding_ontop','mix',0),('random_scenario','loco',2)]:
        task._task_graph_preset=preset;args.before_full_rsi=mode;args.before_follower_near=distance==1;args.before_follower_short=distance==2
        task.reset_buf[:]=1;obs=env.reset();layout=inspect(task)
        if mode=='loco':assert (task._full_rsi_skill==0).all()
        assert torch.isfinite(task._amp_obs_buf).all() and torch.isfinite(task.fetch_amp_obs_demo(128)).all()
        with torch.no_grad():
            for _ in range(16):
                mu,_=net.eval_actor(rms(obs));value=net.eval_critic(rms(obs));obs,reward,done,_=env.step(mu)
                assert all(torch.isfinite(v).all() for v in (mu,value,obs,reward))
        previous={k:getattr(task,k)[1:].clone() for k in ('_full_rsi_skill','_full_rsi_allowed','_before_predecessor','_before_distance_modes','_box_states','_humanoid_root_states')}
        task._reset_envs(torch.tensor([0],device=task.device))
        for k,v in previous.items():assert torch.equal(v,getattr(task,k)[1:]),k
        results.append(dict(preset=preset,mode=mode,distance=distance,finite=True,partial_reset_preserved=True,**layout))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in weights.items())
    report=dict(physics=results,weights_unchanged=True,diagnostics={k:float(v) for k,v in task._consume_sampling_diagnostics().items()})
    p=Path(args.output_path)/'validation.json';p.write_text(json.dumps(report,indent=2)+'\n');print('FULL RSI CHECK COMPLETE',p,flush=True)

if __name__=='__main__':main()

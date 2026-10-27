"""GPU checks for carry50 layout, default loco evaluation and partial reset."""
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
from check_before import check_at_progress
from env.tasks.multi_agent.edge_ontop_reward import vertical_extent


def inspect(task):
    carry=task._before_carry_mask
    eligible=(task._before_predecessor>=0)&(task._templates==4)
    assert not (carry&~eligible).any()
    ground=task._active_physical(torch.arange(task.num_envs,device=task.device)).clone()
    rows,owners=carry.nonzero(as_tuple=True)
    ground[rows,owners]=False
    torch.testing.assert_close(task._box_states[...,2][ground],task._box_size[...,2][ground]/2)
    if len(rows):
        body=task._kinematic_humanoid_rigid_body_states[rows,owners,:,:3]
        box=task._box_states[rows,owners]
        assert (((body[:,task._key_body_ids[:2]].mean(1)-box[:,:3]).square().sum(-1)*-10).exp()>=.9-1e-5).all()
        assert (box[:,2]-vertical_extent(box[:,3:7],task._box_size[rows,owners]/2)>.08-1e-5).all()
        # Compare actor/object relative geometry with the exact selected clip frame.
        es,hs=task._reset_ref_slots['carryWith']
        clips=task._reset_ref_motion_ids['carryWith'];times=task._reset_ref_motion_times['carryWith']
        lib=task._motion_lib['carryWith']
        reference=lib.get_motion_state_max(clips,times)[0]
        source=lib.get_obj_motion_state(clips,times)[0]
        actual=task._kinematic_humanoid_rigid_body_states[es,hs,:,:3]
        payload=task._box_states[es,hs,:3]
        torch.testing.assert_close((actual-payload[:,None]).norm(dim=-1),
                                   (reference-source[:,None]).norm(dim=-1),rtol=1e-4,atol=2e-5)
    mode=task.cfg['args'].before_carry_rsi
    if mode=='loco':assert not carry.any()
    if mode=='carry':
        available=task._size_rsi_cache.carry_available
        assert torch.equal(carry,eligible&available)
    es,followers=(task._before_predecessor>=0).nonzero(as_tuple=True)
    leaders=task._before_predecessor[es,followers]
    modes=task._before_distance_modes[es,followers]
    mix=task.cfg['env']['before']['distance_mixture']
    for distances,bounds in [
        ((task._humanoid_root_states[es,leaders,:2]-task._box_states[es,leaders,:2]).norm(dim=-1),torch.tensor(mix['leader_distance'],device=task.device)[modes]),
        ((task._humanoid_root_states[es,followers,:2]-task._box_states[es,leaders,:2]).norm(dim=-1),torch.where((task._templates[es,followers]==4)[:,None],torch.tensor(mix['ontop_follower_distance'],device=task.device)[modes],torch.where(((task._templates[es,followers]==1)&(modes==0))[:,None],torch.tensor(mix['sit_original_distance'],device=task.device),torch.tensor(mix['follower_distance'],device=task.device)[modes])))]:
        assert ((distances>=bounds[:,0]-2e-5)&(distances<=bounds[:,1]+2e-5)).all()
    top=(task._templates[es,followers]==4)&~carry[es,followers]
    distances=(task._box_states[es[top],followers[top],:2]-task._humanoid_root_states[es[top],followers[top],:2]).norm(dim=-1)
    bounds=torch.tensor(mix['payload_distance'],device=task.device)[modes[top]]
    assert ((distances>=bounds[:,0]-2e-5)&(distances<=bounds[:,1]+2e-5)).all()
    goals=task._scenario_goal(es,leaders)
    d=(task._tar_pos[es,goals,:2]-task._box_states[es,leaders,:2]).norm(dim=-1)
    bounds=torch.tensor(mix['leader_goal_distance'],device=task.device)[modes]
    assert ((d>=bounds[:,0]-2e-5)&(d<=bounds[:,1]+2e-5)).all()
    assert not task._ontop_scene_infeasible(torch.arange(task.num_envs,device=task.device)).any()
    return dict(eligible=int(eligible.sum()),carry=int(carry.sum()),ground=int(ground.sum()),
                modes=task._before_distance_modes[eligible].tolist(),progress=check_at_progress(task))


def main():
    torch.set_num_threads(1)
    args=get_args();args.test=args.eval=True
    cfg,train,_=load_cfg(args);cfg['env']['motion_file']=args.motion_file;cfg['args']=args
    set_seed(42)
    task,env=parse_task(args,cfg,train['params']['config'],parse_sim_params(args,cfg,train))
    builder=BeforeBuilder();builder.load(train['params']['network'])
    net=builder.build('amp',actions_num=32,input_shape=(task.get_obs_size(),),amp_input_shape=(task.get_num_amp_obs(),),value_size=1,num_agents=task.num_agents,num_objects=task.num_objects,observation_mode='clean_scene',scene_entity_sizes=task.get_scene_entity_sizes(),scene_kinematic_size=7,scene_arena_scale=task.get_scene_arena_scale(),relation_reward_mode=task._relation_cfg['mode'],relation_graph_spec=task._relation_graph_spec,device=task.device).to(task.device)
    model=nn.Module();model.a2c_network=net
    checkpoint=torch.load(args.checkpoint,map_location=task.device,weights_only=False);check_checkpoint(checkpoint,task.cfg['env'])
    model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    rms=SceneRunningMeanStd([223,30,1],[4,8,4],normalized_sizes=[223,30,0],kinematic_size=7,extra_passthrough_size=task.get_relation_suffix_size()).to(task.device)
    rms.load_state_dict(checkpoint['running_mean_std'],strict=True);rms.eval()
    before={k:v.clone() for k,v in model.state_dict().items()};results=[]
    with torch.no_grad():
        for preset,mode,distance in [('random_scenario','loco',0),('random_scenario','mix',0),('before_stack','carry',2),('before_stack','carry',1),('all_solo','carry',0),('random_scenario','loco',2)]:
            task._task_graph_preset=preset;args.before_carry_rsi=mode
            args.before_follower_short=distance==2;args.before_follower_near=distance==1
            task.reset_buf[:]=1;obs=env.reset();layout=inspect(task)
            carry=task._before_carry_mask.clone()
            assert torch.isfinite(task.fetch_amp_obs_demo(128)).all()
            for _ in range(16):
                mu,_=net.eval_actor(rms(obs));value=net.eval_critic(rms(obs))
                obs,reward,done,info=env.step(mu)
                assert all(torch.isfinite(x).all() for x in (mu,value,obs,reward))
            untouched=task._root_states.view(task.num_envs,-1,13)[1:].clone()
            untouched_carry=task._before_carry_mask[1:].clone()
            untouched_modes=task._before_distance_modes[1:].clone()
            task._reset_envs(torch.tensor([0],device=task.device))
            assert torch.equal(untouched,task._root_states.view(task.num_envs,-1,13)[1:])
            assert torch.equal(untouched_carry,task._before_carry_mask[1:])
            assert torch.equal(untouched_modes,task._before_distance_modes[1:])
            results.append(dict(preset=preset,mode=mode,distance=distance,finite=True,partial_reset_preserved=True,**layout))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    p=Path(args.output_path)/'validation.json';p.write_text(json.dumps(dict(weights_unchanged=True,physics=results),indent=2)+'\n')
    print('CARRY RSI CHECK COMPLETE',p,flush=True)

if __name__=='__main__':main()

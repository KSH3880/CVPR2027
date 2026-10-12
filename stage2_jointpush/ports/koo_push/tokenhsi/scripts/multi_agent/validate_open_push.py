"""Non-training joint sampler, simulator, network and AMP audit."""
from isaacgym import gymapi, gymtorch
import os
import json
from pathlib import Path
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from utils.unified_training import amp_family_ids

torch.set_num_threads(1)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
args=get_args();assert args.test and args.eval and args.num_envs==2
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True
sim=parse_sim_params(args,cfg,train);sim.physx.num_threads=1
set_seed(42)
task=HumanoidMAOpenPush(cfg,sim,args.physics_engine,args.device,args.device_id,True)
ids=torch.arange(2,device=task.device);out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
report=dict(training=False,device=task.device,checks={},notes='No trained-policy success claim. Contact tensors are spatial/force proxies, not GPU collision pairs.')
try:
    samples=[]
    for _ in range(30):
        task._reset_envs(ids);samples.append(task.relation_runtime.tasks.clone())
        assert torch.isfinite(task.obs_buf).all()
        assert (task._amp_obs_buf[...,-2:].argmax(-1)==task.relation_runtime.tasks[:,:,None]).all()
        assert not task.relation_runtime.done.any()
    samples=torch.stack(samples)
    assert samples.eq(0).any() and samples.eq(1).any()
    report['checks']['sampling']={'push_fraction_per_owner':samples.float().mean(dim=(0,1)).tolist(),'scene_combinations':torch.unique(samples.reshape(-1,2),dim=0).tolist()}
    human=task._humanoid_root_states[1].clone();graph=task.relation_runtime.suffix()[1].clone();task._reset_envs(ids[:1])
    torch.testing.assert_close(task._humanoid_root_states[1],human);torch.testing.assert_close(task.relation_runtime.suffix()[1],graph)
    assert set(task._motion_lib)=={'loco','doorOpen','push'}
    assert task._motion_lib['push'].num_motions()==2
    demo=task.fetch_amp_obs_demo(128)
    assert demo.shape==(128,1310) and torch.isfinite(demo).all()
    labels=amp_family_ids(demo,10,2);assert set(labels.tolist())=={0,1}
    assert (demo.reshape(128,10,131)[...,-2:].argmax(-1)==labels[:,None]).all()
    from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
    builder=AMPMultiAgentBuilder();builder.load(train['params']['network'])
    network=builder.build('amp',actions_num=32,input_shape=(task.get_obs_size(),),amp_input_shape=(task.get_num_amp_obs(),),value_size=1,
        num_agents=2,num_objects=4,humanoid_obs_size=230,object_obs_size=39,goal_obs_size=6,observation_mode='clean_scene',
        scene_entity_sizes=task.get_scene_entity_sizes(),scene_kinematic_size=7,scene_arena_scale=7.,
        relation_reward_mode=cfg['env']['relationReward']['mode'],relation_graph_spec=cfg['env']['relationGraph'],device=task.device).to(task.device)
    with torch.no_grad():
        logits=network.eval_disc(demo)
        actor=network.eval_actor(task.obs_buf)
        critic=network.eval_critic(task.obs_buf)
    assert logits.shape==(128,1) and torch.isfinite(logits).all()
    assert all(torch.isfinite(x).all() for x in actor) and torch.isfinite(critic).all()
    report['checks']['loader_amp_network']={'obs_size':task.get_obs_size(),'amp_shape':list(demo.shape),'actor_shapes':[list(x.shape) for x in actor],'critic_shape':list(critic.shape)}
    report['checks']['partial_reset']=True
    report['checks']['contract']=task.get_open_push_contract()
    # No-contact stability with all four task combinations, stepped without learning.
    for preset in ('open','push','open_push','push_open'):
        task._task_graph_preset=preset;task._reset_envs(ids)
        start_box=task._box_states[:,2:,:3].clone()
        for _ in range(12):
            task.pre_physics_step(torch.zeros(4,32,device=task.device));task._physics_step();task.post_physics_step()
            assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
        assert not task.relation_runtime.done.any()
        assert not task.relation_runtime.push_eligible.any()
        assert (task._box_states[:,2:,:3]-start_box).norm(dim=-1).max()<.03
    report['checks']['four_combinations_nontraining_step']=True
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
finally:
    if task.viewer is not None:task.gym.destroy_viewer(task.viewer)
    task.gym.destroy_sim(task.sim)

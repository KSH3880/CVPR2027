"""CPU simulator check of actual RSI reset, checkpoint policy and partial resets."""
import isaacgym
from pathlib import Path
import json
import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
from learning.multi_agent.amp_network_builder_ma import AMPMultiAgentBuilder
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from utils.stage1_push_spec import check_checkpoint
from utils.push_scene import forward_xy

args=get_args()
assert args.headless and args.device=='cpu' and not args.use_gpu
cfg,train,_=load_cfg(args)
args.eval=False
cfg['args']=args
cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True
set_seed(42);torch.set_num_threads(1)
sim=parse_sim_params(args,cfg,train);sim.physx.num_threads=1
weights=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
check_checkpoint(weights,cfg['env'],for_training=True)
task=HumanoidMAStage1Push(cfg,sim,args.physics_engine,'cpu',0,True)
ids=torch.arange(task.num_envs)
interactions=trials=0;distances=[];angles=[]
for _ in range(40):
    task._reset_envs(ids)
    active=task._rsi_interaction
    assert torch.equal(task._push_plan,task._push_initial_plan)
    torch.testing.assert_close(task._box_states[:,:2,:3][active],task._push_initial_plan[:,:2,:3][active])
    delta=task._tar_pos[...,:2]-task._box_states[:,:2,:2]
    distance=delta.norm(dim=-1)
    if active.any():
        assert distance[active].min()>=2-1e-5 and distance[active].max()<=10+1e-5
        cosine=(delta/distance[...,None]*forward_xy(task._box_states[:,:2,3:7])).sum(-1)
        assert cosine[active].min()>=2**-.5-1e-6
        distances.extend(distance[active].tolist());angles.extend(cosine[active].clamp(-1,1).acos().tolist())
    interactions+=int(active.sum());trials+=active.numel()
    assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
for en in range(task.num_envs):
    for obj in range(4):
        h=task._box_handles[en*4+obj]
        actual_mass = sum(p.mass for p in task.gym.get_actor_rigid_body_properties(task.envs[en],h))
        is_push = bool(task._push_asset_mask[en, obj])
        expected_mass = 20. if is_push else float(task._box_size[en, obj].prod())*100.
        assert abs(actual_mass-expected_mass)<1e-3
        if is_push:
            assert all(abs(p.friction-.4)<1e-6 for p in task.gym.get_actor_rigid_shape_properties(task.envs[en],h))
        if obj >= 2:
            assert .4-1e-6 <= task._box_size[en,obj,0] <= .8+1e-6
            assert .4-1e-6 <= task._box_size[en,obj,1] <= .8+1e-6
            assert .25-1e-6 <= task._box_size[en,obj,2] <= .5+1e-6
saved=[t[1:].clone() for t in (task._box_states,task._humanoid_root_states,task._tar_pos,task._amp_obs_buf)]
task._reset_envs(ids[:1])
for t,s in zip((task._box_states,task._humanoid_root_states,task._tar_pos,task._amp_obs_buf),saved):
    torch.testing.assert_close(t[1:],s)
builder=AMPMultiAgentBuilder();builder.load(train['params']['network'])
network=builder.build('amp',actions_num=task.num_dof,input_shape=(634,),
    amp_input_shape=(task.get_num_amp_obs(),),value_size=1,num_agents=2,num_objects=4,
    humanoid_obs_size=task.get_humanoid_obs_size(),object_obs_size=task.get_object_obs_size(),
    goal_obs_size=task.get_goal_obs_size(),observation_mode='clean_scene',
    scene_entity_sizes=task.get_scene_entity_sizes(),scene_kinematic_size=7,
    scene_arena_scale=task.get_scene_arena_scale(),relation_reward_mode=task._relation_cfg['mode'],
    relation_graph_spec=task._relation_graph_spec,device='cpu')
network.load_state_dict({k[len('a2c_network.'):] if k.startswith('a2c_network.') else k:v for k,v in weights['model'].items()},strict=True)
network.eval()
normalizer=SceneRunningMeanStd(entity_sizes=task.get_scene_entity_sizes(),entity_counts=[2,4,2],
    normalized_sizes=task.get_scene_normalized_entity_sizes(),kinematic_size=task.get_scene_kinematic_size(),
    extra_passthrough_size=task.get_relation_suffix_size())
normalizer.load_state_dict(weights['running_mean_std'],strict=True);normalizer.eval()
with torch.no_grad():
    for _ in range(24):
        actions=network.eval_actor(normalizer(task.obs_buf))[0]
        task.pre_physics_step(actions.reshape(-1,task.num_dof));task._physics_step();task.post_physics_step()
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
report=dict(cpu=True,gpu_executed=False,reset_trials=trials,rsi_samples=interactions,
    rsi_fraction=interactions/trials,distance_min=min(distances),distance_max=max(distances),
    angle_max_degrees=max(angles)*180/torch.pi,boxes_unchanged_by_phase=True,
    partial_reset=True,mass_kg=20,friction=.4,observation=task.get_obs_size(),
    amp_input=task.get_num_amp_obs(),strict_model_and_rms=True,policy_physics_steps=24,
    support_sizes=task._box_size[:,2:].tolist(),physical_boxes=task.num_physical_objects,
    limitations='PUSH-only CPU reset/short policy check; not long-run success or GPU/PPO validation.')
out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
(out/'scene_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));task.gym.destroy_sim(task.sim)

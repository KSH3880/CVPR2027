"""Non-training scene placement audit over all task and RSI combinations."""
from isaacgym import gymapi
import json
from pathlib import Path
import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.open_push_spec import scene_layout
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush

torch.set_num_threads(1)
args=get_args();assert args.test and args.eval and args.num_envs==2
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True
set_seed(42)
task=HumanoidMAOpenPush(cfg,parse_sim_params(args,cfg,train),args.physics_engine,args.device,args.device_id,True)
layout=scene_layout(cfg['env']);ids=torch.arange(2,device=task.device)
task._is_eval=False;task._skill=list(cfg['env']['skill'])
task._skill_init_prob=torch.tensor(cfg['env']['skillInitProb'],device=task.device)
ys=task._box_states.new_tensor([-.5,.5])*layout['lane_spacing']
minimum_separation=float('inf');records=[]
try:
    for preset in ('open','push','open_push','push_open'):
        task._task_graph_preset=preset
        for interaction in (False,True):
            task._joint_rsi_spec['interaction_probability']=1. if interaction else 0.
            for _ in range(10):
                task._reset_envs(ids)
                doors=task._box_states[:,:2,:2];initial=task.relation_runtime.start[:,:,:2]
                torch.testing.assert_close(doors[...,0],torch.full_like(doors[...,0],layout['door_x']))
                torch.testing.assert_close(doors[...,1],ys.expand(2,-1))
                torch.testing.assert_close(initial[...,0],torch.full_like(initial[...,0],layout['push_x']))
                torch.testing.assert_close(initial[...,1],ys.expand(2,-1))
                push=task.relation_runtime.tasks==1
                expected=task.relation_runtime.start.clone();expected[...,1]+=cfg['env']['relationReward']['push']['target_distance']
                torch.testing.assert_close(task._tar_pos[push],expected[push])
                torch.testing.assert_close(task._tar_pos[~push],task.door_handle_position[~push])
                if not interaction:
                    roots=task._humanoid_root_states[...,:3]
                    x=torch.where(push,layout['push_x'],layout['door_x']+.78)
                    torch.testing.assert_close(roots[...,0],x)
                    back=torch.where(push,cfg['env']['pushBox']['size'][1]/2,.1)
                    distance=ys[None]-roots[...,1]-back
                    lo,hi=cfg['env']['doorSpawnDistanceRange']
                    assert ((distance>=lo-1e-5)&(distance<=hi+1e-5)).all()
                pose=task._kinematic_humanoid_rigid_body_states[...,:3]
                minimum_separation=min(minimum_separation,float(torch.cdist(pose[:,0],pose[:,1]).amin()))
                assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
                # Short PD check, never run a learned policy or PPO optimizer.
                for _ in range(12):
                    task.pre_physics_step(torch.zeros(4,32,device=task.device))
                    task._physics_step();task.post_physics_step()
                    minimum_separation=min(minimum_separation,float(torch.cdist(task._rigid_body_pos[:,0],task._rigid_body_pos[:,1]).amin()))
                    assert minimum_separation>.5
                records.append(dict(preset=preset,interaction=interaction))
    human=task._humanoid_root_states[1].clone();boxes=task._box_states[1].clone();goals=task._tar_pos[1].clone()
    task._reset_envs(ids[:1])
    for actual,expected in ((task._humanoid_root_states[1],human),(task._box_states[1],boxes),(task._tar_pos[1],goals)):
        torch.testing.assert_close(actual,expected)
    report=dict(training=False,policy_rollout=False,layout=layout,reset_scenes=len(records)*2,
        presets=['open','push','open_push','push_open'],loco_and_rsi=True,partial_reset=True,
        door_centers=[[layout['door_x'],float(y)] for y in ys],
        box_start_centers=[[layout['push_x'],float(y)] for y in ys],
        push_goal_centers=[[layout['push_x'],float(y)+cfg['env']['relationReward']['push']['target_distance']] for y in ys],
        minimum_humanoid_joint_distance_m=minimum_separation,physics_steps=12*len(records),
        notes='Joint-center separation during sampled resets/short PD only; not an arbitrary-policy collision guarantee.')
    out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True);(out/'layout_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
finally:task.gym.destroy_sim(task.sim)

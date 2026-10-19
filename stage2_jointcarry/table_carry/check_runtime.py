"""GPU6 state/partial-reset/contact check, without starting an optimizer."""
import json,sys
from pathlib import Path
import isaacgym
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tokenhsi'))
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.parse_task import parse_task
args=get_args();cfg,train,_=load_cfg(args);set_seed(42,False)
cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True
cfg['env']['debug']['reward']=False
task,env=parse_task(args,cfg,train,parse_sim_params(args,cfg,train));env.reset()
n=task.num_envs;assert task._joint_env_mask.all();assert task.get_obs_size()==1268
assert (task._joint_reset.skill_index==task._skill.index('carryWith')).all()
initial=task._box_states[:,0,:3].clone();body=task._kinematic_humanoid_rigid_body_states[...,:3]
from env.tasks.multi_agent.joint_carry_reward import anchor_scores,opposite_gate
scores,_=anchor_scores(body[...,task._key_body_ids[[0,1]],:],task._box_states[:,0],task._box_size[:,0],.15,anchor_z=.351)
assert opposite_gate(scores,.9).all()
# Commit the first full reset through PhysX before another indexed setter call.
# Isaac Gym batches indexed root setters until simulate().
task.step(torch.zeros(n*2,task.num_actions,device=task.device))
print('INITIAL_TABLE_BOTTOM', (task._box_states[:,0,2]-.43).tolist(), flush=True)
ids=torch.arange(0,n,2,device=task.device);others=torch.arange(1,n,2,device=task.device)
old=task._humanoid_root_states[others].clone();old_obs=task.obs_buf[others].clone();old_box=task._box_states[others].clone();task._reset_envs(ids)
torch.testing.assert_close(old_box,task._box_states[others],atol=0,rtol=0)
torch.testing.assert_close(old,task._humanoid_root_states[others],atol=0,rtol=0);torch.testing.assert_close(old_obs,task.obs_buf[others],atol=0,rtol=0)
records=[]
for step in range(120):
    task.step(torch.zeros(n*2,task.num_actions,device=task.device))
    assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
    if step in (0,7,29,59,119):
        bodies=task._rigid_body_pos;hands=bodies[...,task._key_body_ids[[0,1]],:]
        scores,_=anchor_scores(hands,task._box_states[:,0],task._box_size[:,0],.15,anchor_z=.351)
        records.append(dict(step=step+1,holding=opposite_gate(scores,.9).float().mean().item(),bottom_min=(task._box_states[:,0,2]-.43).min().item(),bottom_mean=(task._box_states[:,0,2]-.43).mean().item(),root_height_min=task._humanoid_root_states[...,2].min().item()))
report=dict(num_envs=n,finite_steps=120,partial_reset=True,initial_opposite_undergrasp=True,static_pd_probe=records)
Path(args.output_path,'table_physics_check.json').write_text(json.dumps(report,indent=2)+'\n');print('TABLE_PHYSICS_CHECK',json.dumps(report),flush=True)

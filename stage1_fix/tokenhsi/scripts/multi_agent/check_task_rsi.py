import sys,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tokenhsi'))
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.parse_task import parse_task
from utils.torch_utils import quat_rotate
from utils.task_rsi import frame_clearance
args=get_args();cfg,train,_=load_cfg(args);set_seed(42,False)
cfg['env']['motion_file']=args.motion_file
p=parse_sim_params(args,cfg,train);task,env=parse_task(args,cfg,train,p)
ids=torch.arange(task.num_envs,device=task.device,dtype=torch.long)
door_count=early_count=0
selected_count=total=0;max_door=max_push=0;positions=[]
for i in range(12):
 task.reset(ids)
 tasks=task._tasks;sel=task._last_rsi_selected
 positions.append(task._humanoid_root_states[...,:3].clone())
 hands=task._kinematic[:,:,task._key_body_ids[:2],:3]
 door=sel&tasks.bool();push=sel&~tasks.bool()
 if door.any():
  bases=task._root_states[task.doors.actor_indices.long()].view(task.num_envs,task.num_agents,13)[...,:3]
  clearance=frame_clearance(task,task._kinematic[...,:3]-bases[:,:,None,:])
  assert (clearance[door]>=.0299).all(), float(clearance[door].min())
  handles=task._door_geometry()[4]
  distance=torch.linalg.vector_norm(hands-handles[:,:,None,:],dim=-1).min(-1).values
  max_door=max(max_door,float(distance[door].max()))
  assert torch.all(distance[door]<.12)
  angle=task._door_state_view[...,0]
  assert (angle[door]>0).all() and (angle[door]<math.radians(80)).all()
  torch.testing.assert_close(task._best_angle[door],angle[door])
  early_count+=int((angle[door]<=math.radians(25)).sum());door_count+=int(door.sum())
  requested=task._last_rsi_door_early & door
  assert (angle[requested]<=math.radians(25)).all()
 if push.any():
  boxes=task._box_states;q=boxes[...,3:7].clone();q[...,:3]*=-1
  local=quat_rotate(q[:,:,None,:].expand(-1,-1,2,-1).reshape(-1,4),(hands-boxes[:,:,None,:3]).reshape(-1,3)).view(task.num_envs,2,2,3)
  gap=(-task._box_size[0]/2-local[...,0]).min(-1).values
  assert (gap[push]>.025).all() and (gap[push]<.035).all()
  assert (hands[...,2][push]>0).all() and (hands[...,2][push]<1.1).all()
  rootlocal=quat_rotate(q.reshape(-1,4),(task._humanoid_root_states[...,:3]-boxes[...,:3]).reshape(-1,3)).view(task.num_envs,2,3)
  assert (rootlocal[...,0][push]<-task._box_size[0]/2-.2).all()
  max_push=max(max_push,float(gap[push].max()))
 assert (task._elapsed==0).all() and not task._done_task.any()
 assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
 task._compute_reward(None)
 assert (task.extras['reward_terms'][:,[0,2]]<1e-4).all(), (i,task.extras['reward_terms'][:,[0,2]].max().item()) # tolerate float roundoff
 selected_count+=int(sel.sum());total+=sel.numel()
 task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True);task._refresh_sim_tensors()
 assert torch.isfinite(task._root_states).all()
assert not torch.equal(positions[0],positions[1])
# Reset an isolated environment without touching the others' input state.
before=task._humanoid_root_states[1:].clone();task.reset(ids[:1])
torch.testing.assert_close(before,task._humanoid_root_states[1:])
task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True)
assert early_count/door_count>.65, (early_count,door_count)
report=dict(door_rsi_count=door_count,small_angle_fraction=early_count/door_count,sample_count=total,rsi_fraction=selected_count/total,max_door_hand_distance_m=max_door,push_surface_gap_m=max_push,frame_clearance_m=.03,finite=True,partial_reset=True,initial_progress_reward_zero=True)
out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True);(out/'rsi_report.json').write_text(json.dumps(report,indent=2))
print('PASS',report,flush=True)
task.gym.destroy_sim(task.sim)

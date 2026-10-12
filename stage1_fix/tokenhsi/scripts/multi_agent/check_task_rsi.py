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
physical_gaps=[]
initial_sizes=task._box_sizes.clone()
for environment,handles in zip(task.envs,task._box_handles):
 for handle in handles:
  mass=sum(body.mass for body in task.gym.get_actor_rigid_body_properties(environment,handle))
  assert abs(mass-task._box_mass)<1e-3, (mass,task._box_mass)

door_count=early_count=0
selected_count=total=0;max_door=max_push=0;positions=[]
for i in range(12):
 task.reset(ids)
 tasks=task._tasks;sel=task._last_rsi_selected
 torch.testing.assert_close(task._box_sizes,initial_sizes)
 if task._interaction['push']['box'].get('size_range') is not None:
  distances=(task._targets[...,:2]-task._box_states[...,:2]).norm(dim=-1)[~tasks.bool()]
  assert ((distances>=1.9999)&(distances<=4.0001)).all(),distances

 positions.append(task._humanoid_root_states[...,:3].clone())
 hands=task._kinematic[:,:,task._key_body_ids[:2],:3]
 door=sel&tasks.bool();push=sel&~tasks.bool()
 if door.any():
  if task._door_rsi_hand is not None:
   assert (task._last_rsi_contact_hand[door]==task._door_rsi_hand).all()
   assert torch.equal(door,tasks.bool())
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
  if 'enter_push_surface_distance' in task._interaction['amp'].get('approach_loco',{}):
   assert (task._family()[push]==0).all()
  boxes=task._box_states;q=boxes[...,3:7].clone();q[...,:3]*=-1
  local=quat_rotate(q[:,:,None,:].expand(-1,-1,2,-1).reshape(-1,4),(hands-boxes[:,:,None,:3]).reshape(-1,3)).view(task.num_envs,2,2,3)
  gap=-task._box_sizes[...,0,None]/2-local[...,0]-task._interaction['task_rsi']['push_hand_radius']
  expected=task._interaction['task_rsi']['push_surface_gap']
  assert ((gap[push]-expected).abs()<1e-4).all(), gap[push]
  assert (hands[...,2][push]>0).all() and (hands[...,2][push]<task._box_sizes[...,2,None].expand(-1,-1,2)[push]).all()
  rootlocal=quat_rotate(q.reshape(-1,4),(task._humanoid_root_states[...,:3]-boxes[...,:3]).reshape(-1,3)).view(task.num_envs,2,3)
  assert (rootlocal[...,0][push]<-task._box_sizes[...,0][push]/2-.2).all()
  max_push=max(max_push,float(gap[push].max()))
 assert (task._elapsed==0).all() and not task._done_task.any()
 assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
 task._compute_reward(None)
 progress_ids=[0,2]+([12] if 'relation_rewards' in task._interaction['push'] else [])
 assert (task.extras['reward_terms'][:,progress_ids]<3e-4).all(), (i,task.extras['reward_terms'][:,[0,2]].max().item()) # tolerate float roundoff
 selected_count+=int(sel.sum());total+=sel.numel()
 task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True);task._refresh_sim_tensors()
 assert torch.isfinite(task._root_states).all()
 if push.any():
  boxes=task._box_states;q=boxes[...,3:7].clone();q[...,:3]*=-1
  actual_hands=task._rigid_body_pos[:,:,task._key_body_ids[:2]]
  actual_local=quat_rotate(q[:,:,None,:].expand(-1,-1,2,-1).reshape(-1,4),(actual_hands-boxes[:,:,None,:3]).reshape(-1,3)).view(task.num_envs,2,2,3)
  actual_gap=-task._box_sizes[...,0,None]/2-actual_local[...,0]-task._interaction['task_rsi']['push_hand_radius']
  physical_gaps.append(actual_gap[push].flatten().clone())

assert not torch.equal(positions[0],positions[1])
# Reset an isolated environment without touching the others' input state.
before=task._humanoid_root_states[1:].clone();task.reset(ids[:1])
torch.testing.assert_close(before,task._humanoid_root_states[1:])
task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True)
if task._door_rsi_hand is None or task._task_rsi_pools['doorOpen'][3].any():
 assert early_count/door_count>.65, (early_count,door_count)
physical=torch.cat(physical_gaps) if physical_gaps else torch.zeros(1,device=task.device)
if physical_gaps:assert physical.abs().max()<.003, physical.abs().max().item()
report=dict(post_step_push_gap_p95_m=float(torch.quantile(physical,.95)),post_step_push_gap_min_m=float(physical.min()),post_step_push_gap_max_m=float(physical.max()),door_rsi_count=door_count,small_angle_fraction=early_count/door_count,sample_count=total,rsi_fraction=selected_count/total,max_door_hand_distance_m=max_door,push_surface_gap_m=max_push,frame_clearance_m=.03,finite=True,partial_reset=True,initial_progress_reward_zero=True)
report['push_rsi_measured']=bool(physical_gaps)
if not physical_gaps:
 for key in ('post_step_push_gap_p95_m','post_step_push_gap_min_m','post_step_push_gap_max_m','push_surface_gap_m'):report[key]=None
report['box_side_min_m']=float(task._box_sizes.min());report['box_side_max_m']=float(task._box_sizes.max())
if 'enter_push_surface_distance' in task._interaction['amp'].get('approach_loco',{}):
 push_slots=task._tasks.eq(0)
 assert push_slots.any()
 task._contact_forces.zero_();task._all_contacts.zero_()
 task._humanoid_root_states[...,:2]=task._box_states[...,:2]+20
 task._compute_reward(None)
 assert (task._family()[push_slots]==5).all()
 local=torch.zeros_like(task._humanoid_root_states[...,:3]);local[...,0]=task._box_sizes[...,0]/2+.5
 task._humanoid_root_states[...,:3]=task._box_states[...,:3]+quat_rotate(task._box_states[...,3:7].reshape(-1,4),local.reshape(-1,3)).view_as(local)
 task._compute_reward(None)
 assert (task._family()[push_slots]==0).all()
 task._compute_amp_observations()
 assert (task._amp_obs_buf[...,-task._amp_label_count:].argmax(-1)[push_slots]==0).all()
 report['push_loco_transition']=True
out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True);(out/'rsi_report.json').write_text(json.dumps(report,indent=2))
print('PASS',report,flush=True)
task.gym.destroy_sim(task.sim)

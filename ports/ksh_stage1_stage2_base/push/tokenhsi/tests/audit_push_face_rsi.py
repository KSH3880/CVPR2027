from isaacgym import gymapi
import sys,json,math,subprocess,itertools
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
from utils.stage1_push import sample_boxes_and_goals
from utils.push_point import world_points, common_face_xy
from utils.door_reference import rotate
CFG_PATH=Path(sys.argv[sys.argv.index('--cfg_env')+1])
import yaml
CFG=yaml.safe_load(CFG_PATH.read_text())
with np.load(CFG['env']['pushTask']['rsi_cache_file'],allow_pickle=False) as data:
 SIZES=data['sizes'].copy();VALID=data['valid'].copy();OFFSETS=data['clip_offsets'].copy()
def sizes(t):
 s=torch.tensor(SIZES,device=t.device,dtype=torch.float32)
 return s[torch.arange(t.num_envs,device=t.device)%len(s)][:,None].expand(-1,4,-1).clone()
HumanoidMAStage1Push._sample_push_box_sizes=sizes
@torch.no_grad()
def probe(t):
 out=Path(t.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
 ids=torch.arange(t.num_envs,device=t.device);t._reset_envs(ids);lib=t._motion_lib['push']
 names=list(lib._motions[0].skeleton_tree.node_names);parents=lib._motions[0].skeleton_tree.parent_indices.cpu().tolist();head=names.index('head');geometry=t._push_reset_geometry()
 per_profile=[]
 for profile in range(len(SIZES)):
  frames=np.flatnonzero(VALID[profile]);clips=np.searchsorted(OFFSETS[1:],frames,side='right');local=frames-OFFSETS[clips]
  per_profile.append(list(zip(clips.tolist(),local.tolist())))
 slots=[[(e,a) for e in range(t.num_envs) for a in range(2) if e%len(SIZES)==p] for p in range(len(SIZES))]
 metrics=[];selected={};timeline=[];violations=dict(body_overlap=0,wrong_goal_side=0,outside_face_projection=0);max_dof_error=0.
 def record(e,a,label,clip=-1,time=0.):
  body=t._kinematic_humanoid_rigid_body_states[e,a].cpu().numpy().copy();box=t._box_states[e,a].cpu().numpy().copy();goal=t._tar_pos[e,a].cpu().numpy().copy()
  origin=box[:3].copy();origin[2]=0;body[:,:3]-=origin;box[:3]-=origin;goal-=origin
  return dict(label=label,body=body.tolist(),box=box.tolist(),size=t._box_size[e,a].cpu().tolist(),goal=goal.tolist(),clip=int(clip),time=float(time))
 batches=max(math.ceil(len(rows)/len(s)) for rows,s in zip(per_profile,slots))
 for batch in range(batches):
  rr=[];aa=[];cc=[];ff=[];pp=[]
  for p,(cases,available) in enumerate(zip(per_profile,slots)):
   segment=cases[batch*len(available):(batch+1)*len(available)]
   for (clip,frame),(e,a) in zip(segment,available):rr.append(e);aa.append(a);cc.append(clip);ff.append(frame);pp.append(p)
  r=torch.tensor(rr,device=t.device);a=torch.tensor(aa,device=t.device);clips=torch.tensor(cc,device=t.device,dtype=torch.long)
  times=torch.tensor(ff,device=t.device)*lib._motion_dt[clips];phase=times/lib._motion_lengths[clips]
  states,goals,forward,distances=sample_boxes_and_goals(t.num_envs,t.device,t._env_origins,t.cfg['env'],t._box_size)
  sample=dict(motion_id=clips,time=times,phase=phase,box_progress=torch.minimum(phase*t.cfg['env']['pushTask']['rsi_box_progress_scale'],1-(t.cfg['env']['pushTask']['minimum_rsi_remaining_m']+.001)/distances[r,a]),late_requested=torch.zeros(len(r),device=t.device,dtype=torch.bool))
  t._box_states.copy_(states);t._tar_pos.copy_(goals)
  allr=ids.repeat_interleave(2);alla=torch.arange(2,device=t.device).repeat(t.num_envs)
  t._reset_ref_slots['loco']=(allr,alla);t._reset_ref_motion_ids['loco']=torch.zeros_like(allr);t._reset_ref_motion_times['loco']=torch.zeros(len(allr),device=t.device)
  t._apply_push_rsi(r,a,sample,states,ids,goals)
  pose=t._kinematic_humanoid_rigid_body_states[r,a];box=t._box_states[r,a];size=t._box_size[r,a];mid=pose[:,t._key_body_ids[:2],:3].mean(1)
  inv=box[:,3:7].clone();inv[:,:3]*=-1;local=rotate(inv,mid-box[:,:3]);plane_gap=-local[:,1]-size[:,1]/2
  hand_local=rotate(inv[:,None].expand(-1,2,-1),pose[:,t._key_body_ids[:2],:3]-box[:,None,:3])
  projection=((hand_local[...,0].abs()<=size[:,None,0]/2)&(hand_local[...,2].abs()<=size[:,None,2]/2)).all(-1)
  head_center=pose[:,head,:3]+rotate(pose[:,head,3:7],pose.new_tensor([0.,0.,.175]).expand(len(r),3))
  hl=rotate(inv,head_center-box[:,:3]);head_gap=(hl.abs()-size/2).clamp_min(0).norm(dim=-1)-.095
  invalid=geometry.invalid(pose[...,:3],pose[...,3:7],box,size)
  target=world_points(box,t._push_local[r,a]);xy=(mid[:,:2]-target[:,:2]).norm(dim=-1)
  side=((target-box[:,:3])[:,:2]*(goals[r,a]-box[:,:3])[:,:2]).sum(-1)>=0
  violations['body_overlap']+=int(invalid.sum());violations['wrong_goal_side']+=int(side.sum());violations['outside_face_projection']+=int((~projection).sum())
  ref=lib.get_motion_state(clips,times);max_dof_error=max(max_dof_error,float((t._dof_pos[r,a]-ref[2]).abs().max()))
  face_dist,_,face_side=common_face_xy(pose[:,None,t._key_body_ids[:2],:3],box[:,None],size[:,None],goals[r,a,None])
  d=face_dist[:,0].amax(-1)
  data=torch.stack((clips.float(),times,plane_gap,d,mid[:,2],head_gap),-1).cpu().numpy()
  for k,row in enumerate(data):
   metrics.append([pp[k],*row.tolist()])
   if pp[k]==2:timeline.append(record(int(r[k]),int(a[k]),'selected_push',int(clips[k]),float(times[k])))
   for key,value,better in [('largest_hand_gap',float(xy[k]),lambda x,y:x>y),('smallest_head_gap',float(head_gap[k]),lambda x,y:x<y),('typical_push',abs(float(xy[k])-.08)+abs(float(mid[k,2])-.95),lambda x,y:x<y)]:
    if key not in selected or better(value,selected[key][0]):selected[key]=(value,record(int(r[k]),int(a[k]),key,int(clips[k]),float(times[k])))
  if batch%20==0:print('AUDIT_BATCH',batch+1,batches,flush=True)
 loco_best={};loco_dist=[];loco_bad=0
 t.cfg['env']['eval']['skill']=['loco'];t.cfg['env']['eval']['skillInitProb']=[1.]
 for iteration in range(24):
  t._reset_envs(ids);assert not t._push_rsi_interaction.any()
  pose=t._kinematic_humanoid_rigid_body_states[:,:,:,:3];dist=(pose[:,:,0,:2]-t._box_states[:,:2,:2]).norm(dim=-1)
  flat=dist.flatten();loco_dist.extend(flat.cpu().tolist())
  kin=t._kinematic_humanoid_rigid_body_states.reshape(-1,t.num_bodies,13)
  loco_bad+=int(geometry.invalid(kin[...,:3],kin[...,3:7],t._box_states[:,:2].reshape(-1,13),t._box_size[:,:2].reshape(-1,3)).sum())
  for goal,label in [(1.,'loco_near'),(3.,'loco_middle'),(5.,'loco_far')]:
   score,index=(flat-goal).abs().min(0);value=float(score);e,a=divmod(int(index),2)
   if label not in loco_best or value<loco_best[label][0]:loco_best[label]=(value,record(e,a,label))
 records=[selected[k][1] for k in ['typical_push','largest_hand_gap','smallest_head_gap']]+[loco_best[k][1] for k in ['loco_near','loco_middle','loco_far']]
 arr=np.array(metrics);assert len(arr)==int(VALID.sum())
 def stats(x):return dict(min=float(np.min(x)),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(np.max(x)))
 xy=arr[:,4];z=arr[:,5]
 report=dict(policy_actions=0,physics_steps=0,push_valid_size_frames=len(arr),sizes=SIZES.tolist(),original_motion_box_files=0,violations=violations,max_original_dof_error=max_dof_error,push_hand_midpoint_to_face_gap_m=stats(arr[:,3]),push_max_hand_common_face_xy_distance_m=stats(xy),push_hand_midpoint_z_m=stats(z),head_sphere_to_box_gap_m=stats(arr[:,6]),initial_common_face_state_above09=int((np.exp(-10*xy**2)>.9).sum()),loco_samples=len(loco_dist),loco_center_distance_m=stats(loco_dist),loco_invalid_body_overlap=loco_bad,method='All valid current RSI cache size-frame pairs, passed through actual _apply_push_rsi without policy/physics. Loco uses actual reset path; 7 box sizes instantiated in a separate evaluation.')
 (out/'audit.json').write_text(json.dumps(report,indent=2));(out/'scenes.json').write_text(json.dumps(dict(names=names,parents=parents,scenes=records),indent=2));np.savez_compressed(out/'all_push_metrics.npz',metrics=arr,columns=np.array(['profile','clip','time','hand_plane_gap','max_hand_common_face_distance','hand_z','head_gap']))
 (out/'timeline.json').write_text(json.dumps(dict(names=names,parents=parents,scenes=timeline)))
 print('AUDIT',json.dumps(report),flush=True)
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
args=get_args();cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file
set_seed(42);torch.set_num_threads(1)
task=HumanoidMAStage1Push(cfg,parse_sim_params(args,cfg,train),args.physics_engine,args.device,args.device_id,True)
try:probe(task)
finally:task.gym.destroy_sim(task.sim)

"""Recorded 2H/4H failures and timing diagnostics; no causal attribution."""
from pathlib import Path
import json,collections,statistics,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
GPU=ROOT/'teacher_runtime/output_etc/check/eval_protocol_v1_legacy_20261111_165027/eval'
API=ROOT/'output_etc/check/api_no_unnecessary_wait_20261111_151400/api'
OUT=ROOT/'output_etc/check/small_scene_analysis_20261112'

def desc(v):return dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v))) if v else dict(n=0)
def main():
 OUT.mkdir(exist_ok=True);report={'mixed':{},'coupled_table2':{},'limitations':['Timing error uses only reached predecessor events; unreached events are censored, not zero.','Estimated follower event before measured predecessor success is a prediction risk, not measured interaction ordering.','Hand-nearness readiness is a diagnostic proxy; LLM approach event has no implemented exact detector.','Counts are pooled; official report rates are seed means.']}
 for n in [2,4]:
  count=collections.Counter();bytask={};errors=[];pred=[];actual=[];rows=[]
  for seed in [1,6,7]:
   path=GPU/f'table1/mixed_{n}h/general_seed{seed}';z=np.load(path/'trace.npz');boxes=z['boxes'];valid=z['valid'];root=z['roots'];dt=json.loads((path/'result.json').read_text())['control_dt']
   for e in range(len(z['predecessor'])):
    req=json.loads((API/f'table1/mixed_{n}h/general_seed{seed}/env_{e:04}/request.json').read_text());ans=json.loads((API/f'table1/mixed_{n}h/general_seed{seed}/env_{e:04}/response.raw.txt').read_text());dur={p['event_id']:p['estimated_elapsed_time_s'] for p in ans['predictions']};rel={p['agent_id']:p['release_time_s'] for p in ans['schedule']};constraints=req['current_scene']['coordination_constraints']
    for follower,leader in enumerate(z['predecessor'][e]):
     if leader<0:continue
     L=f'A{leader+1:02}';F=f'A{follower+1:02}';c=next(c for c in constraints if c['type']=='enabling' and c['enabling_event_id'].startswith(L+'_'))
     p=dur[c['enabling_event_id']];q=dur[c['dependent_event_id']];pred.append(p)
     ls=bool(z['ever'][e,leader,1]);fs=bool(z['ever'][e,follower,1]);pair=bool(z['simultaneous'][e,follower,1]);kind={0:'SIT',1:'CLIMB',3:'ON_TOP'}[int(z['packet'][e,follower,1])]
     row=dict(seed=seed,env=e,leader=L,follower=F,task=kind,leader_success=ls,follower_success=fs,pair_success=pair)
     key='pair_success' if pair else ('leader_never_success' if not ls else ('follower_never_success' if not fs else 'both_success_without_overlap'))
     count['pairs']+=1;count[key]+=1;t=bytask.setdefault(kind,collections.Counter());t['pairs']+=1;t[key]+=1
     count['leader_fall']+=int(z['fall_step'][e,leader]>=0);count['follower_fall']+=int(z['fall_step'][e,follower]>=0)
     if ls:
      good=(np.linalg.norm(boxes[:,e,leader,:3].astype(float)-z['goals'][e,leader],axis=1)<=.2)&valid[:,e,leader];assert good.any();time=(int(np.argmax(good))+1)*dt;actual.append(time);errors.append(time-p);count['predicted_p_too_short']+=time>p;count['predicted_follower_event_before_actual_leader']+=rel[F]+q<=time
      row.update(predicted_p=p,actual_p=time,predicted_follower_event=rel[F]+q)
     else:
      end=int(z['fall_step'][e,leader]);end=end if end>=0 else len(root);b=boxes[:end,e,leader,:3];r=root[:end,e,leader];dd=np.linalg.norm(r[:,:2]-b[:,:2],axis=1);gain=float(b[:,2].max()-b[0,2]);away=dd[-1]-dd[0]>=2 and np.linalg.norm(r[:,:2]-r[0,:2],axis=1).max()>=3
      count['failed_leader_no_lift_and_away']+=int(gain<.15 and away);count['failed_leader_fell']+=int(z['fall_step'][e,leader]>=0)
     rows.append(row)
  report['mixed'][str(n)]={'counts':dict(count),'by_follower_task':{k:dict(v) for k,v in bytask.items()},'predicted_p_all':desc(pred),'actual_p_successful':desc(actual),'actual_minus_predicted_p_successful':desc(errors),'rows':rows}
 sys.path[:0]=[str(ROOT/'teacher_runtime/tokenhsi'),str(ROOT/'teacher_runtime/eval')];from table2_metrics import surface_distance
 import torch;torch.set_num_threads(1)
 for n in [2,4]:
  counts=collections.Counter();spreads=[];success_spreads=[];firsts=[]
  for seed in [1,6,7]:
   p=GPU/f'table2/coupled_{n}/general_seed{seed}';z=np.load(p/'trace.npz');a=np.load(p/'cooperation.npz');hands=a['hands'];pose=a['physical_boxes'];size=a['physical_sizes'];initial=torch.load(p/'initial.pt',map_location='cpu',weights_only=False)['shared_centers'][:,0].numpy();T,B=pose.shape[:2]
   distance=surface_distance(torch.from_numpy(hands).reshape(T*B,n,2,3),torch.from_numpy(pose).reshape(-1,7),torch.from_numpy(np.broadcast_to(size[None],(T,B,3)).copy()).reshape(-1,3)).numpy().reshape(T,B,n,2)
   near=(distance<.12).all(-1)&z['valid'];first_success=np.load(p/'table2_trials.npz')['first_success']
   for e in range(B):
    success=first_success[e]>=0;counts['trials']+=1;counts['success']+=success
    if success:
     end=int(first_success[e]);arrived=near[:end,e].any(0);counts['success_all_reached_before_completion']+=int(arrived.all());counts['success_any_fall']+=int((z['fall_step'][e]>=0).any())
     if arrived.all():
      f=near[:end,e].argmax(0);success_spreads.append(float((max(f)-min(f))/30));moved=np.linalg.norm(pose[:end,e,:2].astype(float)-initial[e,:2],axis=1)>=.2;counts['success_box_started_before_last_arrival']+=int(moved.any() and np.argmax(moved)<max(f))
     continue
    counts['failures']+=1;fall=z['fall_step'][e]>=0;counts['failed_any_fall']+=int(fall.any());arrived=near[:,e].any(0);counts['failed_not_all_reached_hand_nearness']+=int(not arrived.all());counts['failed_all_reached_hand_nearness']+=int(arrived.all());counts['failed_ever_all_near_simultaneously']+=int(near[:,e].all(-1).any())
    if arrived.all():
     f=near[:,e].argmax(0);spreads.append(float((max(f)-min(f))/30));moved=np.linalg.norm(pose[:,e,:2].astype(float)-initial[e,:2],axis=1)>=.2
     counts['failed_box_started_before_last_arrival']+=int(moved.any() and np.argmax(moved)<max(f))
  report['coupled_table2'][str(n)]={'counts':{k:int(v) for k,v in counts.items()},'first_hand_nearness_spread_failed_all_arrived_s':desc(spreads),'first_hand_nearness_spread_successful_all_arrived_s':desc(success_spreads)}
 (OUT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:{n:{a:b for a,b in d.items() if a!='rows'} for n,d in v.items()} for k,v in report.items() if isinstance(v,dict)},indent=2))
if __name__=='__main__':main()

"""Trajectory-based failure phenotypes; proximity is not confirmed contact."""
from pathlib import Path
import json,csv,collections
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/hwanhee/ksh/stage1_2_base_teacher/output_etc/check/sct_fsm_eval_v2_20261109/table1/mixed_16h')
OUT=ROOT/'output_etc/check/fsm_predecessor_failure_analysis_20261111'
PARAMS=dict(lift_height_gain_m=.15,root_near_payload_xy_m=1.2,runaway_distance_increase_m=2.,runaway_root_displacement_m=3.,human_proximity_xy_m=.8,prefall_window_s=.5)

def main():
 OUT.mkdir(parents=True,exist_ok=False);rows=[];all_leaders=0;successes=0
 for seed in [1,3,6]:
  folder=SOURCE/f'general_seed{seed}';z=np.load(folder/'trace.npz');root=z['roots'];box=z['boxes'];parent=z['predecessor'];ever=z['ever'];falls=z['fall_step'];groups=z['groups'];valid=z['valid'];dt=json.loads((folder/'result.json').read_text())['control_dt']
  for env in range(root.shape[1]):
   for lead in sorted(set(int(i) for i in parent[env] if i>=0)):
    all_leaders+=1
    if ever[env,lead,1]:successes+=1;continue
    fall=int(falls[env,lead]);end=fall if fall>=0 else len(root)
    r=root[:end,env,lead];p=box[:end,env,lead,:3];dist=np.linalg.norm(r[:,:2]-p[:,:2],axis=1)
    height_gain=float(p[:,2].max()-p[0,2]);lift=height_gain>=PARAMS['lift_height_gain_m'];near=float(dist.min())<=PARAMS['root_near_payload_xy_m'];drift=float(np.linalg.norm(r[:,:2]-r[0,:2],axis=1).max());away=bool(float(dist[-1]-dist[0])>=2 and drift>=3)
    others=[i for i in range(16) if i!=lead];nearest=None;near_actor=None;near_time=None;human_close=None
    if fall>=0:
     start=max(0,end-int(round(.5/dt))-1);d=np.linalg.norm(np.take(root[start:end,env,:,:2],others,axis=1)-r[start:end,None,:2],axis=-1)
     t,a=np.unravel_index(np.argmin(d),d.shape);nearest=float(d[t,a]);near_actor=int(others[a]);near_time=(start+t+1)*dt;human_close=nearest<.8
    if fall>=0:category='fall_with_nearby_human' if human_close else 'fall_without_nearby_human'
    elif not lift and away:category='no_lift_and_moved_away'
    elif not lift and near:category='approached_without_lift'
    elif not lift:category='no_near_approach_or_lift'
    else:category='lift_evidence_but_no_goal_success'
    followers=np.flatnonzero(parent[env]==lead).tolist()
    row=dict(seed=seed,environment=env,predecessor=f'A{lead+1:02}',followers=','.join(f'A{i+1:02}' for i in followers),category=category,fall_frame=fall,fall_time_s=fall*dt if fall>=0 else None,lift_evidence=bool(lift),near_payload=bool(near),moved_away=away,max_payload_height_gain_m=height_gain,min_root_payload_xy_m=float(dist.min()),initial_root_payload_xy_m=float(dist[0]),last_alive_root_payload_xy_m=float(dist[-1]),max_root_xy_displacement_m=drift,min_goal_xyz_distance_alive_m=float(np.linalg.norm(p-z['goals'][env,lead],axis=1).min()),nearest_human_prefall_xy_m=nearest,nearest_human=f'A{near_actor+1:02}' if near_actor is not None else None,nearest_is_follower=near_actor in followers if near_actor is not None else None,nearest_same_group=bool(groups[env,near_actor]==groups[env,lead]) if near_actor is not None else None,nearest_time_s=near_time)
    rows.append(row)
  print('processed seed',seed,flush=True)
 counts=collections.Counter(r['category'] for r in rows);report=dict(source=str(SOURCE),agents=16,envs_per_seed=512,seeds=[1,3,6],predecessors=all_leaders,successes=successes,failures=len(rows),heuristic_parameters=PARAMS,categories={k:dict(count=v,percent=100*v/len(rows)) for k,v in counts.items()},overlapping_flags=dict(failed_and_fell=sum(r['fall_frame']>=0 for r in rows),no_lift_evidence=sum(not r['lift_evidence'] for r in rows),no_lift_and_moved_away=sum(not r['lift_evidence'] and r['moved_away'] for r in rows),fall_nearest_is_follower=sum(r['category']=='fall_with_nearby_human' and r['nearest_is_follower'] for r in rows)),limitations=['No hand/contact force traces: lifting is motion evidence, not confirmed grasp.','Root XY proximity within 0.8m during 0.5s before fall is a heuristic, not proof of collision or causation.','No nearby human does not exclude box, ground or self collision.','Only failed predecessors are classified; successful-then-fallen agents excluded.'],examples={k:[r for r in rows if r['category']==k][:3] for k in counts})
 (OUT/'summary.json').write_text(json.dumps(report,indent=2))
 with (OUT/'failures.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({k:report[k] for k in ['predecessors','successes','failures','categories','overlapping_flags']},indent=2))
if __name__=='__main__':main()

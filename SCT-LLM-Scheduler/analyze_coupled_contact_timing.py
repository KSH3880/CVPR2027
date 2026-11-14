"""Recorded-trial association analysis; no simulation or API calls."""
from pathlib import Path
import json,csv
import numpy as np
import torch
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'teacher_runtime/output_etc/check/eval_protocol_v1_legacy_20261111_165027/eval/table2'
OUT=ROOT/'output_etc/check/coupled_contact_timing_20261112';OUT.mkdir(exist_ok=True)
rows=[]
for n in (2,3,4):
 for seed in (1,6,7):
  p=BASE/f'coupled_{n}/general_seed{seed}'
  c=np.load(p/'cooperation.npz');tr=np.load(p/'trace.npz');initial=torch.load(p/'initial.pt',map_location='cpu',weights_only=False)
  first=np.load(p/'protocol_v1_trials.npz')['first_success']
  pose=c['physical_boxes'];hands=c['hands'];sizes=c['physical_sizes'];valid=tr['valid']
  delta=hands-pose[:,:,:3][:,:,None,None,:];xyz=-pose[:,:,3:6][:,:,None,None,:];w=pose[:,:,6][:,:,None,None,None]
  uv=np.cross(xyz,delta);local=delta+2*(w*uv+np.cross(xyz,uv))
  excess=np.abs(local)-sizes[None,:,None,None,:]/2
  distance=np.abs(np.linalg.norm(np.maximum(excess,0),axis=-1)+np.minimum(excess.max(-1),0))
  touch=(distance<.12).all(-1)&valid
  for i in range(64):
   start=initial['shared_centers'][i,0].numpy();goal=initial['shared_center_goals'][i,0].numpy();forward=goal[:2]-start[:2];forward/=np.linalg.norm(forward);side=np.array([-forward[1],forward[0]])
   end=int(first[i]) if first[i]>=1 else 900
   contact=[]
   for a in range(n):
    ix=np.flatnonzero(touch[:end,i,a]);contact.append(int(ix[0]) if len(ix) else None)
   complete=all(t is not None for t in contact)
   err=pose[:,i,:3]-goal;closest=int(np.linalg.norm(err,axis=-1).argmin());last=err[-1]
   # Positive longitudinal error means overshoot, negative means short of goal.
   yaw=lambda q:np.arctan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2))
   yawdelta=yaw(pose[-1,i,3:7])-yaw(pose[0,i,3:7]);yawdelta=np.arctan2(np.sin(yawdelta),np.cos(yawdelta))
   gap=(max(contact)-min(contact))/30 if complete else None
   row=dict(n=n,seed=seed,env=i,success=bool(first[i]>=1),all_contact_before_success_or_end=complete,contact_gap_s=gap,first_contacts_s=[None if x is None else (x+1)/30 for x in contact],final_lateral_m=float(last[:2]@side),final_longitudinal_m=float(last[:2]@forward),final_distance_m=float(np.linalg.norm(last)),closest_lateral_m=float(err[closest,:2]@side),closest_distance_m=float(np.linalg.norm(err[closest])),yaw_change_deg=float(np.degrees(yawdelta)),any_fall_before_end=bool((~valid[:end,i]).any()),all_alive_at_last_first_contact=bool(valid[max(contact),i].all()) if complete else None)
   rows.append(row)
def summary(rs):
 complete=[r for r in rs if r['contact_gap_s'] is not None];fail=[r for r in rs if not r['success']]
 def mean(key,rr,absval=False):return float(np.mean([abs(r[key]) if absval else r[key] for r in rr])) if rr else None
 return dict(count=len(rs),success_rate=mean('success',rs),all_contact_count=len(complete),gap_mean_s=mean('contact_gap_s',complete),abs_final_lateral_m=mean('final_lateral_m',rs,True),abs_closest_lateral_failed_m=mean('closest_lateral_m',fail,True),any_fall_rate=mean('any_fall_before_end',rs),failure_count=len(fail),lateral_dominant_failures=sum(abs(r['final_lateral_m'])>abs(r['final_longitudinal_m']) for r in fail),shortfall_dominant_failures=sum(r['final_longitudinal_m']<0 and abs(r['final_longitudinal_m'])>=abs(r['final_lateral_m']) for r in fail))
result={'source':str(BASE),'definition':'Contact proxy: both hands <0.12m to actual box boundary while valid; first contact before success or episode end. No force measurements. Missing contact is censored, not zero. Lateral axis perpendicular to initial box-to-goal XY direction.','groups':{},'correlations':{}}
for n in (2,3,4):
 rs=[r for r in rows if r['n']==n];complete=[r for r in rs if r['contact_gap_s'] is not None]
 result['groups'][str(n)]={'all':summary(rs),'success':summary([r for r in rs if r['success']]),'failure':summary([r for r in rs if not r['success']]),'gap_bins':{label:summary([r for r in complete if lo<=r['contact_gap_s']<hi]) for label,lo,hi in [('0-0.5',0,.5),('0.5-2',.5,2),('2+',2,float('inf'))]},'missing_contact':summary([r for r in rs if r['contact_gap_s'] is None])}
 for subset,rr in [('all_contact',complete),('failed_contact',[r for r in complete if not r['success']]),('alive_at_contacts',[r for r in complete if r['all_alive_at_last_first_contact']])]:
  for metric in ('final_lateral_m','closest_lateral_m','final_distance_m'):
   rho,pval=spearmanr([r['contact_gap_s'] for r in rr],[abs(r[metric]) for r in rr]) if len(rr)>2 else (float('nan'),float('nan'))
   result['correlations'][f'{n}/{subset}/{metric}']=dict(count=len(rr),rho=float(rho) if np.isfinite(rho) else None,p=float(pval) if np.isfinite(pval) else None)
(OUT/'summary.json').write_text(json.dumps(result,indent=2));(OUT/'trials.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(result,indent=2))

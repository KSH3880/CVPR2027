"""Proxy diagnosis only: root approach is not an observed interaction attempt."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent;BASE=ROOT/'teacher_runtime/output_etc/check/eval_protocol_v1_legacy_20261111_165027/eval';API=ROOT/'output_etc/check/eval_protocol_v1_legacy_20261111_165027/api';OUT=ROOT/'output_etc/check/enable_early_arrival_20261112';OUT.mkdir(exist_ok=True)
rows=[]
for setting in ['table2/enable_sit','table2/enable_climb','table2/enable_ontop','table1/mixed_2h','table1/mixed_4h']:
 for seed in [1,6,7]:
  p=BASE/setting/f'general_seed{seed}';z=np.load(p/'trace.npz')
  for e,parent in enumerate(z['predecessor']):
   response=json.loads((API/setting/f'general_seed{seed}/env_{e:04}'/'response.raw.txt').read_text());release={r['agent_id']:r['release_time_s'] for r in response['schedule']}
   for f,l in enumerate(parent):
    if l<0:continue
    target=z['goals'][e,l];root=z['roots'][:,e,f];box=z['boxes'][:,e,l,:3];dist=np.linalg.norm(root[:,:2]-target[:2],axis=-1);boxerr=np.linalg.norm(box-target,axis=-1);times=(np.arange(len(root))+1)/30
    active=z['valid'][:,e,f] & (times>=release[f'A{f+1:02}'])
    row=dict(setting=setting,seed=seed,env=e,follower=f,leader=int(l),follower_success=bool(z['ever'][e,f,1]),pair_success=bool(z['simultaneous'][e,f,1]),leader_success=bool(z['ever'][e,l,1]),follower_fell=bool(z['fall_step'][e,f]>=0),proxies={})
    for radius in [.3,.5,.75]:
     ix=np.flatnonzero(active&(dist<=radius));t=int(ix[0]) if len(ix) else None
     # Geometrically unready is distinct from formally invalid predecessor.
     unready=t is not None and boxerr[t]>.2
     absent=t is not None and np.linalg.norm(box[t,:2]-target[:2])>.5
     ready=(boxerr<=.2)&z['valid'][:,e,l]
     row['proxies'][str(radius)]=dict(arrival_s=None if t is None else float(times[t]),unready=bool(unready),clearly_displaced=bool(absent),box_error_m=None if t is None else float(boxerr[t]),leader_reaches_later=bool(unready and ready[t+1:].any()),leader_never_reaches=bool(unready and not ready.any()))
    rows.append(row)
summary={}
for setting in sorted(set(r['setting'] for r in rows)):
 rs=[r for r in rows if r['setting']==setting];fail=[r for r in rs if not r['follower_success']];groups={}
 for radius in ['0.3','0.5','0.75']:
  candidates=[r for r in fail if r['proxies'][radius]['unready']]
  groups[radius]=dict(all_pairs=len(rs),failed_followers=len(fail),failed_arrived_unready=len(candidates),percent_all=100*len(candidates)/len(rs),percent_failed=100*len(candidates)/len(fail) if fail else None,failed_clearly_displaced=sum(r['proxies'][radius]['clearly_displaced'] for r in candidates),leader_reaches_later=sum(r['proxies'][radius]['leader_reaches_later'] for r in candidates),leader_never_reaches=sum(r['proxies'][radius]['leader_never_reaches'] for r in candidates),follower_fell=sum(r['follower_fell'] for r in candidates))
 summary[setting]=groups
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'trials.json').write_text(json.dumps(rows,indent=2));print(json.dumps({s:v['0.5'] for s,v in summary.items()},indent=2))

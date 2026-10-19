"""Summarize saved carryWith trajectories; movement proxies are not carry success."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[2]
base=root/'runs/jointcarry_zeroshot'
result={}
for label in ('box','round_table'):
    run=base/(label+'_carrywith_gpu6_cpu_physx')
    metrics_paths=sorted((run/'metrics').glob('edge_context_*.json'))
    if not metrics_paths: continue
    metrics=json.loads(metrics_paths[-1].read_text())
    rows=[]
    for p in sorted(run.glob('trajectory_*.npz')):
        data=np.load(p); payload=data['payload']; alive=data['alive']
        target=data['target']
        for env in range(payload.shape[1]):
            valid=np.flatnonzero(alive[:,env]); last=valid[-1]
            xy=payload[:last+1,env,:2]; z=payload[:last+1,env,2]
            goal=target[0,env,0,:2]
            distance=np.linalg.norm(xy-goal,axis=-1)
            travel=np.linalg.norm(xy-xy[0],axis=-1)
            rows.append(dict(repeat=p.stem,env=env,initial=distance[0],final=distance[-1],
                             gain=distance[0]-distance[-1],max_xy_displacement=travel.max(),
                             min_z=z.min(),final_z=z[-1]))
    values=lambda key:np.array([r[key] for r in rows])
    result[label]=dict(trials=len(rows),
        scene_goal_success_mean=np.mean([r['final_scene_goal_success'] for r in metrics.values()]),
        ever_scene_goal_success_mean=np.mean([r['ever_scene_goal_success'] for r in metrics.values()]),
        mean_final_holding_proxy=np.mean([r['final_edge_own_success']['holding'] for r in metrics.values()]),
        mean_final_xy_error=float(values('final').mean()),
        mean_xy_error_reduction=float(values('gain').mean()),
        mean_max_xy_displacement=float(values('max_xy_displacement').mean()),
        final_payload_above_0_5m_fraction=float((values('final_z')>.5).mean()),
        note='Movement/height proxies do not establish simultaneous human support.',
        per_trial=rows)
if set(result)=={'box','round_table'}:
    # Compare initial payload pose/target equality separately from rollout divergence.
    checks=[]
    for rep in range(3):
        a=np.load(base/'box_carrywith_gpu6_cpu_physx'/('trajectory_%d.npz'%rep))
        b=np.load(base/'round_table_carrywith_gpu6_cpu_physx'/('trajectory_%d.npz'%rep))
        checks.append(dict(repeat=rep,payload_initial_max_abs=float(np.abs(a['payload'][0]-b['payload'][0]).max()),
                           target_initial_max_abs=float(np.abs(a['target'][0]-b['target'][0]).max())))
    result['initial_state_checks']=checks
out=base/'carrywith_summary.json'
out.write_text(json.dumps(result,indent=2,default=lambda x:x.item())+'\n')
print(json.dumps({k:{a:b for a,b in v.items() if a!='per_trial'} if isinstance(v,dict) else v for k,v in result.items()},indent=2))

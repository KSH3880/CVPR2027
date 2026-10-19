"""Summarize direct GPU PhysX native-object probe; height/travel are proxies only."""
import json
from pathlib import Path
import numpy as np

base=Path(__file__).resolve().parents[2]/'runs/jointcarry_zeroshot/native_gpu_probe_20261015'
context=json.loads((base/'gpu_context.json').read_text())
assert context['physics']=='gpu' and context['pipeline']=='gpu' and context['physical_gpu']==6 and context['process_type']=='C'
paths=sorted((base/'metrics').glob('*.json'))
if not paths:raise SystemExit('GPU evaluation is still running; no result fabricated')
metrics=json.loads(paths[-1].read_text());mapping=json.loads((base/'native_env_assets.json').read_text())
raw=np.load(base/'trajectory_0.npz');states=raw['payload'];targets=raw['target'];alive=raw['alive']
assert np.isfinite(states).all() and np.isfinite(targets).all()
rows=[]
for item in mapping:
    env=item['env'];last=np.flatnonzero(alive[:,env])[-1]
    xy=states[:last+1,env,:2];z=states[:last+1,env,2]
    goal=targets[0,env,0,:2];dist=np.linalg.norm(xy-goal,axis=-1)
    rows.append(dict(name=item['name'],env=env,initial_xy_error=float(dist[0]),final_xy_error=float(dist[-1]),
                     xy_error_reduction=float(dist[0]-dist[-1]),max_height_gain=float((z-z[0]).max()),
                     final_z=float(z[-1]),initial_z=float(z[0])))
summary={'gpu_context':context,'groups':{},'finite_payload_and_target':True,'note':'Height increase and XY movement are proxies, not joint support/carry success.'}
for name in sorted(set(r['name'] for r in rows)):
    chosen=[r for r in rows if r['name']==name]
    summary['groups'][name]=dict(metrics['repeat_0']['asset_groups'][name],
        mean_final_xy_error=float(np.mean([r['final_xy_error'] for r in chosen])),
        mean_xy_error_reduction=float(np.mean([r['xy_error_reduction'] for r in chosen])),
        scenes_height_increase_above_0_1m=sum(r['max_height_gain']>.1 for r in chosen))
summary['per_scene']=rows
(base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='per_scene'},indent=2))

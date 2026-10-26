"""Existing original point loop + existing inner80%-XY/z20cm region classifier."""
from pathlib import Path
import sys,importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tokenhsi'),str(ROOT)]
# Execute the established point adapter, adding only geometry to its saved trace.
path=Path(__file__).with_name('launch.py');text=path.read_text()
old="trace.append((task._humanoid_root_states[:,0,:3].cpu().numpy().copy(),target.cpu().numpy().copy(),(has_collected==0).cpu().numpy().copy(),done.cpu().numpy().copy()))"
new="trace.append((task._humanoid_root_states[:,0,:3].cpu().numpy().copy(),target.cpu().numpy().copy(),(has_collected==0).cpu().numpy().copy(),done.cpu().numpy().copy(),obj.cpu().numpy().copy(),task._box_size[:,0,:2].cpu().numpy().copy(),task._climb_metadata_heights[:,0].cpu().numpy().copy()))"
assert old in text;text=text.replace(old,new)
old="done=np.stack([r[3] for r in trace]))"
new="done=np.stack([r[3] for r in trace]),objects=np.stack([r[4] for r in trace]),extent=trace[0][5],metadata_height=trace[0][6])"
assert old in text;text=text.replace(old,new)
exec(compile(text,str(path)+'[region-trace]','exec'),dict(__file__=str(path),__name__='__main__'))
from evaluation.finetuned.backend import classify
out=Path(sys.argv[sys.argv.index('--output_path')+1]);files=list(out.glob('metrics_*.json'));assert len(files)==1;raw=json.loads(files[0].read_text());rows=[]
for i in range(3):
 t=np.load(out/f'trace_{i}.npz');N=t['root'].shape[1];surface=np.zeros((N,3));surface[:,2]=t['metadata_height'];points=[];regions=[]
 for root,obj,target in zip(t['root'],t['objects'],t['target']):
  p,r,goal=classify(root,obj,surface,t['extent'],'CLIMB');np.testing.assert_allclose(goal,target,atol=2e-6,rtol=0);points.append(p);regions.append(r)
 point=(np.stack(points)&t['alive']).any(0);region=(np.stack(regions)&t['alive']).any(0)
 assert int(point.sum())==raw[f'ObjectSet_test_{i}']['success_trials']
 assert ((t['done'].astype(bool)&t['alive']).sum(0)==1).all()
 row=dict(repeat=i,num_envs=N,point_ever=float(point.mean()),region_ever=float(region.mean()),trace_rescore_exact=True);rows.append(row)
 np.savez_compressed(out/f'success_{i}.npz',point=point,region=region)
import torch
checkpoint=Path(sys.argv[sys.argv.index('--checkpoint')+1]);epoch=int(torch.load(checkpoint,map_location='cpu',weights_only=False)['epoch'])
result=dict(epoch=epoch,checkpoint=str(checkpoint),records=rows,point_mean=100*float(np.mean([x['point_ever'] for x in rows[1:]])),point_std=100*float(np.std([x['point_ever'] for x in rows[1:]],ddof=1)),region_mean=100*float(np.mean([x['region_ever'] for x in rows[1:]])),region_std=100*float(np.std([x['region_ever'] for x in rows[1:]],ddof=1)),protocol='fixed1human1object/Box-Cabinet-Table equal category mix/test/512env/seed0/3repeat/10seconds;repeat0excluded',region='object-local XY inner80%; target Z+-20cm; first episode ever including terminal frame',point='root target 3D distance<=20cm; original TokenHSI metric')
(out/'point_region.json').write_text(json.dumps(result,indent=2));print('POINT_REGION',json.dumps(result),flush=True)

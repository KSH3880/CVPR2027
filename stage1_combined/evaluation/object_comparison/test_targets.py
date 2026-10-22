from pathlib import Path
import json,sys,numpy as np
from prepare import prepare
source=prepare();sys.path.insert(0,str(source/'tokenhsi'))
from utils.single_match_metrics import classify_chair
catalog=json.loads((source/'comparison_assets/catalog.json').read_text())
assert {k:len(v) for k,v in catalog.items()}==dict(Box=18,Chair=26,Cabinet=9,Table=8)
expected={(x,y,z) for x,y in [(.4,.5),(.5,.3),(.6,.4)] for z in [.2,.3,.4,.5,.6,.7]}
assert {tuple(e['bbox']) for e in catalog['Box']}==expected
for category,entries in catalog.items():
    for e in entries:
        assert (Path(e['path'])/'asset.urdf').is_file()
        for action,key,h in [('SIT','sit_surface',.1),('CLIMB','climb_surface',.94)]:
            for yaw in [0,np.pi/2]:
                box=np.zeros((1,1,13));box[...,2]=.5;box[...,5]=np.sin(yaw/2);box[...,6]=np.cos(yaw/2)
                surf=np.array(e[key]);target=np.array([np.cos(yaw)*surf[0]-np.sin(yaw)*surf[1],np.sin(yaw)*surf[0]+np.cos(yaw)*surf[1],.5+surf[2]+h])
                root=np.zeros((1,1,13));root[0,0,:3]=target
                p,r,t=classify_chair(action,root,box,np.array(e['bbox'])[None,None],np.zeros((1,1,3)),np.array([0]),np.array([0]),surf[None,None],np.array(e['support_extent'])[None,None])
                assert p.all() and r.all();np.testing.assert_allclose(t[0],target,atol=1e-12)
                root[0,0,2]+=.201
                p,r,_=classify_chair(action,root,box,np.array(e['bbox'])[None,None],np.zeros((1,1,3)),np.array([0]),np.array([0]),surf[None,None],np.array(e['support_extent'])[None,None]);assert not p.any() and not r.any()
print('PASS 61 native assets, original Box18 dimensions, SIT/CLIMB targets/yaw/z thresholds for all categories')

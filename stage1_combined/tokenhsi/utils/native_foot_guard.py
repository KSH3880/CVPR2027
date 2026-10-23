"""Exact foot-edge checks against every native object in a reset scene."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial.transform import Rotation


class NativeFootGuard:
    def __init__(self,task,bank):
        root=Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:sys.path.insert(0,str(root))
        from rsi_native.geometry import collision_samples,NativeGeometry
        from rsi_native.foot_surface import FootSurface
        manifest=json.loads((Path(bank)/'manifest.json').read_text());names=manifest['body_names'];shapes=collision_samples(root/'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml',names)
        self.feet=FootSurface(root/'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml',names);self.geometry={};self.shapes=shapes;self.geometry_class=NativeGeometry
        self.paths={e['id']:e['path'] for e in manifest['assets']}
        catalog=json.loads((root/'chair_assets/catalog.json').read_text())
        for split in ['train','test','tables']:
            for e in catalog[split]:self.paths[Path(e['path']).name]=e['path']
        self.t=task;self.rejections=0

    def rejected(self,env_ids):
        t=self.t;env=env_ids.cpu().tolist();body=t._kinematic_humanoid_rigid_body_states[env_ids].detach().cpu().numpy();objects=t._box_states[env_ids].detach().cpu().numpy();groups={};bad=np.zeros(len(env),dtype=bool)
        for row,e in enumerate(env):
            for obj,key in enumerate(t._native_asset_keys[e]):
                if key:
                    for owner in range(t.num_agents):groups.setdefault(key,[]).append((row,owner,obj))
        for key,pairs in groups.items():
            if key not in self.geometry:self.geometry[key]=self.geometry_class(self.paths[key],self.shapes)
            for start in range(0,len(pairs),128):
                chunk=pairs[start:start+128];r,o,p=np.asarray(chunk).T;b=body[r,o].copy();obj=objects[r,p];rot=Rotation.from_quat(obj[:,3:7]).inv()
                b[:,:,:3]=np.einsum('nij,nkj->nki',rot.as_matrix(),b[:,:,:3]-obj[:,None,:3])
                q=(Rotation.from_quat(np.repeat(rot.as_quat(),t.num_bodies,axis=0))*Rotation.from_quat(b[:,:,3:7].reshape(-1,4))).as_quat().reshape(len(b),t.num_bodies,4);b[:,:,3:7]=q
                crossed=self.feet.crossing(b,self.geometry[key],0.)
                bad[r[crossed]]=True
        self.rejections+=int(bad.sum());return bad

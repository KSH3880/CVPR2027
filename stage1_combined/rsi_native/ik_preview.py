"""Export physics-accepted IK poses as adjacent before/after snapshot scenes."""
from pathlib import Path
import argparse,json
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--static-bank',action='store_true',help='Preview generated static candidates; not physics-validated')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    m=json.loads((a.audit/'manifest.json').read_text());records=[]
    for category in ('Chair','Table'):
        for skill in ('sit','climb'):
            candidates=[]
            for e in m['assets']:
                if e['category']!=category:continue
                d=dict(np.load(a.audit/('' if a.static_bank else 'audited_bank')/e['id']/(skill+'.npz')))
                for i in np.flatnonzero(d['ik_accepted'] if a.static_bank else d['usable']):
                    change=float(np.linalg.norm(d['ik_joint_delta'][i]))
                    candidates.append((change,e,d,int(i)))
            if candidates:
                _,e,d,i=min(candidates,key=lambda x:x[0])
                for before in (True,False):records.append((e,skill,d,i,before))
    if not records:raise RuntimeError('No physics-accepted IK candidates to preview')
    n=len(records);root=np.zeros((n,2,13),np.float32);root[...,6]=1
    humans=np.zeros((n,2,15,13),np.float32);humans[...,6]=1
    objects=np.zeros((n,4,13),np.float32);objects[...,6]=1
    dof=np.zeros((n,2,32),np.float32);dof_vel=dof.copy();sizes=np.full((n,4,3),.3,np.float32)
    kinds=np.zeros((n,4),np.int64);goal=np.zeros((n,4,3),np.float32);valid=np.zeros((n,4),bool);scenes=[]
    for k,(e,skill,d,i,before) in enumerate(records):
        prefix='ik_before_' if before else ''
        root[k,0]=d[prefix+'root_state'][i];dof[k,0]=d[prefix+'dof'][i];dof_vel[k,0]=d[prefix+'dof_vel'][i];humans[k,0]=d[prefix+'body_state'][i]
        root[k,1,:3]=[6,6,1];humans[k,1,:,:3]=[6,6,1]
        objects[k,0,2]=e['bbox'][2]/2;sizes[k,0]=e['bbox'];kinds[k,0]=2 if e['category']=='Chair' else 1
        for j in range(1,4):objects[k,j,:3]=[6+j,6,.15]
        goal[k,0]=np.r_[e['seat'][:2],e['bbox'][2]/2+e['seat'][2]+(.1 if skill=='sit' else .94)];valid[k,0]=True
        scenes.append(dict(env=k,category=int(kinds[k,0]),task=1 if skill=='sit' else 2,owner=0,stage='BEFORE_failed' if before else ('AFTER_small_IK_static_only' if a.static_bank else 'AFTER_IK_physics_pass'),asset_keys=[e['id'],'','',''],source_index=i,skill=skill,joint_delta_max=float(np.abs(d['ik_joint_delta'][i]).max())))
    np.savez_compressed(a.output/'rsi_reset_snapshot.npz',root=root,humans=humans,objects=objects,dof=dof,dof_vel=dof_vel,sizes=sizes,kinds=kinds,goal=goal,goal_valid=valid)
    (a.output/'rsi_reset_validation.json').write_text(json.dumps(dict(scenes=scenes,source=str(a.audit.resolve()),optimizer_steps=0),indent=2))
    print('IK_PREVIEW',n,'scenes',a.output,flush=True)
if __name__=='__main__':main()

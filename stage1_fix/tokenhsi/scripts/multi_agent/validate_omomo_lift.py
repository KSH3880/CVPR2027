"""Check generated OMOMO references through actual CPU MotionLib and AMP."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'tokenhsi')]
from isaacgym import gymapi  # Before torch.
import hashlib
import json
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation
from utils.motion_lib import MotionLib
from env.tasks.multi_agent.humanoid_ma_carry import build_amp_observations
from tokenhsi.scripts.multi_agent.convert_bones_amp import DOF_BODIES, DOF_OFFSETS, joint_bounds
from tokenhsi.scripts.multi_agent.convert_omomo_lift import NumpyOnly, foot_floor


def validate():
    torch.set_num_threads(1)
    data=ROOT/'tokenhsi/data'
    results=[]
    for variant in ('full','lift'):
        manifest=data/f'dataset_omomo_lift_{variant}_all.yaml'
        entries=yaml.safe_load(manifest.read_text())['motions']['lift']
        lib=MotionLib(str(manifest),'lift',DOF_BODIES,DOF_OFFSETS,[5,8,11,14],'cpu')
        for i,entry in enumerate(entries):
            motion=lib.get_motion(i)
            dest=(data/entry['file']).parent
            report=json.loads((dest/'conversion_report.json').read_text())
            source=ROOT/report['source']
            assert hashlib.sha256(source.read_bytes()).hexdigest()==report['source_sha256']
            with source.open('rb') as f: raw=NumpyOnly(f).load()
            a,b=report['source_frames_inclusive']
            count=b-a+1
            q=motion.local_rotation.numpy()
            assert q.shape==(count,15,4) and motion.fps==30
            np.testing.assert_allclose(np.linalg.norm(q,axis=-1),1,atol=1e-5)
            np.testing.assert_allclose(q[:,[5,8]],np.broadcast_to([0,0,0,1],(count,2,4)),atol=1e-5)
            shift=report['shared_vertical_shift_m']
            source_root=raw['motion'][a:b+1,:3]+np.array([0,0,shift])
            np.testing.assert_allclose(motion.root_translation.numpy(),source_root,atol=1e-6)
            box=np.load(data/entry['obj_file'])
            scale=np.load(dest/'object_scale.npy')
            assert box.shape==(count,7) and len(scale)==count
            np.testing.assert_allclose(np.linalg.norm(box[:,3:],axis=-1),1,atol=1e-5)
            mesh=dest/report['object_mesh']
            vertices=np.array([[float(x) for x in line.split()[1:4]] for line in mesh.read_text().splitlines() if line.startswith('v ')])
            centre=np.asarray(report['object_mesh_raw_bbox_center'])
            # Several mesh vertices, every frame: verify local-origin conversion.
            local=vertices[np.linspace(0,len(vertices)-1,9).astype(int)]
            expected=np.einsum('tij,vj->tvi',raw['obj_rot_mat'][a:b+1],local+centre)*scale[:,None,None]+raw['obj_trans'][a:b+1,None,:]+np.array([0,0,shift])
            actual=np.einsum('tij,vj->tvi',Rotation.from_quat(box[:,3:]).as_matrix(),local)*scale[:,None,None]+box[:,None,:3]
            np.testing.assert_allclose(actual,expected,atol=2e-6)
            assert float(foot_floor(motion).min())>=.0049
            times=torch.linspace(0,(count-1)/30,2*count-1)
            ids=torch.full((len(times),),i,dtype=torch.long)
            states=lib.get_motion_state(ids,times)
            assert states[2].shape==(len(times),32) and all(torch.isfinite(s).all() for s in states)
            root,rot,dof,vel,ang,dvel,key=states
            obs=build_amp_observations(root,rot,vel,ang,dof,dvel,key,True,False,72,DOF_OFFSETS)
            assert obs.shape==(len(times),129) and torch.isfinite(obs).all()
            history_times=(torch.tensor(min((count-1)/30,.5))-torch.arange(10)/30).clamp_min(0)
            history=lib.get_motion_state(torch.full((10,),i,dtype=torch.long),history_times)
            hr,hq,hd,hv,ha,hdv,hk=history
            history_obs=build_amp_observations(hr,hq,hv,ha,hd,hdv,hk,True,False,72,DOF_OFFSETS)
            assert history_obs.reshape(1,-1).shape==(1,1290) and torch.isfinite(history_obs).all()
            obj_states=lib.get_obj_motion_state(ids,times)
            assert all(torch.isfinite(x).all() for x in obj_states)
            if variant=='lift':
                full=dest.parents[1]/'full/phys_humanoid_v3'
                from lpanlib.poselib.skeleton.skeleton3d import SkeletonMotion
                full_motion=SkeletonMotion.from_file(str(full/'ref_motion.npy'))
                np.testing.assert_allclose(q,full_motion.local_rotation.numpy()[a:b+1],atol=1e-6)
                np.testing.assert_allclose(box,np.load(full/'box_motion.npy')[a:b+1],atol=1e-6)
            rv=Rotation.from_quat(q.reshape(-1,4)).as_rotvec().reshape(count,15,3)
            values=np.concatenate([rv[:,j] if hi-lo==3 else rv[:,j,1:2]
                for j,lo,hi in zip(DOF_BODIES,DOF_OFFSETS[:-1],DOF_OFFSETS[1:])],-1)
            lo,hi=joint_bounds(motion.skeleton_tree)
            violation=float(np.maximum(lo-values,values-hi).clip(0).max())
            results.append(dict(sequence=report['sequence'],variant=variant,frames=count,
                amp_features=129,dof=32,source_and_object_geometry_verified=True,
                max_joint_limit_excess_rad=violation,hand_fit_mean_m=report['hand_fit_mean_m']))
    output=data/'dataset_omomo_lift/validation.json'
    output.write_text(json.dumps(dict(status='PASS',scope='format, interpolation, native AMP and shared geometry; no physics validation',results=results),indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__': validate()

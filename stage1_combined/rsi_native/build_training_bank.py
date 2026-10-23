"""Build clamped, kinematically consistent AMP history for accepted native RSI."""
from pathlib import Path
import sys,argparse,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tokenhsi')]
import isaacgym
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from utils.motion_lib import MotionLib
from utils.native_rsi_sampling import audited_candidates
from rsi_native.repair import NativeFK
from rsi_native.geometry import collision_samples, NativeGeometry
from rsi_native.foot_surface import FootSurface
from rsi_native.support import project_ground
from env.tasks.multi_agent.humanoid_ma_carry import build_amp_observations


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--full-motion',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);manifest=json.loads((a.audit/'manifest.json').read_text());fk=NativeFK('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml');offsets=[0,3,6,9,10,13,14,17,20,23,26,29,32];report=[];shapes=collision_samples('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml',fk.names)
    for skill in ['sit','climb']:
        lib=MotionLib('tokenhsi/data/dataset_loco_sit_carry_climb.yaml',skill,[1,2,3,4,6,7,9,10,11,12,13,14],offsets,[5,8,11,14],'cpu')
        for e in manifest['assets']:
            raw=dict(np.load(a.audit/'audited_bank'/e['id']/(skill+'.npz')));mask=audited_candidates(raw,a.full_motion);d={k:v[mask] for k,v in raw.items() if not k.startswith('physics_') and k!='usable'};n=len(d['root_state'])
            if not n:continue
            ids=torch.tensor(np.repeat(d['clip'],9),dtype=torch.long);times=torch.tensor((d['time'][:,None]-np.arange(1,10)[None]*2/60).reshape(-1),dtype=torch.float32)
            root,q,joint,vel,ang,jv,_=lib.get_motion_state(ids,times)
            root=np.concatenate([root.numpy(),q.numpy(),vel.numpy(),ang.numpy()],-1);joint=joint.numpy();jv=jv.numpy()
            if 'ik_joint_delta' in d:
                # Same constant offset throughout the local reference window. Its
                # derivative is zero; coordinates hitting a limit lose velocity.
                base=np.clip(joint,fk.lower,fk.upper);jv[np.abs(base-joint)>1e-6]=0
                joint=base+np.repeat(d['ik_joint_delta'],9,axis=0)
            clamped=np.clip(joint,fk.lower,fk.upper);jv[np.abs(clamped-joint)>1e-6]=0
            transform=np.repeat(d['transform'],9,axis=0);rot=Rotation.from_quat(transform[:,:4]);root[:,:3]=rot.apply(root[:,:3])+transform[:,4:];root[:,3:7]=(rot*Rotation.from_quat(root[:,3:7])).as_quat();root[:,7:10]=rot.apply(root[:,7:10]);root[:,10:13]=rot.apply(root[:,10:13]);body=fk.state(root,clamped,jv)
            if manifest.get('support_mode'):
                projected=np.repeat(d['ground_projected'],9)
                if projected.any():
                    r,b,_=project_ground(root[projected],body[projected],shapes);root[projected]=r;body[projected]=b
            def t(x):return torch.tensor(x,dtype=torch.float32).contiguous()
            amp=build_amp_observations(t(root[:,:3]),t(root[:,3:7]),t(root[:,7:10]),t(root[:,10:13]),t(clamped),t(jv),t(body[:,[5,8,11,14],:3]),True,False,72,offsets).numpy().reshape(n,9,-1)
            assert np.isfinite(amp).all()
            history_rejected=0
            if 'ik_joint_delta' in d:
                # An endpoint correction alone is insufficient for AMP. Reject the
                # candidate if any of its nine corrected history frames penetrates.
                geo=NativeGeometry(e['path'],shapes);feet=FootSurface('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml',fk.names)
                clearance,floor=geo.check(body[...,:3],body[...,3:7],e['bbox'][2]/2)
                crossing=feet.crossing(body,geo,e['bbox'][2]/2)
                safe=((clearance>=-.015)&(floor>=-.02)&~crossing).reshape(n,9).all(-1)
                history_rejected=int((~safe).sum());d={k:v[safe] for k,v in d.items()};amp=amp[safe];n=int(safe.sum())
                print('IK_HISTORY',e['id'],skill,'accepted',n,'rejected',history_rejected,flush=True)
                if not n:continue
            folder=a.output/e['id'];folder.mkdir(exist_ok=True);np.savez_compressed(folder/(skill+'.npz'),**d,amp_history=amp,source_clip_weights=lib._motion_weights.cpu().numpy())
            report.append(dict(asset=e['id'],skill=skill,count=n,phase=int(d['phase'].sum()),nonphase=int((~d['phase']).sum()),lowering=int(((~d['phase'])&(d['root_state'][:,2]-(e['bbox'][2]/2+e['seat'][2]+(.1 if skill=='sit' else .94))>.05)&(d['root_state'][:,2]-(e['bbox'][2]/2+e['seat'][2]+(.1 if skill=='sit' else .94))<.35)).sum()),amp_width=amp.shape[-1],ik_history_rejected=history_rejected));print('NATIVE_TRAIN_BANK',json.dumps(report[-1]),flush=True)
    manifest.update(frame_selection='full_safe_plus_late70' if a.full_motion else 'phase_only',source_audit=str(a.audit.resolve()),history_dt=2/60,history_steps=9,root_height_obs=False,local_root_obs=True,history_recipe='same source clip/time, native32 joint clamp+FK+velocity; same retarget rigid transform; append graph AMP family at reset',builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));(a.output/'report.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()

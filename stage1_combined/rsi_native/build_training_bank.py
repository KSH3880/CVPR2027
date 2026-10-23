"""Build clamped, kinematically consistent AMP history for accepted native RSI."""
from pathlib import Path
import sys,argparse,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tokenhsi')]
import isaacgym
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from utils.motion_lib import MotionLib
from rsi_native.repair import NativeFK
from env.tasks.multi_agent.humanoid_ma_carry import build_amp_observations


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);manifest=json.loads((a.audit/'manifest.json').read_text());fk=NativeFK('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml');offsets=[0,3,6,9,10,13,14,17,20,23,26,29,32];report=[]
    for skill in ['sit','climb']:
        lib=MotionLib('tokenhsi/data/dataset_loco_sit_carry_climb.yaml',skill,[1,2,3,4,6,7,9,10,11,12,13,14],offsets,[5,8,11,14],'cpu')
        for e in manifest['assets']:
            d=dict(np.load(a.audit/'audited_bank'/e['id']/(skill+'_accepted.npz')));n=len(d['root_state'])
            if not n:continue
            ids=torch.tensor(np.repeat(d['clip'],9),dtype=torch.long);times=torch.tensor((d['time'][:,None]-np.arange(1,10)[None]*2/60).reshape(-1),dtype=torch.float32)
            root,q,joint,vel,ang,jv,_=lib.get_motion_state(ids,times)
            root=np.concatenate([root.numpy(),q.numpy(),vel.numpy(),ang.numpy()],-1);joint=joint.numpy();jv=jv.numpy();clamped=np.clip(joint,fk.lower,fk.upper);jv[np.abs(clamped-joint)>1e-6]=0
            transform=np.repeat(d['transform'],9,axis=0);rot=Rotation.from_quat(transform[:,:4]);root[:,:3]=rot.apply(root[:,:3])+transform[:,4:];root[:,3:7]=(rot*Rotation.from_quat(root[:,3:7])).as_quat();root[:,7:10]=rot.apply(root[:,7:10]);root[:,10:13]=rot.apply(root[:,10:13]);body=fk.state(root,clamped,jv)
            def t(x):return torch.tensor(x,dtype=torch.float32).contiguous()
            amp=build_amp_observations(t(root[:,:3]),t(root[:,3:7]),t(root[:,7:10]),t(root[:,10:13]),t(clamped),t(jv),t(body[:,[5,8,11,14],:3]),True,False,72,offsets).numpy().reshape(n,9,-1)
            assert np.isfinite(amp).all()
            folder=a.output/e['id'];folder.mkdir(exist_ok=True);np.savez_compressed(folder/(skill+'.npz'),**d,amp_history=amp)
            report.append(dict(asset=e['id'],skill=skill,count=n,amp_width=amp.shape[-1]));print('NATIVE_TRAIN_BANK',json.dumps(report[-1]),flush=True)
    manifest.update(source_audit=str(a.audit.resolve()),history_dt=2/60,history_steps=9,root_height_obs=False,local_root_obs=True,history_recipe='same source clip/time, native32 joint clamp+FK+velocity; same retarget rigid transform; append graph AMP family at reset',builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));(a.output/'report.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()

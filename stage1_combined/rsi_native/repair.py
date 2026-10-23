"""Bounded native-joint correction and complete native FK; keep raw candidates."""
from pathlib import Path
import argparse,json,hashlib,math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from rsi_native.geometry import NativeGeometry,collision_samples


class NativeFK:
    def __init__(self,xml_path):
        root=ET.parse(xml_path).getroot().find('worldbody/body');self.names=[];self.parents=[];self.offsets=[];self.joints=[];lo=[];hi=[]
        def add(body,parent):
            index=len(self.names);self.names.append(body.get('name'));self.parents.append(parent);self.offsets.append(np.fromstring(body.get('pos','0 0 0'),sep=' '))
            joints=body.findall('joint');self.joints.append((len(lo),len(joints)))
            for j in joints:
                bounds=np.deg2rad(np.fromstring(j.get('range'),sep=' '));lo.append(bounds[0]);hi.append(bounds[1])
            for child in body.findall('body'):add(child,index)
        add(root,-1);self.lower=np.asarray(lo);self.upper=np.asarray(hi)
        assert len(lo)==32

    def forward(self,root,joint):
        n=len(root);pos=np.empty((n,len(self.names),3));quat=np.empty((n,len(self.names),4));pos[:,0]=root[:,:3];quat[:,0]=root[:,3:7]
        for i,parent in enumerate(self.parents[1:],1):
            offset,size=self.joints[i];parent_rot=Rotation.from_quat(quat[:,parent])
            local=(Rotation.from_rotvec(joint[:,offset:offset+3]) if size==3 else Rotation.from_euler('y',joint[:,offset]) if size==1 else Rotation.identity(n))
            pos[:,i]=pos[:,parent]+parent_rot.apply(np.tile(self.offsets[i],(n,1)));quat[:,i]=(parent_rot*local).as_quat()
        return pos,quat

    def state(self,root,joint,velocity):
        pos,quat=self.forward(root,joint);eps=1e-4
        nextroot=root.copy();nextroot[:,:3]+=root[:,7:10]*eps;nextroot[:,3:7]=(Rotation.from_rotvec(root[:,10:13]*eps)*Rotation.from_quat(root[:,3:7])).as_quat()
        nextjoint=np.clip(joint+velocity*eps,self.lower,self.upper);nextpos,nextquat=self.forward(nextroot,nextjoint)
        bv=(nextpos-pos)/eps;ba=(Rotation.from_quat(nextquat.reshape(-1,4))*Rotation.from_quat(quat.reshape(-1,4)).inv()).as_rotvec().reshape(pos.shape)/eps
        return np.concatenate([pos,quat,bv,ba],-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();source=a.source.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text());xml=Path('tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml');fk=NativeFK(xml);assert fk.names==manifest['body_names']
    shapes=collision_samples(xml,fk.names);report=[]
    for index,e in enumerate(manifest['assets']):
        geo=NativeGeometry(e['path'],shapes);folder=out/e['id'];folder.mkdir()
        for skill in ['sit','climb']:
            d=dict(np.load(source/e['id']/(skill+'.npz')));joint=d['dof'];clamped=np.clip(joint,fk.lower,fk.upper);delta=clamped-joint
            # Reconstructed FK must describe the same original skeleton before correction.
            original_pos,_=fk.forward(d['root_state'],joint);errors=np.linalg.norm(original_pos-d['body_state'][...,:3],axis=-1); fk_error=errors[:,[i for i,n in enumerate(fk.names) if 'hand' not in n]].max(); hand_error=errors[:,[i for i,n in enumerate(fk.names) if 'hand' in n]].max()
            if fk_error>.01:raise ValueError(f'Native FK mismatch {e["id"]} {skill}: {fk_error}')
            max_delta=np.abs(delta).max(-1);changed=np.abs(delta)>1e-6;velocity=d['dof_vel'].copy();velocity[changed]=0.
            d['source_dof']=joint.copy();d['dof']=clamped;d['dof_vel']=velocity;d['dof_correction_max']=max_delta
            d['body_state']=fk.state(d['root_state'],clamped,velocity)
            clearance,floor=geo.check(d['body_state'][...,:3],d['body_state'][...,3:7],e['bbox'][2]/2)
            static=d['anchor_valid']&(max_delta<=.30)&(clearance>=-.015)&(floor>=-.02)
            facing=np.asarray(e['facing'] if e['category']=='Chair' else [1.,0.],dtype=float);facing/=np.linalg.norm(facing)
            offsets=sorted([(f,z) for f in [0.,.075,.15] for z in [0.,.025,.05,.075,.1]],key=lambda x:x[0]**2+x[1]**2)
            target=np.r_[e['seat'][:2],e['bbox'][2]/2+e['seat'][2]+(.1 if skill=='sit' else .94)]
            for front,lift in offsets:
                candidates=np.flatnonzero(d['phase']&d['anchor_valid']&(max_delta<=.30)&~static)
                if not len(candidates):break
                shift=np.r_[facing*front,lift];depth,ground=geo.check(d['body_state'][candidates,:,:3]+shift,d['body_state'][candidates,:,3:7],e['bbox'][2]/2)
                ok=(depth>=-.015)&(ground>=-.02)&(np.linalg.norm(d['root_state'][candidates,:3]+shift-target,axis=-1)<=.20)
                chosen=candidates[ok];d['root_state'][chosen,:3]+=shift;d['body_state'][chosen,:,:3]+=shift;d['transform'][chosen,4:]+=shift;d['correction'][chosen]+=shift
                clearance[chosen]=depth[ok];floor[chosen]=ground[ok];static[chosen]=True
            d.update(clearance=clearance,floor_clearance=floor,static_pass=static)
            np.savez_compressed(folder/(skill+'.npz'),**d)
            row=dict(asset=e['id'],category=e['category'],skill=skill,candidates=len(joint),static_pass=int(static.sum()),phase_pass=int((static&d['phase']).sum()),joint_excess_rejected=int((max_delta>.3).sum()),max_joint_correction=float(max_delta.max()),native_fk_error=float(fk_error),source_hand_fk_error=float(hand_error),mesh_watertight=geo.watertight)
            report.append(row);print('RSI_REPAIR',index+1,len(manifest['assets']),json.dumps(row),flush=True)
        (out/'geometry_report.json').write_text(json.dumps(report,indent=2))
    manifest.update(source_bank=str(source),repair='clamp actual native32 joint limits; maximum correction .30rad; recompute FK/body velocities; bounded extra pose offsets; raw source_dof retained',repair_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('REPAIRED',out,flush=True)
if __name__=='__main__':main()

"""Bounded arm IK for a table held above the floor; candidates need runtime checks."""
from pathlib import Path
import sys, json, hashlib, xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'joint_carry/scripts'))
from carry_joint_preview import forward_kinematics, DOF_BODIES, DOF_SIZES
from correct_carry_candidates import physical_rotations

def main():
    source=ROOT/'joint_carry/carry_joint_corrected/carryWith/cooperative_rsi_candidate.npz'
    data=dict(np.load(source,allow_pickle=False));names=data['joint_names'].tolist()
    xml=ET.parse(ROOT/'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml')
    bodies={b.get('name'):b for b in xml.findall('.//worldbody//body')}
    urdf=ROOT.parent/'runs/table_carry/assets/teamhoi_default_rectangle/asset.urdf'
    geom=[]
    for c in ET.parse(urdf).findall('.//collision'):
        geom.append((np.fromstring(c.find('origin').get('xyz'),sep=' '),np.fromstring(c.find('geometry/box').get('size'),sep=' ')))
    frames=np.linspace(10,len(data['dof_position'])-11,16).astype(int)
    local=physical_rotations(data)[frames].copy();roots=data['root_position'][frames].copy()
    inverse=R.from_quat(data['box_rotation_xyzw'][frames]).inv()
    for a in range(2):
        roots[:,a]=inverse.apply(roots[:,a]-data['box_position'][frames])+np.array([0,0,.73])
        local[:,a,0]=(inverse*R.from_quat(local[:,a,0])).as_quat()
    signs=np.sign(roots[:,:,1]); roots[:,:,0]=0;roots[:,:,1]=signs*.74
    # Restore stance elevation after canonicalizing relative to the source box.
    for f in range(len(frames)):
      for a in range(2):
        xyz,wq=forward_kinematics(local[f:f+1,a],roots[f:f+1,a],data['parent_indices'],data['local_translation'])
        floor=np.inf
        for foot in ('left_foot','right_foot'):
            j=names.index(foot)
            vertices=np.array([[float(x) for x in line.split()[1:4]] for line in (ROOT/'tokenhsi/data/assets/smpl_foot_meshes'/ (foot+'.obj')).read_text().splitlines() if line.startswith('v ')])
            floor=min(floor,float((R.from_quat(wq[0,j]).apply(vertices)+xyz[0,j])[:,2].min()))
        roots[f,a,2]+=.003-floor
    errors=[];collisions=[]
    for f in range(len(frames)):
      for a in range(2):
       for shoulder,elbow,hand in [(3,4,5),(6,7,8)]:
        xyz,wq=forward_kinematics(local[f:f+1,a],roots[f:f+1,a],data['parent_indices'],data['local_translation'])
        sp=xyz[0,shoulder]; torso=R.from_quat(wq[0,1])
        target=np.array([np.sign(xyz[0,hand,0])*.22,signs[f,a]*.45,1.081])
        joints=bodies[names[shoulder]].findall('joint')+bodies[names[elbow]].findall('joint')
        bounds=np.deg2rad([np.fromstring(j.get('range'),sep=' ') for j in joints])
        base=np.r_[R.from_quat(local[f,a,shoulder]).as_rotvec(),R.from_quat(local[f,a,elbow]).as_rotvec()[1]]
        clouds=[np.linspace(*np.fromstring(bodies[names[j]].find('geom').get('fromto'),sep=' ').reshape(2,3),16) for j in (shoulder,elbow)]
        def locations(v):
            u=torso*R.from_rotvec(v[:3]);ep=sp+u.apply(data['local_translation'][elbow]);l=u*R.from_rotvec([0,v[3],0]);h=ep+l.apply(data['local_translation'][hand])
            return h,np.r_[sp+u.apply(clouds[0]),ep+l.apply(clouds[1]),h[None]],np.r_[np.full(16,.045),np.full(16,.04),.04]
        def penetration(points,radii):
            out=[]
            for center,size in geom:
                d=np.abs(points-(center+np.array([0,0,.73])))-size/2
                sdf=np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(-1),0)
                out.extend(np.maximum(radii-.002-sdf,0))
            return np.array(out)
        def residual(v):
            h,p,r=locations(v)
            return np.r_[(h-target)/.003,penetration(p,r)/.002,.08*(v-base)]
        rng=np.random.default_rng(1900+f*4+a*2+shoulder)
        starts=[np.clip(base,bounds[:,0]+1e-4,bounds[:,1]-1e-4)]
        starts.extend(rng.uniform(bounds[:,0]+.01,bounds[:,1]-.01,size=(18,4)))
        fit=None
        for start in starts:
            candidate=least_squares(residual,start,bounds=(bounds[:,0]+1e-5,bounds[:,1]-1e-5),max_nfev=160)
            if fit is None or candidate.cost<fit.cost: fit=candidate
            h,p,r=locations(fit.x)
            if np.linalg.norm(h-target)<.008 and penetration(p,r).max()<.008: break
        local[f,a,shoulder]=R.from_rotvec(fit.x[:3]).as_quat();local[f,a,elbow]=R.from_rotvec([0,fit.x[3],0]).as_quat()
        h,p,r=locations(fit.x);errors.append(float(np.linalg.norm(h-target)));collisions.append(float(penetration(p,r).max()))
    frame_errors=np.asarray(errors).reshape(len(frames),4).max(1)
    frame_collision=np.asarray(collisions).reshape(len(frames),4).max(1)
    print('Per-frame IK error',frame_errors.tolist(),flush=True)
    print('Per-frame arm penetration',frame_collision.tolist(),flush=True)
    valid=(frame_errors<.025)&(frame_collision<.008)
    if valid.sum()<4: raise SystemExit('Too few feasible held poses: '+str(int(valid.sum())))
    local,roots,frames=local[valid],roots[valid],frames[valid]
    errors=frame_errors[valid].tolist();collisions=frame_collision[valid].tolist()
    states=[]
    for a in range(2):
        xyz,quat=forward_kinematics(local[:,a],roots[:,a],data['parent_indices'],data['local_translation'])
        states.append(np.concatenate([xyz,quat,np.zeros((*xyz.shape[:-1],6))],-1))
    body=np.stack(states,1);angles=R.from_quat(local.reshape(-1,4)).as_rotvec().reshape(*local.shape[:-1],3)
    dof=np.concatenate([angles[:,:,j] if n==3 else angles[:,:,j,1:2] for j,n in zip(DOF_BODIES,DOF_SIZES)],-1)
    box=np.zeros((len(frames),13));box[:,2]=.73;box[:,6]=1
    target=ROOT.parent/'runs/table_carry/table_undergrasp_rsi.npz'
    np.savez_compressed(target,root_state=body[:,:,0],body_state=body,dof_position=dof,dof_velocity=np.zeros_like(dof),box_state=box,source_frame=frames)
    report=dict(frames=len(frames),max_hand_error_m=max(errors),max_arm_penetration_m=max(collisions),source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),physics_validated=False,table_bottom_height=.30,anchor_z=.351)
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if max(errors)>.025 or max(collisions)>.015:raise SystemExit('IK candidate failed geometric checks')
if __name__=='__main__':main()

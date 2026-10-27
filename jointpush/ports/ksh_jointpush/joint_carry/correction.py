from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from joint_carry.kinematics import DOF_BODIES, DOF_SIZES, forward_kinematics, pack

ASSET = Path("tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml")

def sdf_box(points, size):
    d=np.abs(points)-size/2
    return np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(-1),0)

def physical_rotations(data):
    q=np.zeros_like(data['local_rotation_xyzw'])
    q[...,3]=1
    q[:,:,0]=data['root_rotation_xyzw']
    offset=0
    for j,size in zip(DOF_BODIES,DOF_SIZES):
        rv=np.zeros(q.shape[:2]+(3,))
        if size==3:
            rv[:]=data['dof_position'][:,:,offset:offset+3]
        else:
            rv[:,:,1]=data['dof_position'][:,:,offset]
        q[:,:,j]=R.from_rotvec(rv.reshape(-1,3)).as_quat().reshape(q.shape[:2]+(4,))
        offset+=size
    return q

def feet_floor_lift(q, roots, data):
    xyz,rot=forward_kinematics(q[:,0],roots[:,0],data['parent_indices'],data['local_translation'])
    floor=np.full(len(q),np.inf)
    for name in ('left_foot','right_foot'):
        j=data['joint_names'].tolist().index(name)
        path=ASSET.parent/'../smpl_foot_meshes'/ (name+'.obj')
        vertices=np.asarray([[float(v) for v in line.split()[1:4]] for line in path.read_text().splitlines()
                             if line.startswith('v ')])
        r=R.from_quat(rot[:,j])
        min_z=np.stack([r.apply(v)[:,2] for v in vertices],axis=1).min(1)+xyz[:,j,2]
        floor=np.minimum(floor,min_z)
    # Constant lift over the clip avoids injecting artificial vertical velocity.
    lift=max(0.,.001-float(floor.min()))
    return lift

def correct(data):
    q=physical_rotations(data)
    roots=data['root_position'].copy()
    lift=feet_floor_lift(q,roots,data)
    roots[:,:,2]+=lift
    xyz,world_q=forward_kinematics(q[:,0],roots[:,0],data['parent_indices'],data['local_translation'])
    inv=R.from_quat(data['box_rotation_xyzw']).inv()
    size=data['box_size']
    xml=ET.parse(ASSET).getroot()
    traces=[]
    for shoulder,elbow,hand in [(3,4,5),(6,7,8)]:
        shoulder_name=data['joint_names'][shoulder]
        elbow_name=data['joint_names'][elbow]
        body=next(b for b in xml.findall('.//body') if b.get('name')==shoulder_name)
        lower=next(b for b in xml.findall('.//body') if b.get('name')==elbow_name)
        joints=body.findall('joint')+lower.findall('joint')
        bounds=np.asarray([np.fromstring(j.get('range'),sep=' ') for j in joints])*np.pi/180
        upper_segment=np.fromstring(body.find('geom').get('fromto'),sep=' ').reshape(2,3)
        lower_segment=np.fromstring(lower.find('geom').get('fromto'),sep=' ').reshape(2,3)
        upper_points=np.linspace(*upper_segment,24)
        lower_points=np.linspace(*lower_segment,24)
        prev_delta=np.zeros(4)
        original=np.concatenate([R.from_quat(q[:,0,shoulder]).as_rotvec(),
                                 R.from_quat(q[:,0,elbow]).as_rotvec()[:,1:2]],axis=1)
        for f in range(len(q)):
            shoulder_position=inv[f].apply(xyz[f,shoulder]-data['box_position'][f])
            torso_rotation=inv[f]*R.from_quat(world_q[f,1])
            reference_hand=inv[f].apply(data['body_position'][f,0,hand]-data['box_position'][f])
            reference_hand[2]+=lift
            # Smoothly enable a side-face grasp as the source hands approach.
            gate=np.clip((size[1]/2+.15-abs(reference_hand[1]))/.15,0,1)
            gate*=np.clip((size[2]/2+.15-abs(reference_hand[2]))/.15,0,1)
            target=reference_hand.copy()
            desired_x=np.sign(reference_hand[0])*(size[0]/2+.04-.005)
            target[0]=(1-gate)*target[0]+gate*desired_x
            base=original[f]

            def locations(values):
                upper=torso_rotation*R.from_rotvec(values[:3])
                elbow_position=shoulder_position+upper.apply(data['local_translation'][elbow])
                lower_r=upper*R.from_rotvec([0,values[3],0])
                hand_position=elbow_position+lower_r.apply(data['local_translation'][hand])
                upper_cloud=shoulder_position+upper.apply(upper_points)
                lower_cloud=elbow_position+lower_r.apply(lower_points)
                return hand_position,upper_cloud,lower_cloud

            def residual(values):
                h,u,l=locations(values)
                collision=np.concatenate([np.maximum(.045+.003-sdf_box(u,size),0),
                                          np.maximum(.04+.003-sdf_box(l,size),0),
                                          np.maximum(.04-.008-sdf_box(h[None],size),0)])
                return np.concatenate([(h-target)/.003,collision/.002,
                                       (values-base)*.8,((values-base)-prev_delta)*.4])

            lo=bounds[:,0]+1e-5
            hi=bounds[:,1]-1e-5
            result=least_squares(residual,np.clip(base+prev_delta,lo+1e-7,hi-1e-7),
                                 bounds=(lo,hi),max_nfev=100,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            q[f,0,shoulder]=R.from_rotvec(result.x[:3]).as_quat()
            q[f,0,elbow]=R.from_rotvec([0,result.x[3],0]).as_quat()
            prev_delta=result.x-base
            h,_,_=locations(result.x)
            traces.append(dict(frame=f,arm=str(shoulder_name),grasp_gate=float(gate),
                               target_error_m=float(np.linalg.norm(h-target)),
                               maximum_joint_change_deg=float(np.abs(prev_delta).max()*180/np.pi)))
    # Clone articulated pose exactly; the existing opposite root rotation stays.
    q[:,1,1:]=q[:,0,1:]
    output=pack(q,roots,data['parent_indices'],data['local_translation'],
                np.concatenate([data['box_position'],data['box_rotation_xyzw']],axis=-1),
                size,int(data['fps']),data['joint_names'].tolist())
    for key in ('skill','source_rsi_allowed','grip_inset_m'):
        output[key]=data[key]
    output['physics_valid_frame_mask']=np.zeros(len(q),dtype=bool)
    output['ik_applied']=np.asarray(True)
    output['simulator_pose_exact']=np.asarray(True)
    output['render_caption']=np.asarray('32-DOF arm IK | {:g} cm interior grips'.format(float(data['grip_inset_m'])*100))
    output['render_footer']=np.asarray('Corrected reference | pending physics screen | not a policy rollout')
    unchanged=[0,1,2]+list(range(9,15))
    physical=physical_rotations(data)
    np.testing.assert_allclose(q[:,:,unchanged],physical[:,:,unchanged],atol=1e-12)
    return output,dict(root_height_lift_m=lift,arm_fit=traces,
                       maximum_arm_dof_change_deg=max(t['maximum_joint_change_deg'] for t in traces),
                       maximum_hand_target_error_m=max(t['target_error_m'] for t in traces))

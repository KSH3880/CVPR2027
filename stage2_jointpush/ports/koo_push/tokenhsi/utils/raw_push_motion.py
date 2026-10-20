"""BONES source-space AMP features: no humanoid sizing, IK, limits or grounding."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch
from utils.bones_bvh import read_bvh

FORMAT = 'bones_source_amp_unscaled_v1'
BODY_NAMES = ('pelvis','torso','head','right_upper_arm','right_lower_arm','right_hand',
              'left_upper_arm','left_lower_arm','left_hand','right_thigh','right_shin',
              'right_foot','left_thigh','left_shin','left_foot')
SOURCE_NAMES = ('Hips','Spine2','Head','RightArm','RightForeArm','RightHand',
                'LeftArm','LeftForeArm','LeftHand','RightLeg','RightShin','RightFoot',
                'LeftLeg','LeftShin','LeftFoot')
PARENTS = (-1,0,1,1,3,4,1,6,7,0,9,10,0,12,13)


def unit(vector):
    length = np.linalg.norm(vector)
    if length < 1e-9: raise ValueError('Degenerate source anatomical axis')
    return vector/length


def frame(up, left):
    z = unit(up); y = unit(left-np.dot(left,z)*z)
    return np.column_stack((unit(np.cross(y,z)),y,z))


def limb_frame(upper, lower, fallback):
    z = -unit(upper)
    normal = -np.cross(unit(upper),unit(lower))
    y = unit(normal if np.linalg.norm(normal) > .02 else fallback)
    y = unit(y-np.dot(y,z)*z)
    return np.column_stack((unit(np.cross(y,z)),y,z))


def source_arrays(nodes, positions, rotations, dt, start, end, fps=30):
    """Map source anatomy to AMP slots while retaining measured metre positions.

    Anatomical axes express source joints in the policy's rotation convention;
    no target skeleton lengths, mesh, joint bounds or optimization enter here.
    """
    ids = {node['name']: i for i,node in enumerate(nodes)}
    missing = set(SOURCE_NAMES+('Chest','Neck1','LeftToeBase','RightToeBase'))-ids.keys()
    if missing: raise ValueError('Missing source joints: '+str(sorted(missing)))
    if not 0 <= start < end <= (len(positions)-1)*dt or fps <= 0:
        raise ValueError('Invalid source crop/fps')
    original = np.arange(len(positions))*dt
    times = np.arange(start,end+1e-8,1/fps)
    if len(times)<2: raise ValueError('AMP clip needs at least two frames')
    pos = np.stack([np.stack([np.interp(times,original,positions[:,j,k])
        for k in range(3)],-1) for j in range(len(nodes))],1)
    rot = np.stack([Slerp(original,Rotation.from_matrix(rotations[:,j]))(times).as_matrix()
                    for j in range(len(nodes))],1)
    left = positions[0,ids['LeftArm']]-positions[0,ids['RightArm']]
    left[1]=0; left=unit(left); up=np.array([0.,1.,0.])
    basis=np.stack((unit(np.cross(left,up)),left,up))
    pos=pos@basis.T
    # Translation does not change source heights, segment lengths or velocities.
    xy_origin=pos[0,ids['Hips'],:2].copy()
    pos[...,:2]-=xy_origin
    mapped=pos[:,[ids[name] for name in SOURCE_NAMES]]
    global_rot=[]
    for f,p in enumerate(pos):
        root=frame(p[ids['Chest']]-p[ids['Hips']],p[ids['LeftLeg']]-p[ids['RightLeg']])
        desired=np.tile(root,(15,1,1))
        desired[1]=frame(p[ids['Neck1']]-p[ids['Spine2']],p[ids['LeftArm']]-p[ids['RightArm']])
        desired[2]=basis@rot[f,ids['Head']]@np.array([[0,0,1],[1,0,0],[0,1,0]])
        for prefix,arm,thigh in (('Right',3,9),('Left',6,12)):
            upper=p[ids[prefix+'ForeArm']]-p[ids[prefix+'Arm']]
            lower=p[ids[prefix+'Hand']]-p[ids[prefix+'ForeArm']]
            desired[arm]=limb_frame(upper,lower,root[:,1])
            angle=np.arccos(np.clip(np.dot(unit(upper),unit(lower)),-1,1))
            desired[arm+1]=desired[arm]@Rotation.from_rotvec([0,-angle,0]).as_matrix()
            # Fixed axis remapping keeps wrist orientation changes from BVH.
            source_forearm=basis@rot[f,ids[prefix+'ForeArm']]
            source_hand=basis@rot[f,ids[prefix+'Hand']]
            desired[arm+2]=desired[arm+1]@source_forearm.T@source_hand
            upper=p[ids[prefix+'Shin']]-p[ids[prefix+'Leg']]
            lower=p[ids[prefix+'Foot']]-p[ids[prefix+'Shin']]
            # The same anatomical down-bone convention, with no target offsets.
            desired[thigh]=limb_frame(upper,lower,root[:,1])
            z=-unit(lower); y=desired[thigh][:,1]; y=unit(y-np.dot(y,z)*z)
            desired[thigh+1]=np.column_stack((unit(np.cross(y,z)),y,z))
            toe=unit(p[ids[prefix+'ToeBase']]-p[ids[prefix+'Foot']])
            source_foot=basis@rot[f,ids[prefix+'Foot']]
            initial=basis@rot[0,ids[prefix+'Foot']]
            axis=int(np.argmax(np.abs(initial[2])))
            foot_up=source_foot[:,axis]*np.sign(initial[2,axis])
            z=unit(foot_up-toe*np.dot(foot_up,toe))
            desired[thigh+2]=np.column_stack((toe,unit(np.cross(z,toe)),z))
        global_rot.append(desired)
    global_rot=np.stack(global_rot)
    local_rot=global_rot.copy()
    for joint,parent in enumerate(PARENTS):
        if parent>=0: local_rot[:,joint]=global_rot[:,parent].transpose(0,2,1)@global_rot[:,joint]
    def quaternions(matrices):
        result=Rotation.from_matrix(matrices.reshape(-1,3,3)).as_quat().reshape(len(times),15,4)
        for f in range(1,len(result)):
            result[f,(result[f]*result[f-1]).sum(-1)<0]*=-1
        return result.astype(np.float32)
    # First differences match MotionLib's frame velocity convention.
    velocity=np.diff(mapped,axis=0)*fps
    velocity=np.concatenate((velocity,velocity[-1:]),0)
    angular=Rotation.from_matrix((global_rot[1:]@global_rot[:-1].transpose(0,1,3,2)).reshape(-1,3,3)).as_rotvec().reshape(-1,15,3)*fps
    angular=np.concatenate((angular,angular[-1:]),0)
    return dict(format=np.array(FORMAT),fps=np.array(float(fps)),
        global_translation=mapped.astype(np.float32),global_rotation=quaternions(global_rot),
        local_rotation=quaternions(local_rot),global_velocity=velocity.astype(np.float32),
        global_angular_velocity=angular.astype(np.float32))


def read_source(path,start,end,fps=30):
    nodes,pos,rot,dt=read_bvh(path)
    return source_arrays(nodes,pos,rot,dt,start,end,fps)


class SourceAmpMotion:
    """MotionLib-compatible data container; source references are AMP-only."""
    source_amp = True
    @classmethod
    def from_file(cls,path):
        with np.load(path,allow_pickle=False) as data:
            if str(data['format'].item())!=FORMAT:
                raise ValueError('Not an unscaled BONES source AMP cache: '+str(path))
            result=cls();result.fps=float(data['fps']);result.num_joints=15
            for field in ('global_translation','global_rotation','local_rotation',
                          'global_velocity','global_angular_velocity'):
                value=np.array(data[field],copy=True)
                if not np.isfinite(value).all(): raise ValueError('Nonfinite raw AMP '+field)
                setattr(result,field,torch.as_tensor(value,dtype=torch.float32))
        count=len(result.global_translation)
        if count<2 or not np.isfinite(result.fps) or result.fps<=0:
            raise ValueError('Invalid raw AMP length/fps')
        for field,width in (('global_translation',3),('global_rotation',4),('local_rotation',4),
                            ('global_velocity',3),('global_angular_velocity',3)):
            if getattr(result,field).shape!=(count,15,width): raise ValueError('Invalid raw AMP '+field+' shape')
        result.global_root_velocity=result.global_velocity[:,0]
        result.global_root_angular_velocity=result.global_angular_velocity[:,0]
        result.tensor=result.global_translation.flatten(1)
        return result

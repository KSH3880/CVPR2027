"""Closing spring and contact-aware continuous hold kernels."""
import math
import torch


def validate_hold(env):
    spec=env['doorHold']; spring=env['door']['autoClose']
    distance_range=env['doorSpawnDistanceRange']
    if (len(distance_range)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in distance_range)
            or not 0<distance_range[0]<=distance_range[1]
            or distance_range[1]!=env['doorApproachAMP']['spawn_distance']):
        raise ValueError('Invalid DoorHold spawn distance range')
    if env['numAgents']!=1 or env['numObjects']!=1 or env['door'].get('openingDirection','push')!='push':
        raise ValueError('DoorHold currently requires one person and one push door')
    if set(spec)!={'duration','hand_distance','min_hand_force','min_door_force','contact_grace','posture'} or set(spring)!={'stiffness','damping','max_torque'}:
        raise ValueError('Invalid DoorHold/spring fields')
    for value in [v for k,v in spec.items() if k!='posture']+list(spring.values()):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError('Hold/spring parameters must be finite and positive')

    validate_posture(spec['posture'])


def validate_posture(spec):
    if set(spec)!={'min_pelvis_height_ratio','max_torso_tilt','max_foot_height','min_foot_force'}:
        raise ValueError('Invalid door posture fields')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in spec.values()):
        raise ValueError('Door posture parameters must be finite and positive')
    if not 0<spec['min_pelvis_height_ratio']<=1 or not 0<spec['max_torso_tilt']<math.pi/2:
        raise ValueError('Invalid door posture thresholds')


def standing_posture(pelvis_height, torso_quat, foot_positions, foot_forces, standing_height, spec):
    """At least one low foot with an upward support force; torso local +Z is up."""
    pelvis_ok=pelvis_height>=standing_height*spec['min_pelvis_height_ratio']
    up_z=1-2*(torso_quat[...,0].square()+torso_quat[...,1].square())
    torso_ok=up_z>=math.cos(spec['max_torso_tilt'])
    foot_ok=((foot_positions[...,2]<=spec['max_foot_height']) &
             (foot_forces[...,2]>=spec['min_foot_force'])).any(-1)
    return pelvis_ok & torso_ok & foot_ok


def init_posture_bodies(task):
    actor=task.humanoid_handles_all[0][0]
    indices=[task.gym.find_actor_rigid_body_handle(task.envs[0],actor,name)
             for name in ('pelvis','torso','right_foot','left_foot')]
    if min(indices)<0:
        raise ValueError('Door posture requires pelvis, torso and both feet')
    task._posture_body_ids=indices
    task._posture_foot_forces=torch.zeros_like(task._contact_forces[...,[indices[2],indices[3]],:])
    # Fixed asset standing pose, never an episode's sampled/crouched AMP pose.
    task._standing_pelvis_height=task._initial_humanoid_rigid_body_states[...,indices[0],2].clone()
    if not (task._standing_pelvis_height>0).all():
        raise ValueError('Invalid asset standing pelvis height')


def collect_foot_support(task):
    right,left=task._posture_body_ids[2:]
    low=task._rigid_body_pos[...,[right,left],2]<=task._hold_spec['posture']['max_foot_height']
    forces=torch.where(low[...,None],task._contact_forces[...,[right,left],:],0.)
    task._posture_foot_forces.copy_(torch.maximum(task._posture_foot_forces,forces))


def current_posture(task):
    pelvis,torso,right,left=task._posture_body_ids
    return standing_posture(task._rigid_body_pos[...,pelvis,2],
        task._rigid_body_rot[...,torso,:],task._rigid_body_pos[...,[right,left],:],
        task._posture_foot_forces,task._standing_pelvis_height,task._hold_spec['posture'])


def support_with_grace(age, hits, near, dt, grace):
    age=torch.where(hits & near,torch.zeros_like(age),age+dt)
    age=torch.where(near,age,torch.full_like(age,grace+dt)).clamp(max=grace+dt)
    return age,near & (age<=grace+1e-6)


def closing_torque(angle, velocity, spec):
    return (-spec['stiffness']*angle-spec['damping']*velocity).clamp(-spec['max_torque'],spec['max_torque'])


def hand_support_contact(hands, handles, hand_forces, leaf_forces, angles, spec):
    """GPU contact proxy: nearby own hand, push reaction and leaf push force.

    Net force tensors do not expose collision pairs. CPU physics audit checks
    real hand/door pairs separately; this is not a grasp constraint.
    """
    normal=torch.stack((-torch.sin(angles),torch.cos(angles),torch.zeros_like(angles)),-1)
    near=(hands-handles[:,:,None]).norm(dim=-1)<=spec['hand_distance']
    reaction=-(hand_forces*normal[:,:,None]).sum(-1)
    door_push=(leaf_forces*normal).sum(-1)
    return (near & (reaction>=spec['min_hand_force'])).any(-1) & (door_push>=spec['min_door_force'])


def update_hold(elapsed, angles, contact, threshold, duration, dt):
    supported=(angles>=threshold)&contact
    elapsed=torch.where(supported,(elapsed+dt).clamp(max=duration),torch.zeros_like(elapsed))
    success=supported&(elapsed>=duration-1e-6)
    return elapsed, success

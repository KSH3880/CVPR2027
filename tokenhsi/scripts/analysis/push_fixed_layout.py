"""User-authorized PUSH positions; native RSI pose and variable headings."""
import math
import os
from pathlib import Path
import torch
from utils.jointpush_geometry import rotate, multiply, yaw_from_quat

ROOT=Path(__file__).resolve().parents[3]
REFERENCE=ROOT/'output_etc/check/push_natural_prefilter_20261112/source/seed34/initial.pt'
A=1
B=0
A_GAP=.9
B_GAP=9.
GOAL_DISTANCE=5.
A_HEADINGS=(-60.,-45.,-30.,-15.,0.,15.,30.,45.,60.)
B_HEADINGS=(-90.,-60.,-30.,0.,30.,60.,90.)
PROFILE=os.environ.get('PUSH_FIXED_PROFILE','reference')
if PROFILE not in ('reference','eval_min'):
    raise ValueError('Unknown PUSH_FIXED_PROFILE: '+PROFILE)


def blueprint(initial):
    env=367;physical=int(initial['physical_indices'][env])
    boxes=initial['boxes'][env].clone();size=initial['sizes'][env].clone()
    if PROFILE=='eval_min':
        size[physical]=size.new_tensor([3.2,1.6,1.45])
    center=boxes[physical,:3].clone();yaw=yaw_from_quat(boxes[physical,3:7])
    q=boxes.new_tensor([0.,0.,0.,1.]);q[2]=(-yaw/2).sin();q[3]=(-yaw/2).cos()
    boxes[:,:3]=rotate(q.expand(len(boxes),4),boxes[:,:3]-center)
    boxes[:,2]+=size[physical,2]/2
    boxes[:,3:7]=multiply(q.expand(len(boxes),4),boxes[:,3:7]);boxes[:,7:]=0.
    direction=rotate(q,initial['center_goals'][env,0]-center)[:2]
    direction/=direction.norm()
    goal=boxes.new_tensor([0.,0.,float(size[physical,2]/2)])
    goal[:2]=direction*GOAL_DISTANCE
    xy=boxes.new_zeros(2,2)
    # A on the right lane can exert a left-turning moment while B is far away.
    xy[A]=boxes.new_tensor([float(size[physical,0]/4),-float(size[physical,1]/2)-A_GAP])
    xy[B]=boxes.new_tensor([-float(size[physical,0]/4),-float(size[physical,1]/2)-B_GAP])
    return dict(boxes=boxes,sizes=size,physical=physical,goal=goal,xy=xy,regions=torch.tensor([0,1]))


def install():
    from env.tasks.multi_agent.humanoid_ma_coupled_push import HumanoidMACoupledPush
    from utils import coupled_push_spec as spec
    from utils.stage1_platforms import reset_stage1_platforms
    initial=torch.load(REFERENCE,map_location='cpu',weights_only=False);plan=blueprint(initial)
    original_bank=spec.asset_bank

    def fixed_bank(count,env,device):
        native,mass=original_bank(count,env,device)
        fixed=plan['sizes'].to(device).expand_as(native).clone()
        return fixed,mass/native.prod(-1)*fixed.prod(-1)

    spec.asset_bank=fixed_bank

    def fixed_reset(task,ids):
        if not len(ids):return
        assert task.num_agents==2 and task.num_objects==4
        task._sample_episode_graph(ids)
        assert (task._agent_box_assignment[ids]==plan['physical']).all()
        task._reset_default_slots=None
        task._reset_ref_slots,task._reset_ref_motion_ids,task._reset_ref_motion_times={},{},{}
        task._reset_actors(ids)  # Native motion, reference time, joint pose and speed.
        old=task._humanoid_root_states[ids].clone()
        target=old[:,:,:3].clone();target[:,:,:2]=plan['xy'].to(task.device)[None]+task._env_origins[ids,None,:2]
        offset=old.new_zeros(len(ids),2)
        offset[:,A]=old.new_tensor(A_HEADINGS)[ids%len(A_HEADINGS)]
        offset[:,B]=old.new_tensor(B_HEADINGS)[(ids//len(A_HEADINGS))%len(B_HEADINGS)]
        wanted=math.pi/2+offset*math.pi/180
        angle=wanted-yaw_from_quat(old[:,:,3:7])
        q=old.new_zeros(len(ids),2,4);q[:,:,2]=(angle/2).sin();q[:,:,3]=(angle/2).cos()
        root=old.clone();root[:,:,:3]=target;root[:,:,3:7]=multiply(q,old[:,:,3:7])
        for start in (7,10):root[:,:,start:start+3]=rotate(q,old[:,:,start:start+3])
        body=task._kinematic_humanoid_rigid_body_states[ids].clone()
        expanded=q[:,:,None].expand(*body.shape[:-1],4)
        body[:,:,:,:3]=rotate(expanded,body[:,:,:,:3]-old[:,:,None,:3])+target[:,:,None]
        body[:,:,:,3:7]=multiply(expanded,body[:,:,:,3:7])
        for start in (7,10):body[:,:,:,start:start+3]=rotate(expanded,body[:,:,:,start:start+3])
        task._humanoid_root_states[ids]=root
        task._kinematic_humanoid_rigid_body_states[ids]=body
        boxes=plan['boxes'].to(task.device)[None].expand(len(ids),-1,-1).clone()
        boxes[:,:,:3]+=task._env_origins[ids,None]
        task._box_states[ids]=boxes
        task._regions[ids]=plan['regions'].to(task.device)
        goals=plan['goal'].to(task.device)[None,None]+task._env_origins[ids,None]
        task._center_tar_pos[ids]=goals
        assigned=task._assigned_box_values(task._box_states,ids)
        sizes=task._assigned_box_values(task._box_size,ids)
        tokens,_=spec.region_states(assigned,sizes,task._members[ids],task._regions[ids])
        task._tar_pos[ids]=goals+(tokens[:,:,:3]-assigned[:,:,:3]) if task._individual_push_goals else goals
        reset_stage1_platforms(task,ids)
        task._reset_env_tensors(ids);task._refresh_sim_tensors();task._reset_relation_history(ids)
        if task._coupled_peer_mode!='all':task._single_peer_selector.reset(ids)
        task._compute_observations(ids);task._init_amp_obs(ids)
        local=task._humanoid_root_states[ids,:,:2]-task._env_origins[ids,None,:2]
        max_error=float((local-plan['xy'].to(task.device)).abs().max())
        assert max_error<.001,max_error
        shared=task._box_states[ids,plan['physical'],:2]
        distance=(task._center_tar_pos[ids,0,:2]-shared).norm(dim=-1)
        assert (distance-GOAL_DISTANCE).abs().max()<.001
        torch.testing.assert_close(task._box_size[ids],plan['sizes'].to(task.device)[None].expand(len(ids),-1,-1),atol=0,rtol=0)
        if PROFILE=='eval_min':
            env_index=int(ids[0]);actor=task._box_handles[env_index*len(plan['sizes'])+plan['physical']]
            props=task.gym.get_actor_rigid_body_properties(task.envs[env_index],actor)
            shapes=task.gym.get_actor_rigid_shape_properties(task.envs[env_index],actor)
            assert abs(props[0].mass-22.272)<.001,props[0].mass
            assert all(abs(shape.friction-.4)<1e-6 for shape in shapes)
            print('[eval_min physics] mass_kg=',props[0].mass,'friction=',shapes[0].friction,flush=True)
        task._push_fixed_metadata=dict(profile=('eval_min_3p2_1p6_1p45_nearA_farB_goal5_v1' if PROFILE=='eval_min' else 'reference34_367_nearA_farB_goal5_v1'),source=str(REFERENCE),
            source_seed=34,source_env=367,A=A,B=B,a_surface_normal_gap_m=A_GAP,b_surface_normal_gap_m=B_GAP,
            goal_center_xy_distance_m=GOAL_DISTANCE,agent_xy=plan['xy'].tolist(),
            shared_box_size=plan['sizes'][plan['physical']].tolist(),
            a_heading_offsets_degrees=list(A_HEADINGS),b_heading_offsets_degrees=list(B_HEADINGS),
            pose_sampling='native loco RSI; joint poses and speeds retained; root headings varied',
            fixed_positions=True,heading_combinations=len(A_HEADINGS)*len(B_HEADINGS))
        print('[fixed PUSH layout]',task._push_fixed_metadata,flush=True)

    HumanoidMACoupledPush._reset_ontop_context_envs=fixed_reset
    return plan

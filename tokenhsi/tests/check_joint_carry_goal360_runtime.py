import json
import runpy
import sys
from pathlib import Path

import isaacgym
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.joint_carry_goals import invalid_goals
from utils.joint_carry_rsi import rotate_vectors, transform_states


def heading(q):
    x,y,z,w=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def check_states(actual, expected):
    torch.testing.assert_close(actual[...,:3],expected[...,:3],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(actual[...,7:],expected[...,7:],atol=1e-5,rtol=1e-5)
    # Recovered yaw is wrapped to [-pi, pi]; q and -q represent the same rotation.
    dot=(actual[...,3:7]*expected[...,3:7]).sum(-1).abs()
    torch.testing.assert_close(dot,torch.ones_like(dot),atol=1e-5,rtol=0)


context=runpy.run_path(str(ROOT/'tests/check_joint_carry_amp_runtime.py'))
task=context['task']
ids=task._joint_env_mask.nonzero().flatten()
original_weights=task._template_rsi_weights.clone()
original_preset=task._task_graph_preset
report=[]
try:
    for preset in ('joint_carry_at','joint_carry_ontop'):
        for skill in ('loco','pickUp','carryWith','putDown'):
            if skill=='putDown' and preset=='joint_carry_ontop':
                continue
            task._task_graph_preset=preset
            task._template_rsi_weights.zero_()
            index=task._skill.index(skill)
            task._template_rsi_weights[:,index]=1.
            task._reset_envs(ids)
            assert (task._joint_reset.skill_index[ids]==index).all()
            boxes=task._box_states[ids]
            goals=task._tar_pos[ids]
            torch.testing.assert_close(goals[:,0],goals[:,1],rtol=0,atol=0)
            yaw=heading(boxes[:,2,3:7])
            origins=task._env_origins[ids]
            local_boxes=boxes.clone();local_boxes[...,:3]-=origins[:,None]
            local_boxes=transform_states(local_boxes,-yaw,torch.zeros_like(origins))
            body=task._kinematic_humanoid_rigid_body_states[ids].clone()
            body[...,:3]-=origins[:,None,None]
            local_body=transform_states(body,-yaw,torch.zeros_like(origins))
            local_goal=rotate_vectors(goals[:,0]-origins,-yaw)
            delta=local_goal[:,:2]-local_boxes[:,0,:2]
            distance=delta.norm(dim=-1)
            item=dict(preset=preset,skill=skill,count=len(ids),distance_min=distance.min().item(),
                      distance_max=distance.max().item())
            if skill!='loco':
                pool=task._joint_reset.pools[skill]
                item['rsi_pool_size']=len(pool['source_frame'])
                item['sampled_source_frames']=task._joint_reset.source_frame[ids].unique().tolist()
                if skill=='pickUp' and context['cfg']['env']['jointCarryRsi'].get('dataset_sha256'):
                    assert pool['source_frame'].tolist()==list(range(100))
                    if context['cfg']['env']['jointCarryRsi'].get('pickupFrameSampling'):
                        frames=task._joint_reset.source_frame[ids]
                        grasp=(frames>=25)&(frames<=39)
                        lift=(frames>=55)&(frames<=80)
                        fractions=torch.stack([grasp.float().mean(),lift.float().mean(),
                                               (~(grasp|lift)).float().mean()])
                        torch.testing.assert_close(fractions,frames.new_tensor([.395,.428,.177],dtype=torch.float),
                                                   atol=.04,rtol=0)
                        item['pickup_frame_fractions']=fractions.tolist()
                    else:
                        assert item['sampled_source_frames']==list(range(100))
                rows=torch.searchsorted(pool['source_frame'],task._joint_reset.source_frame[ids])
                reference=pool['box_state'][rows,0]
                center=reference[:,:3].clone();center[:,2]=0
                root=pool['root_state'][rows].clone();root[...,:3]-=center[:,None]
                expected=transform_states(root,yaw,origins)
                check_states(task._humanoid_root_states[ids],expected)
                torch.testing.assert_close(task._dof_pos[ids],pool['dof_position'][rows],atol=1e-6,rtol=0)
                torch.testing.assert_close(task._dof_vel[ids],pool['dof_velocity'][rows],atol=1e-6,rtol=0)
                expected_box=reference.clone();expected_box[:,:3]-=center
                expected_box=transform_states(expected_box,yaw,origins)
                check_states(boxes[:,0],expected_box)
                item['paired_pose_and_velocities_preserved']=True
            if skill=='putDown':
                expected=pool['terminal_box_xy'][rows]-center[:,:2]
                torch.testing.assert_close(local_goal[:,:2],expected,atol=5e-5,rtol=1e-5)
                item['terminal_reference_goal_matches']=True
                radius=task._box_size[ids,:,:2].norm(dim=-1)/2
                obstacle_distance=(local_goal[:,None,:2]-local_boxes[:,1:,:2]).norm(dim=-1)
                assert (obstacle_distance>=radius[:,:1]+radius[:,1:]+.1-1e-5).all()
                item['terminal_goal_obstacles_clear']=True
            else:
                assert distance.min()>=2.2-1e-5 and distance.max()<=2.8+1e-5
                bins=((torch.atan2(delta[:,1],delta[:,0])+torch.pi)/(2*torch.pi)*8).long().clamp_max(7)
                counts=torch.bincount(bins,minlength=8)
                assert counts.min()>0
                item['relative_goal_octants']=counts.tolist()
                ontop=torch.full((len(ids),),preset=='joint_carry_ontop',device=task.device,dtype=torch.bool)
                assert not invalid_goals(local_goal[:,:2],local_boxes,task._box_size[ids],
                                         local_body[...,:3],ontop,.1).any()
                item['initial_goal_clearance_pass']=True
            if preset=='joint_carry_ontop':
                torch.testing.assert_close(boxes[:,2,:2],goals[:,0,:2],atol=5e-5,rtol=0)
                torch.testing.assert_close(boxes[:,2,2],task._box_size[ids,2,2]/2)
                assert boxes[:,2,7:].count_nonzero()==0
            assert torch.equal(task._hist_amp_obs_buf[ids],
                task._curr_amp_obs_buf[ids,:,None].expand_as(task._hist_amp_obs_buf[ids]))
            for _ in range(4):
                task.step(torch.zeros(task.num_envs*2,task.num_actions,device=task.device))
                assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
                assert torch.isfinite(task._amp_obs_buf).all()
            item['finite_physics_steps']=4
            report.append(item)
finally:
    task._template_rsi_weights.copy_(original_weights)
    task._task_graph_preset=original_preset
out=Path(context['args'].output_path)
(out/'goal360_runtime_check.json').write_text(json.dumps(report,indent=2)+'\n')
print('GOAL360_RUNTIME',json.dumps(report),flush=True)

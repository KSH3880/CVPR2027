"""Opt-in real simulator contract audit. Runs only on the explicit verification path."""
import json
from pathlib import Path
import torch
from env.tasks.multi_agent.joint_push_reward import push_face_geometry


def audit_runtime(task,path):
    ids=torch.arange(task.num_envs,device=task.device);task._reset_envs(ids)
    graph=task.relation_runtime.graph
    assert (task._push_binding==0).all()
    assert (task._agent_box_assignment==2).all()
    assert (graph.edge_src==torch.tensor([0,4,1,4],device=task.device)).all()
    assert (graph.edge_dst==torch.tensor([4,6,4,6],device=task.device)).all()
    assert (task._tar_pos[:,0]==task._tar_pos[:,1]).all()
    assert set(task._reset_ref_motion_ids)=={'loco'}
    assert torch.allclose(task._hist_amp_obs_buf,task._curr_amp_obs_buf[:,:,None].expand_as(task._hist_amp_obs_buf))
    center,normal,tangent,half,_=push_face_geometry(task._box_states[:,2],task._tar_pos[:,0],task._box_size[:,2],task._joint_push_spec['reward'])
    delta=task._tar_pos[:,0,:2]-task._box_states[:,2,:2]
    goal_cosine=(delta*normal).sum(-1)/delta.norm(dim=-1)
    if task._joint_push_spec.get('goal_mode')=='box_forward':
        assert torch.allclose(goal_cosine,torch.ones_like(goal_cosine),atol=1e-6)
    relative=task._humanoid_root_states[...,:2]-center[:,None,:2]
    depth=(relative*normal[:,None]).sum(-1);lateral=(relative*tangent[:,None]).sum(-1)
    if task._joint_push_spec.get("random_spawn",{}).get("mode")!="surround_independent":
        assert ((depth<=-1.)&(depth>=-2.)).all()
    random_spawn=task._joint_push_spec.get('random_spawn')
    root_gap=(task._humanoid_root_states[:,0,:2]-task._humanoid_root_states[:,1,:2]).norm(dim=-1)
    if random_spawn:
        from utils.door_reference import yaw_from_quat
        torch.testing.assert_close(depth,-task._push_spawn_depth,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(lateral,task._push_spawn_lateral,atol=1e-5,rtol=1e-5)
        heading=yaw_from_quat(task._humanoid_root_states[...,3:7])-torch.atan2(normal[:,1],normal[:,0])[:,None]
        error=heading-task._push_spawn_heading
        assert torch.atan2(error.sin(),error.cos()).abs().max()<1e-5
        assert root_gap.min()>=random_spawn['minimum_root_gap']-1e-5
    else:
        assert (lateral[:,0]*lateral[:,1]<0).all()
        assert root_gap.min()>=1.2
    sizes_before=task._box_size.clone()
    props=task.gym.get_actor_rigid_body_properties(task.envs[0],task._box_handles[2])
    mass=sum(x.mass for x in props)
    expected_mass=task._payload_masses[0].item() if hasattr(task,'_payload_masses') else task.cfg['env']['pushBox']['mass']
    assert abs(mass-expected_mass)<.001
    demo=task._fetch_conditioned_amp_demo(4096)
    labels=demo.reshape(4096,task._num_amp_obs_steps,-1)[...,-2:]
    assert (labels[...,0]==0).all() and (labels[...,1]==1).all()
    assert torch.isfinite(demo).all()
    saved_face=task._push_face_local_normal.clone()
    saved_spawn=task._push_spawn_lateral.clone() if random_spawn else None
    fixed=task._joint_push_spec.get('face_mode')=='reset_fixed'
    if fixed:
        geometry=task._joint_push_face_geometry()
        original_goal=task._tar_pos.clone()
        task._tar_pos[:,:,:2]=task._box_states[:,2,None,:2]-delta[:,None]
        changed=task._joint_push_face_geometry()
        for old,new in zip(geometry,changed):assert torch.equal(old,new)
        task._tar_pos.copy_(original_goal)
    before=task._humanoid_root_states[1:].clone();before_graph=graph.edge_src[1:].clone();before_goal=task._tar_pos[1:].clone()
    task._reset_envs(ids[:1])
    assert torch.equal(before,task._humanoid_root_states[1:])
    assert torch.equal(before_graph,graph.edge_src[1:]) and torch.equal(before_goal,task._tar_pos[1:])
    assert torch.equal(saved_face[1:],task._push_face_local_normal[1:])
    if random_spawn:assert torch.equal(saved_spawn[1:],task._push_spawn_lateral[1:])
    assert torch.equal(sizes_before,task._box_size)
    task._compute_reward(None)
    assert torch.isfinite(task.rew_buf).all()
    report=dict(num_envs=task.num_envs,shared_box=2,shared_goal=0,size=task.cfg['env']['pushBox']['size'],
        random_spawn_checked=bool(random_spawn),initial_min_root_gap=root_gap.min().item(),
        face_mode=task._joint_push_spec.get("face_mode","goal_dynamic"),fixed_face_goal_change_checked=fixed,
        measured_mass=mass,loco_only_rsi=True,repeat_reset_history=True,partial_reset_preserved=True,
        conditioned_push_amp_finite=True,goal_mode=task._joint_push_spec.get('goal_mode','goal360'),
        initial_goal_face_cosine_min=goal_cosine.min().item(),spawn_depth_range=[depth.min().item(),depth.max().item()],
        min_root_gap=(task._humanoid_root_states[:,0,:2]-task._humanoid_root_states[:,1,:2]).norm(dim=-1).min().item())
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2)+'\n')
    print('[Joint Push runtime audit]',report,flush=True)

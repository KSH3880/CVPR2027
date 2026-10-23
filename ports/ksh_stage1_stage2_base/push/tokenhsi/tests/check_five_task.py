"""Small evaluation simulation audit; training environment count is not changed."""
from isaacgym import gymapi, gymtorch
from pathlib import Path
import json
import os
import math
import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from utils.task_role_spec import task_graph_packet
from utils.edge_context_spec import PUSH_CONTACT
from env.tasks.multi_agent.humanoid_ma_five_task import HumanoidMAFiveTask

args = get_args()
cfg, train, _ = load_cfg(args)
cfg['args'] = args
cfg['env']['motion_file'] = args.motion_file
sim = parse_sim_params(args,cfg,train)
set_seed(42)
torch.set_num_threads(1)
task_class = HumanoidMAFiveTask
if args.task == 'HumanoidMAFiveTaskStage2':
    from env.tasks.multi_agent.humanoid_ma_five_task_stage2 import HumanoidMAFiveTaskStage2
    task_class = HumanoidMAFiveTaskStage2
rsi_audit = os.environ.get('FIVE_TASK_PUSH_RSI_AUDIT') == '1'
if rsi_audit:
    # Audit PUSH training RSI without rebuilding the original tasks' PD caches.
    # Those four tasks explicitly fall back to loco in this audit only.
    from env.tasks.multi_agent.humanoid_ma_shared_support import LocoRsi
    class PushRsiAuditTask(HumanoidMAFiveTask):
        _solo_cache_class = LocoRsi
    task_class = PushRsiAuditTask
    args.eval = False
task = task_class(cfg,sim,args.physics_engine,args.device,args.device_id,True)
try:
    from utils.five_task import push_assets
    push_mask = push_assets(task._box_size, cfg['env']).flatten().cpu().tolist()
    push_actor_count = 0
    for index, is_push in enumerate(push_mask):
        if not is_push:
            continue
        actor = task._box_handles[index]
        env = task.envs[index//task.num_physical_objects]
        masses = task.gym.get_actor_rigid_body_properties(env, actor)
        friction = task.gym.get_actor_rigid_shape_properties(env, actor)
        assert abs(sum(body.mass for body in masses)-cfg['env']['pushBox']['mass']) < 1e-4
        assert all(abs(shape.friction-cfg['env']['pushBox']['friction']) < 1e-6 for shape in friction)
        push_actor_count += 1
    ids = torch.arange(task.num_envs,device=task.device)
    seen = torch.zeros(5,device=task.device,dtype=torch.long)
    push_samples = push_rsi = 0
    scenes=[]
    loco_cases={}
    gaps=[]
    initial_goal_distances=[]
    bearings=[]
    pd_cases=pd_falls=pd_nonfinite=0
    random_spawn=cfg['env']['pushTask'].get('loco_spawn_mode') == 'random_azimuth_surface_clearance_v1'
    def record(row,owner,label,clip=-1,time=0.):
        body=task._kinematic_humanoid_rigid_body_states[row,owner].clone()
        box=task._box_states[row,owner].clone()
        origin=box[:3].clone();origin[2]=0
        body[:,:3]-=origin
        all_boxes=task._box_states[row].clone();all_boxes[:,:3]-=origin
        goals=task._tar_pos[row].clone()-origin
        roots=task._kinematic_humanoid_rigid_body_states[row,:,0,:3].clone()-origin
        return dict(label=label,clip=int(clip),time=float(time),body=body.cpu().tolist(),
                    box=all_boxes[owner].cpu().tolist(),size=task._box_size[row,owner].cpu().tolist(),
                    goal=goals[owner].cpu().tolist(),all_boxes=all_boxes.cpu().tolist(),
                    all_sizes=task._box_size[row].cpu().tolist(),all_goals=goals.cpu().tolist(),
                    human_roots=roots.cpu().tolist())
    for _ in range(32 if rsi_audit else 4):
        task._reset_envs(ids)
        packet = task_graph_packet(task.relation_runtime.graph,len(ids)).reshape(len(ids),task.num_agents,5)
        seen += torch.bincount(packet[...,1].long().flatten(),minlength=5)
        assert task.obs_buf.shape[1] == (1420 if task._stage2 else 634)
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
        from utils.unified_training import amp_family_ids
        assert task._amp_obs_buf.shape[-1] == 133
        labels=amp_family_ids(task._amp_obs_buf.flatten(0,1).flatten(1),task._num_amp_obs_steps,task._amp_num_families)
        assert ((labels>=0)&(labels<=3)).all()
        if rsi_audit:
            mask=task._push_rsi_interaction
            r,a=mask.nonzero(as_tuple=True)
            push_samples+=int((packet[...,1]==4).sum())
            push_rsi+=len(r)
            if len(r):
                clips=task._push_rsi_samples['motion_id'][r,a]
                times=task._push_rsi_samples['time'][r,a]
                native=task._motion_lib['push'].get_motion_state(clips,times)
                torch.testing.assert_close(task._dof_pos[r,a],native[2],atol=1e-6,rtol=0)
                torch.testing.assert_close(task._dof_vel[r,a],native[5],atol=1e-6,rtol=0)
                body=task._kinematic_humanoid_rigid_body_states[r,a]
                assert (body[:,task._key_body_ids[:2],2].mean(-1)<=1.+1e-6).all()
                assert not task._push_reset_geometry().invalid(body[...,:3],body[...,3:7],task._box_states[r,a],task._box_size[r,a]).any()
                for row,owner,clip,time in zip(r.tolist(),a.tolist(),clips.tolist(),times.tolist()):
                    scenes.append(record(row,owner,'selected_push_rsi',clip,time))
            if random_spawn:
                from utils.push_random_spawn import body_clearance,unsafe_initial_rows
                from utils.door_reference import rotate
                rr,aa=(packet[...,1]==4).nonzero(as_tuple=True)
                progress=torch.where(mask[rr,aa],task._push_rsi_samples['box_progress'][rr,aa],0.)
                actual=(task._tar_pos[rr,aa]-task._box_states[rr,aa,:3]).norm(dim=-1)
                if cfg['env']['pushGoalRandomization'].get('distance_reference') != 'reset_box_center':
                    actual=actual/(1-progress)
                assert actual.min()>=.998 and actual.max()<=5.002
                initial_goal_distances.extend(actual.cpu().tolist())
                lr,la=((packet[...,1]==4)&~mask).nonzero(as_tuple=True)
                pose=task._kinematic_humanoid_rigid_body_states[lr,la]
                geometry=task._push_reset_geometry()
                gap=body_clearance(geometry,pose[...,:3],pose[...,3:7],task._box_states[lr,la],task._box_size[lr,la])
                assert gap.min()>=.998 and gap.max()<=5.002
                assert not unsafe_initial_rows(geometry,pose,task._box_states[lr],task._box_size[lr],torch.full_like(gap,.2)).any()
                gaps.extend(gap.cpu().tolist())
                inverse=task._box_states[lr,la,3:7].clone();inverse[:,:3]*=-1
                local=rotate(inverse,pose[:,0,:3]-task._box_states[lr,la,:3])
                angle=torch.atan2(local[:,1],local[:,0])
                bearings.extend(angle.cpu().tolist())
                for index,(row,owner) in enumerate(zip(lr.tolist(),la.tolist())):
                    sector=int(((float(angle[index])+2*math.pi)%(2*math.pi))/(math.pi/4))
                    if sector not in loco_cases:
                        label=f'loco_360_sector_{sector}_gap_{float(gap[index]):.2f}m'
                        loco_cases[sector]=record(row,owner,label)
            if len(r):
                # Representative 0.1s PD holds of the actual accepted RSI resets.
                target=task._dof_pos.contiguous().clone()
                task.gym.set_dof_position_target_tensor(task.sim,gymtorch.unwrap_tensor(target))
                fall=torch.zeros(len(r),device=task.device,dtype=torch.bool)
                finite=torch.ones_like(fall)
                for _ in range(math.ceil(.1/task.sim_params.dt)):
                    task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True);task._refresh_sim_tensors()
                    body_now=task._rigid_body_pos[r,a]
                    low=body_now[...,2]<task._termination_heights
                    low[:,task._contact_body_ids]=False
                    force=task._contact_forces[r,a].clone();force[:,task._contact_body_ids]=0
                    fall|=low.any(-1)&(force.abs().sum(-1)>.1).any(-1)
                    finite&=torch.isfinite(task._humanoid_root_states[r,a]).all(-1)
                pd_cases+=len(r);pd_falls+=int(fall.sum());pd_nonfinite+=int((~finite).sum())
    assert (seen>0).all(),seen
    before = [value[1:].clone() for value in (task._box_states, task._tar_pos,task._humanoid_root_states,task._amp_obs_buf)]
    task._reset_envs(ids[:1])
    for value, expected in zip((task._box_states,task._tar_pos,task._humanoid_root_states,task._amp_obs_buf),before):
        torch.testing.assert_close(value[1:],expected,atol=0,rtol=0)
    # Gym applies root setters on simulate: finish the reset-only audit with one
    # full reset so an earlier pending full reset is not replaced by the partial one.
    task._reset_envs(ids)
    phi,diag = task._evaluate_ontop_context()
    graph = task.relation_runtime.graph
    assert (graph.edge_relation == PUSH_CONTACT).any()
    for _ in range(12):
        task.pre_physics_step(torch.zeros(task.num_envs*task.num_agents,task.num_actions,device=task.device))
        task._physics_step()
        task.post_physics_step()
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all()
    metrics = task.consume_relation_diagnostics()
    required_metrics = ('edge/push_contact/phi_raw', 'edge/push_contact/own_success',
                        'push/contact', 'push/at', 'push/contact_and_at',
                        'push/contact_travel_m_per_step', 'push/contact_goal_progress_m_per_step',
                        'push/goal_progress_m_per_step', 'push/contact_tick_fraction')
    assert all(key in metrics for key in required_metrics), metrics.keys()
    assert all(torch.isfinite(value).all() for value in metrics.values())
    assert 0 <= metrics['push/contact'] <= 1 and 0 <= metrics['push/at'] <= 1
    assert metrics['push/travel_m_per_step'] < 1, 'Reset coordinate jump counted as motion'
    assert metrics['push/goal_distance_m'] < 10, 'Pending reset did not reach simulator'
    metric_before = task._push_motion_metrics.episode[1:].clone()
    task._reset_envs(ids[:1])
    assert not task._push_motion_metrics.episode[:1].any()
    torch.testing.assert_close(task._push_motion_metrics.episode[1:], metric_before, atol=0, rtol=0)
    demo = task.fetch_amp_obs_demo(1024)
    assert torch.isfinite(demo).all()
    labels = amp_family_ids(demo,task._num_amp_obs_steps,task._amp_num_families)
    assert torch.unique(labels).numel() == 4
    report = dict(task_ids=list(range(5)),counts=seen.cpu().tolist(),observation_width=task.obs_buf.shape[1],
                  amp_frame_width=task._num_amp_obs_per_step,partial_reset=True,finite_steps=12,
                  push_actor_count=push_actor_count,push_mass_kg=cfg['env']['pushBox']['mass'],
                  push_friction=cfg['env']['pushBox']['friction'],
                  demo_family_counts=torch.bincount(labels,minlength=4).cpu().tolist())
    report['push_metrics'] = {key: float(metrics[key]) for key in required_metrics}
    report['push_metrics_partial_reset'] = True
    if rsi_audit:
        assert .4 < push_rsi/max(push_samples,1) < .6
        report.update(push_reset_samples=push_samples,push_rsi_samples=push_rsi,push_rsi_fraction=push_rsi/push_samples,
                      original_tasks_audit_loco_only=True,native_push_dof_preserved=True,hand_height_le_1m=True)
        report.update(rsi_pd_seconds=.1,rsi_pd_agent_cases=pd_cases,rsi_pd_falls=pd_falls,rsi_pd_nonfinite=pd_nonfinite)
        assert pd_falls==0 and pd_nonfinite==0
        if random_spawn:
            assert len(loco_cases)==8
            report.update(loco_surface_gap_m=[min(gaps),max(gaps)],initial_goal_distance_m=[min(initial_goal_distances),max(initial_goal_distances)],
                          loco_azimuth_sectors=len(loco_cases),loco_samples=len(gaps),initial_box_overlaps=0)
    out=Path(args.output_path)
    out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(report,indent=2))
    if scenes:
        skeleton=task._motion_lib['push']._motions[0].skeleton_tree
        common=dict(names=list(skeleton.node_names),parents=skeleton.parent_indices.cpu().tolist())
        (out/'timeline.json').write_text(json.dumps(dict(**common,scenes=scenes,sampled_resets=True)))
        (out/'scenes.json').write_text(json.dumps(dict(**common,scenes=[loco_cases[k] for k in sorted(loco_cases)])))
    print('FIVE_TASK_AUDIT',json.dumps(report),flush=True)
finally:
    task.gym.destroy_sim(task.sim)

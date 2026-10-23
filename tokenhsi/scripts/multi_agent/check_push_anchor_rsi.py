"""Build a dedicated PD-screened low-point bank and audit actual anchor resets."""
from isaacgym import gymapi
import hashlib,json,os,subprocess
from pathlib import Path
import torch
from utils.config import get_args,load_cfg,parse_sim_params,set_seed
from utils.open_push_rsi import bank_contract,OpenPushRSIBank
from env.tasks.multi_agent.humanoid_ma_open_push_anchor import HumanoidMAOpenPushAnchor
from env.tasks.multi_agent.humanoid_ma_open_push_point import HumanoidMAOpenPushPoint
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from utils.push_point import point_geometry

torch.set_num_threads(1)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
uuid=subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert os.environ.get('TOKENHSI_GPU') in ('6',uuid)
args=get_args();assert args.test and args.eval and args.headless and args.num_envs==64
cfg,train,_=load_cfg(args);cfg['args']=args;cfg['env']['motion_file']=args.motion_file
cfg['env']['enableTrackInitState']=True;cfg['_rsi_build']=True
source=Path('tokenhsi/data/push_fixed130_fixed_face_midpoint_goal3to5_bank.json')
source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
proposals=json.loads(source.read_text())['candidates']['push']
out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
sim=parse_sim_params(args,cfg,train);sim.physx.num_threads=1;set_seed(42)
point_reward_mode=args.task=='HumanoidMAOpenPushPoint'
task_type=HumanoidMAOpenPushPoint if point_reward_mode else HumanoidMAOpenPushAnchor
task=task_type(cfg,sim,args.physics_engine,args.device,args.device_id,True)
ids=torch.arange(64,device=task.device);batch=32
try:
    assert task.get_obs_size()==570 and task.get_clean_object_obs_size()==30 and task.get_num_amp_obs()==1310
    screen_heights=[.6175] if point_reward_mode else [.585,.6175,.650]
    accepted=[];rejected=[];initial_distances=[];begin=0;extra_added=False
    base_count=len(proposals)
    geometry=task._joint_rsi_bank.rows['push']
    if os.environ.get('ANCHOR_RSI_RETRY_MISSING')=='1':
        rejected=json.loads((out/'rejected.json').read_text())
        failed={r['index'] for r in rejected}
        accepted=[r for i,r in enumerate(proposals) if i not in failed]
        missing={mid for mid in range(task._motion_lib['push'].num_motions()) if not any(r['motion_id']==mid for r in accepted)}
        signatures={(r['motion_id'],round(r['time'],6)) for r in proposals}
        proposals=proposals+[r for r in geometry if r['motion_id'] in missing and (r['motion_id'],round(r['time'],6)) not in signatures]
        begin=base_count;extra_added=True
        print('ANCHOR_RSI_EXTRA_GEOMETRY',sorted(missing),len(proposals)-base_count,flush=True)
    # Reward-only midpoint removal does not change the screened reference poses.
    # Keep the original bank immutable and copy only after matching all geometry,
    # motion hashes, retargeting, scene and height settings.
    bank_path=Path(cfg['env']['openPushRSI']['bank_file'])
    previous_bank=Path('tokenhsi/data/push_fixed130_anchor_goal3to5_bank.json')
    if not bank_path.exists() and previous_bank.exists() and os.environ.get('ANCHOR_RSI_RESCREEN')!='1':
        donor=json.loads(previous_bank.read_text())
        import copy
        previous_contract=copy.deepcopy(donor['contract'])
        for key in ('z_tolerance_m','z_reward_sigma_m'):
            previous_contract['push_contact_point'].pop(key)
        if point_reward_mode:
            fixed=cfg['env']['pushContactPoint']['height_range_m']
            assert fixed[0]==fixed[1] and fixed[0] in donor['point_screen']['screen_heights_m']
            assert previous_contract['push_contact_point']['height_range_m'][0]<=fixed[0]<=previous_contract['push_contact_point']['height_range_m'][1]
            previous_contract['push_contact_point']['height_range_m']=fixed
        assert previous_contract==bank_contract(cfg['env']), 'Previous anchor screen geometry differs'
        assert donor['point_screen']['source_sha256']==source_hash
        donor['contract']=previous_contract
        donor['reused_pd_screen']=dict(bank=str(previous_bank),sha256=hashlib.sha256(previous_bank.read_bytes()).hexdigest(),
            reason=('Fixed height is one of the donor screened heights; same geometry/retargeting/motions; reward-only variant'
                    if point_reward_mode else 'Identical geometry/retargeting/heights; remove reward/contact midpoint gates only'))
        bank_path.write_text(json.dumps(donor,indent=2)+'\n')
    reusing_bank=bank_path.exists() and os.environ.get('ANCHOR_RSI_RESCREEN')!='1'
    if reusing_bank:
        payload=json.loads(Path(cfg['env']['openPushRSI']['bank_file']).read_text())
        assert payload['contract']==bank_contract(cfg['env']) and payload['point_screen']['source_sha256']==source_hash
        accepted=payload['candidates']['push'];begin=len(proposals);extra_added=True
    while begin<len(proposals):
        rows=proposals[begin:begin+batch];padded=rows+[rows[-1]]*(batch-len(rows))
        verdict=torch.ones(batch,dtype=torch.bool,device=task.device)
        distances=torch.zeros(batch,device=task.device)
        for height in screen_heights:
            task._point_height_override=height;task._reset_envs(ids)
            sample={k:torch.tensor([r.get(k,0.) for r in padded],device=task.device).repeat_interleave(2)
                    for k in ('motion_id','time','phase','box_progress')}
            task._apply_joint_rsi(ids,torch.zeros_like(ids),'push',sample)
            task._reset_task(ids);task._reset_env_tensors(ids);task._sync_door_reset_states(ids)
            task._compute_observations(ids);task._init_amp_obs(ids)
            initial=task._humanoid_root_states[:,0,:3].clone()
            def pair_all(x):return x.reshape(batch,2,-1).all(-1).all(-1)
            def hand_distance(reference=False):
                hands=(task._kinematic_humanoid_rigid_body_states[...,:3] if reference else task._rigid_body_pos)[:,:,task._key_body_ids[:2]]
                point,_,_,normal=point_geometry(hands,task._owner_push_boxes(),task._owner_push_sizes(),
                    task.relation_runtime.face_side,task.relation_runtime.point_local)
                tangent=torch.stack((normal[...,1],-normal[...,0],torch.zeros_like(normal[...,0])),-1)
                targets=point[:,:,None]+tangent[:,:,None]*hands.new_tensor([.2,-.2])[None,None,:,None]
                return (hands-targets).norm(dim=-1)[:,0]
            d=hand_distance(True).amax(-1)
            distances=torch.maximum(distances,d.reshape(batch,2).amax(-1))
            valid=pair_all(d<.18);support=torch.zeros_like(valid);near=torch.zeros_like(valid)
            actions=torch.zeros(128,task.num_actions,device=task.device)
            for step in range(12):
                task.pre_physics_step(actions);task._physics_step();task.post_physics_step()
                valid &= pair_all(torch.isfinite(task.obs_buf)) & pair_all(torch.isfinite(task.rew_buf))
                valid &= pair_all(task._humanoid_root_states[:,0,2]>.65) & pair_all(task.reset_buf==0)
                valid &= pair_all((task._humanoid_root_states[:,0,:3]-initial).norm(dim=-1)<.45)
                valid &= pair_all((task._owner_push_boxes()[:,0,2]-.65).abs()<.08)
                if step>=2:
                    feet=task._key_body_ids[2:]
                    supported=(task._rigid_body_pos[:,0,feet,2]<.35)&(task._contact_forces[:,0,feet,2]>=1.)
                    support |= pair_all(supported.any(-1))
                near |= pair_all(hand_distance().amax(-1)<.18)
            verdict &= valid&support&near
        flags=verdict.cpu().tolist();ds=distances.cpu().tolist()
        for i,row in enumerate(rows):
            if flags[i]:accepted.append(row);initial_distances.append(ds[i])
            else:rejected.append(dict(index=begin+i,**row))
        print('ANCHOR_RSI_SCREEN',begin,len(proposals),'accepted',len(accepted),flush=True)
        begin+=len(rows)
        if begin==len(proposals) and not extra_added:
            extra_added=True
            missing={mid for mid in range(task._motion_lib['push'].num_motions()) if not any(r['motion_id']==mid for r in accepted)}
            signatures={(r['motion_id'],round(r['time'],6)) for r in proposals}
            proposals.extend(r for r in geometry if r['motion_id'] in missing and (r['motion_id'],round(r['time'],6)) not in signatures)
            print('ANCHOR_RSI_EXTRA_GEOMETRY',sorted(missing),len(proposals)-base_count,flush=True)
    if not reusing_bank:(out/'rejected.json').write_text(json.dumps(rejected,indent=2)+'\n')
    counts={str(mid):sum(r['motion_id']==mid for r in accepted) for mid in range(task._motion_lib['push'].num_motions())}
    if not reusing_bank:(out/'screen_counts.json').write_text(json.dumps(dict(proposed=len(proposals),accepted=len(accepted),per_clip=counts),indent=2)+'\n')
    assert accepted,'No valid low-point RSI poses'
    contract=bank_contract(cfg['env'])
    if not Path(cfg['env']['openPushRSI']['bank_file']).exists() or os.environ.get('ANCHOR_RSI_RESCREEN')=='1':
      payload=dict(contract=contract,screen='PD_12_steps',candidates={'push':accepted},
        point_screen=dict(source_bank=str(source),source_sha256=source_hash,screen_heights_m=screen_heights,
            scenes_per_candidate_per_height=2,control_steps=12,dt_s=float(task.dt),
            criteria='Both initial hands within0.18m of height-specific reference targets; finite; root>0.65m; drift<0.45m; boxheight within0.08m; foot support and hand proximity observed in both independent scenes at each height.',
            source_candidates=base_count,extra_geometry_candidates=len(proposals)-base_count,
            excluded_rsi_motion_ids=[int(mid) for mid,count in counts.items() if not count],amp_expert_clips_unchanged=21,
            limitations='Three sampled heights, finite short PD screen; not continuous-height proof, long-term policy success or 5degree guarantee. RSI excludes clips with no surviving poses; all AMP experts retained.'))
    bank_path=Path(cfg['env']['openPushRSI']['bank_file'])
    if bank_path.exists():
        if json.loads(bank_path.read_text())!=payload:raise ValueError('Existing dedicated bank differs; preserve it')
    else:bank_path.write_text(json.dumps(payload,indent=2)+'\n')
    digest=hashlib.sha256(bank_path.read_bytes()).hexdigest()
    task._joint_rsi_bank=OpenPushRSIBank(cfg['env']['openPushRSI'],task._motion_lib,contract,{'push':accepted},active_skills=('push',),allow_missing_clips=True)
    task._point_height_override=None;task._is_eval=False;task._skill=list(cfg['env']['skill'])
    task._skill_init_prob=torch.tensor(cfg['env']['skillInitProb'],device=task.device)
    heights=[];goal_distances=[];hand_errors=[];interaction=late=0
    for _ in range(256):
        task._reset_envs(ids)
        target=task.contact_target_points();selected=task._owner_push_values(task._point_heights_physical)
        torch.testing.assert_close(target[...,2],selected,atol=1e-5,rtol=0)
        lo,hi=cfg['env']['pushContactPoint']['height_range_m']
        assert (selected>=lo).all() and (selected<=hi).all()
        heights.extend(selected.flatten().cpu().tolist())
        distance=(task._tar_pos[...,:2]-task._owner_push_values(task._push_reset_states)[...,:2]).norm(dim=-1)
        assert (distance>=3-1e-5).all() and (distance<=5+1e-5).all()
        goal_distances.extend(distance.flatten().cpu().tolist())
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task._amp_obs_buf).all()
        assert not task.relation_runtime.done.any()
        interaction+=int(task._rsi_interaction.sum());late+=int(task._rsi_late.sum())
        ref=task._kinematic_humanoid_rigid_body_states[:,:,task._key_body_ids[:2],:3]
        _,error,_,_=point_geometry(ref,task._owner_push_boxes(),task._owner_push_sizes(),task.relation_runtime.face_side,task.relation_runtime.point_local)
        hand_errors.extend(error[...,2].abs()[task._rsi_interaction].cpu().tolist())
    assert .47<interaction/32768<.53 and .65<late/interaction<.75
    # Recompute the raw center-based scene, then verify pose/velocity mapping.
    raw=HumanoidMAOpenPush._compute_clean_scene_obs(task)
    anchored=task._compute_clean_scene_obs();h,o,g=task.get_scene_entity_sizes();end=2*h+2*o+2*g
    poses=anchored[:,end:end+42].reshape(64,6,7)
    expected_position=task._owner_push_boxes()[...,:3] if point_reward_mode else task.contact_target_points()
    torch.testing.assert_close(poses[:,2:4,:3],expected_position,atol=1e-5,rtol=0)
    raw_o=raw[:,2*h:2*h+60].reshape(64,2,30);anchor_o=anchored[:,2*h:2*h+60].reshape(64,2,30)
    torch.testing.assert_close(anchor_o[...,6:],raw_o[...,6:],rtol=0,atol=0)
    expected_velocity=raw_o[...,:3] if point_reward_mode else raw_o[...,:3]+torch.cross(raw_o[...,3:6],task.relation_runtime.point_local,dim=-1)
    torch.testing.assert_close(anchor_o[...,:3],expected_velocity)
    if point_reward_mode:torch.testing.assert_close(anchored,raw,rtol=0,atol=0)
    old=task.relation_runtime.point_local.clone()
    task.relation_runtime.point_local[0,0,2]+=.01
    changed=task._compute_clean_scene_obs()
    if point_reward_mode:torch.testing.assert_close(anchored,changed,rtol=0,atol=0)
    else:assert not torch.equal(anchored[0],changed[0])
    torch.testing.assert_close(anchored[1:],changed[1:],rtol=0,atol=0)
    task.relation_runtime.point_local[:]=old
    saved=[x[1:].clone() for x in (task.relation_runtime.point_local,task.relation_runtime.face_side,task._point_heights_physical,task._humanoid_root_states,task._amp_obs_buf)]
    task._reset_envs(ids[:1])
    for before,after in zip(saved,(task.relation_runtime.point_local,task.relation_runtime.face_side,task._point_heights_physical,task._humanoid_root_states,task._amp_obs_buf)):
        torch.testing.assert_close(before,after[1:],rtol=0,atol=0)
    local=task.relation_runtime.point_local.clone();goals=task._tar_pos.clone()
    for _ in range(12):
        task.pre_physics_step(torch.zeros(128,task.num_actions,device=task.device));task._physics_step();task.post_physics_step()
        assert torch.isfinite(task.obs_buf).all() and torch.isfinite(task.rew_buf).all() and torch.isfinite(task._amp_obs_buf).all()
    torch.testing.assert_close(task.relation_runtime.point_local,local,rtol=0,atol=0)
    torch.testing.assert_close(task._tar_pos,goals,rtol=0,atol=0)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash and hashlib.sha256(bank_path.read_bytes()).hexdigest()==digest
    report=dict(training=False,gpu=6,policy_width=570,object_width=30,amp_width=1310,
        source_candidates=payload['point_screen']['source_candidates'],extra_geometry_candidates=payload['point_screen']['extra_geometry_candidates'],
        accepted_candidates=len(accepted),screen_heights_m=payload['point_screen']['screen_heights_m'],
        excluded_rsi_motion_ids=[int(mid) for mid,count in counts.items() if not count],amp_expert_clips_unchanged=21,
        pd_candidate_scene_height_checks=(payload['point_screen']['source_candidates']+payload['point_screen']['extra_geometry_candidates'])*2*len(payload['point_screen']['screen_heights_m']),
        screened_initial_hand_target_distance_bound_m=.18,
        reset_agent_samples=32768,anchor_height_m_range=[min(heights),max(heights)],anchor_height_m_mean=sum(heights)/len(heights),
        goal_distance_m_range=[min(goal_distances),max(goal_distances)],rsi_fraction=interaction/32768,late_fraction=late/interaction,
        initial_rsi_midpoint_z_error_max_m=max(hand_errors),anchor_pose_observation_verified=not point_reward_mode,anchor_velocity_verified=not point_reward_mode,
        center_pose_observation_verified=point_reward_mode,original_observation_unchanged=point_reward_mode,
        physical_bbox_unchanged=True,point_only_change_changes_observation=not point_reward_mode,partial_reset_preserved=True,
        midpoint_reward_enabled=point_reward_mode,midpoint_contact_gate_enabled=point_reward_mode,
        reused_pd_screen=payload.get('reused_pd_screen'),
        height_alignment_screen=payload.get('height_alignment_screen'),
        finite_simulation_steps=12,local_anchor_fixed_during_rollout=True,at_goal_fixed_during_rollout=True,
        source_bank_unchanged=True,target_bank_unchanged=True,bank_sha256=digest)
    if point_reward_mode and 'height_band_m' in cfg['env']['pushPointReward']:
        lo,hi=cfg['env']['pushPointReward']['height_band_m']
        report.update(reward_height_band_m=[lo,hi],
            initial_height_measurement="Retargeted kinematic reference before physics, not a contact-success requirement",
            initial_rsi_reference_midpoint_height_in_band_fraction=sum(v<=(hi-lo)/2+1e-6 for v in hand_errors)/len(hand_errors))
        if 'height_alignment_screen' in payload:
            assert report['initial_rsi_reference_midpoint_height_in_band_fraction']==1.
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
finally:task.gym.destroy_sim(task.sim)

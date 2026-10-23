"""Actual humanoid PD contact sweep; no policy or external box forces.

Analysis-only process-local overrides extend immutable cube assets to 1--1.5m.
Screened source poses initialize the actor, but modified heights are NOT RSI
bank validation. Wrist net reactions are proxies, not pair-resolved sensors.
"""
from isaacgym import gymapi, gymtorch
import csv
import hashlib
import itertools
import json
import os
import subprocess
from pathlib import Path
import torch
from utils.config import get_args, load_cfg, parse_sim_params, set_seed
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
import utils.open_push_sizes as size_module

SIZES = [round(1+i*.05, 2) for i in range(11)]
HEIGHTS = [round(.4+i*.1, 2) for i in range(10)]
LOADS = [100., 200., 350., 500.]
POSES = [0, 5, 10, 15]
CASES = [(s,h,f,p) for s,h,f,p in itertools.product(SIZES,HEIGHTS,LOADS,POSES)
         if h <= s-.05]
if os.environ.get('HEIGHT_PD_QUICK') == '1':
    CASES = [(1.3,h,f,p) for h,f,p in itertools.product([.5,.7,.9,1.1],[200.,500.],[5,10])]

original_sizes = size_module.size_values
size_module.size_values = lambda env: SIZES if env.get('_height_analysis') else original_sizes(env)
original_build = HumanoidMAOpenPush._build_box


def build(task, e, b, ptr):
    if b >= 2:
        side = CASES[e][0]
        task._push_size_indices[e,b-2] = SIZES.index(side)
        task._box_size[e,b] = side
    original_build(task,e,b,ptr)


HumanoidMAOpenPush._build_box = build
torch.set_num_threads(1)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
args = get_args()
assert args.test and args.eval and args.headless
physical_uuid = subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert os.environ.get('TOKENHSI_GPU') in ('6',physical_uuid)
args.num_envs = len(CASES)
cfg, train, _ = load_cfg(args)
cfg['args'] = args
cfg['env'].update(numEnvs=len(CASES),motion_file=args.motion_file,
    enableTrackInitState=True,_height_analysis=True,
    pushSizeRandomization=dict(side_range=[1.,1.5],num_sizes=11,sampling='uniform_creation'))
cfg['env']['pushSceneRandomization'].update(arena_radius=5.,min_station_distance=8.)
cfg['env']['envSpacing'] = 20
cfg['_rsi_build'] = True
bank_path = Path(cfg['env']['openPushRSI']['bank_file'])
bank_hash = hashlib.sha256(bank_path.read_bytes()).hexdigest()
bank = json.loads(bank_path.read_text())['candidates']['push']
sim = parse_sim_params(args,cfg,train)
sim.physx.num_threads = 1
set_seed(42)
task = HumanoidMAOpenPush(cfg,sim,args.physics_engine,args.device,args.device_id,True)
out = Path(args.output_path)
out.mkdir(parents=True,exist_ok=True)
try:
    n = len(CASES)
    ids = torch.arange(n,device=task.device)
    settings = torch.tensor([row[:3] for row in CASES],device=task.device)
    sides, heights, loads = settings.unbind(-1)
    task._reset_envs(ids)
    task._agent_lanes[:] = torch.tensor([0,1],device=task.device)
    task._push_binding[:] = task._agent_lanes
    task._box_states[:,2:] = 0
    task._box_states[:,2:,2] = sides[:,None]/2
    task._box_states[:,2:,6] = 1
    task._box_states[:,2:,0] = -4
    task._box_states[:,2,1] = -5
    task._box_states[:,3,0] = -12
    task._box_states[:,3,1] = 5
    task._push_reset_states[:] = task._box_states[:,2:]
    task._push_directions[:] = task._push_directions.new_tensor([0.,1.,0.])
    task._push_goal_offsets[:] = task._push_directions*2
    task.relation_runtime.reset(ids,torch.ones(n,2,device=task.device,dtype=torch.long),task._owner_push_boxes())
    task._sample_box_assignments(ids)
    selected = {}
    for mid in POSES:
        rows = [r for r in bank if r['motion_id']==mid]
        selected[mid] = rows[len(rows)//2]
    for height in sorted({row[1] for row in CASES}):
        chosen = torch.tensor([i for i,row in enumerate(CASES) if row[1]==height],device=task.device)
        sample = {k:torch.tensor([selected[CASES[i][3]].get(k,0.) for i in chosen.cpu().tolist()],device=task.device)
                  for k in ('motion_id','time','phase','angle','box_progress')}
        sample['box_progress'][:] = 0
        task._relation_cfg['push']['hand_target_height'] = height
        task._apply_joint_rsi(chosen,torch.zeros_like(chosen),'push',sample)
    task._humanoid_root_states[:,1,:2] += 100
    task._kinematic_humanoid_rigid_body_states[:,1,:,:2] += 100
    task._humanoid_root_states[...,7:] = 0
    task._dof_vel[:] = 0
    task._kinematic_humanoid_rigid_body_states[...,7:] = 0
    task._reset_task(ids)
    task._reset_env_tensors(ids)
    rest = task._dof_pos.clone()
    names = task.gym.get_actor_dof_names(task.envs[0],task.humanoid_handles_all[0][0])
    props = task.gym.get_actor_dof_properties(task.envs[0],task.humanoid_handles_all[0][0])
    kp = torch.tensor(props['stiffness'],device=task.device)
    arm_ids = torch.tensor([i for i,name in enumerate(names) if 'shoulder' in name or 'elbow' in name],device=task.device)
    ankles = [names.index(x+'_ankle_y') for x in ('right','left')]
    hips = [names.index(x+'_hip_y') for x in ('right','left')]
    lo, hi = task._pd_action_offset-task._pd_action_scale, task._pd_action_offset+task._pd_action_scale
    jac = gymtorch.wrap_tensor(task.gym.acquire_jacobian_tensor(task.sim,'humanoid_0'))
    hands = task._key_body_ids[:2]
    feet = task._key_body_ids[2:]
    task._refresh_sim_tensors()
    start = task._box_states[:,2,:3].clone()
    actual_masses = []
    for e in range(n):
        handle = task._box_handles[4*e+2]
        mass = sum(p.mass for p in task.gym.get_actor_rigid_body_properties(task.envs[e],handle))
        assert abs(mass-20*(CASES[e][0]/1.3)**3)<.001
        assert all(abs(p.friction-.6)<1e-5 for p in task.gym.get_actor_rigid_shape_properties(task.envs[e],handle))
        actual_masses.append(mass)
    zeros = torch.zeros(n,device=task.device)
    stats = {k:zeros.clone() for k in ('max_tilt_deg','contact_ticks','both_ticks','supported_contact_ticks',
        'force_y_sum','force_y_peak','force_z_abs_peak','midpoint_z_sum','midpoint_z_min',
        'midpoint_z_max','height_difference_peak','force_imbalance_peak','hand_contact_path_m',
        'peak_nonhand_reaction_N','body_fall','max_forward_m','first_10cm_time_s',
        'nonhand_near_box_ticks','min_root_height_m')}
    stats['midpoint_z_min'][:] = float('inf')
    stats['first_10cm_time_s'][:] = -1
    stats['min_root_height_m'][:] = float('inf')
    previous = start.clone()
    trace = []
    actions = torch.zeros(n*2,task.num_actions,device=task.device)
    ticks = 240  # four seconds, each point sampled every 1/60s physics tick
    for tick in range(ticks):
        task.gym.refresh_jacobian_tensors(task.sim)
        body = task._rigid_body_pos[:,0]
        box = task._box_states[:,2]
        target = rest.clone()
        ramp = min(1.,tick/30.)
        target[:,0,ankles] += .15*ramp
        target[:,0,hips] += .2*ramp
        # Hands follow the translating rear face; no root/box teleport or force.
        desired = body[:,hands].clone()
        desired[:,:,0] = box[:,0,None]+torch.tensor([.2,-.2],device=task.device)[None]
        desired[:,:,1] = box[:,1,None]-sides[:,None]/2+.025
        desired[:,:,2] = heights[:,None]
        error = (desired-body[:,hands]).clamp(-.08,.08).reshape(n,6)
        j = jac[:,hands,:3,6:][:,:,:,arm_ids].reshape(n,6,-1)
        correction = (j.transpose(1,2)@torch.linalg.solve(j@j.transpose(1,2)+torch.eye(6,device=task.device)*.01,error[:,:,None])).squeeze(-1)
        target[:,0,arm_ids] = task._dof_pos[:,0,arm_ids]+correction.clamp(-.10,.10)
        load_ramp = max(0.,min(1.,(tick-30)/60.))
        tau = (jac[:,hands,1,6:]*loads[:,None,None]*load_ramp/2).sum(1)
        # Whole-body J^T load sharing includes the support legs, as in the
        # original validated actuator-load search; arm-only bias loses balance.
        target[:,0] += tau/kp.clamp_min(1.)
        task._every_env_init_dof_pos[:] = target.clamp(lo,hi)
        task.pre_physics_step(actions)
        task.gym.simulate(task.sim)
        task.gym.fetch_results(task.sim,True)
        task._refresh_sim_tensors()
        body = task._rigid_body_pos[:,0]
        box = task._box_states[:,2]
        q = box[:,3:7]
        tilt = torch.acos((1-2*(q[:,0].square()+q[:,1].square())).clamp(-1,1))*180/torch.pi
        reactions = task._contact_forces[:,0,hands]
        near = ((body[:,hands,1]-(box[:,1]-sides/2)[:,None]).abs()<.15)
        near &= (body[:,hands,0]-box[:,0,None]).abs() < sides[:,None]/2+.1
        near &= (body[:,hands,2]>.05)&(body[:,hands,2]<sides[:,None]+.05)
        contact = near & ((-reactions[:,:,1])>.5)
        any_contact, both = contact.any(-1),contact.all(-1)
        foot_contact = (task._contact_forces[:,0,feet,2]>1).all(-1)
        supported = foot_contact & (task._humanoid_root_states[:,0,2]>.45)
        force_y = (-reactions[:,:,1]).clamp_min(0)*contact
        force = force_y.sum(-1)
        midpoint = body[:,hands,2].mean(-1)
        dy = box[:,1]-start[:,1]
        stats['max_tilt_deg'] = torch.maximum(stats['max_tilt_deg'],tilt)
        stats['max_forward_m'] = torch.maximum(stats['max_forward_m'],dy)
        stats['contact_ticks'] += any_contact
        stats['both_ticks'] += both
        stats['supported_contact_ticks'] += supported&both
        stats['body_fall'] = torch.maximum(stats['body_fall'],(task._humanoid_root_states[:,0,2]<.35).float())
        stats['min_root_height_m'] = torch.minimum(stats['min_root_height_m'],task._humanoid_root_states[:,0,2])
        stats['force_y_sum'] += force
        stats['force_y_peak'] = torch.maximum(stats['force_y_peak'],force)
        stats['force_z_abs_peak'] = torch.maximum(stats['force_z_abs_peak'],(reactions[:,:,2]*contact).sum(-1).abs())
        stats['midpoint_z_sum'] += midpoint*both
        stats['midpoint_z_min'] = torch.minimum(stats['midpoint_z_min'],torch.where(both,midpoint,float('inf')))
        stats['midpoint_z_max'] = torch.maximum(stats['midpoint_z_max'],midpoint*both)
        stats['height_difference_peak'] = torch.maximum(stats['height_difference_peak'],(body[:,hands[0],2]-body[:,hands[1],2]).abs()*both)
        stats['force_imbalance_peak'] = torch.maximum(stats['force_imbalance_peak'],(force_y[:,0]-force_y[:,1]).abs()*both)
        travel = (box[:,:2]-previous[:,:2]).norm(dim=-1)
        stats['hand_contact_path_m'] += travel*any_contact
        others = torch.ones(task.num_bodies,dtype=torch.bool,device=task.device)
        others[hands] = False
        others[feet] = False
        near_box = ((body[:,others,:2]-box[:,None,:2]).abs()<sides[:,None,None]/2+.12).all(-1)
        near_box &= (body[:,others,2]>.1)&(body[:,others,2]<sides[:,None]+.1)
        nonhand_contact = near_box & (task._contact_forces[:,0,others,:2].norm(dim=-1)>1.)
        stats['nonhand_near_box_ticks'] += nonhand_contact.any(-1)
        stats['peak_nonhand_reaction_N'] = torch.maximum(stats['peak_nonhand_reaction_N'],task._contact_forces[:,0,others,:2].norm(dim=-1).sum(-1))
        first = (dy>=.1)&(stats['first_10cm_time_s']<0)
        stats['first_10cm_time_s'][first] = (tick+1)/60
        trace.append(torch.stack((tilt,dy,midpoint,force,(reactions[:,:,2]*contact).sum(-1),both.float(),supported.float(),task._humanoid_root_states[:,0,2],nonhand_contact.any(-1).float()),-1).cpu())
        previous = box[:,:3].clone()
        assert torch.isfinite(task._box_states).all() and torch.isfinite(task._dof_pos).all()
        if tick%60==0:
            print('HEIGHT_PD_PROGRESS',tick,ticks,'cases',n,flush=True)
    rows = []
    for e,(side,height,load,pose) in enumerate(CASES):
        row = dict(side_m=side,mass_kg=actual_masses[e],command_midpoint_z_m=height,
            command_total_horizontal_load_N=load,motion_id=pose)
        row.update({k:float(v[e]) for k,v in stats.items()})
        row['final_forward_m'] = float(task._box_states[e,2,1]-start[e,1])
        row['contact_mean_horizontal_force_N'] = row['force_y_sum']/max(1,row['contact_ticks'])
        row['paired_contact_mean_midpoint_z_m'] = row['midpoint_z_sum']/max(1,row['both_ticks'])
        if not row['both_ticks']:row['midpoint_z_min']=None
        row['moved_10cm_max_tilt_le5'] = bool(row['final_forward_m']>=.1 and row['max_tilt_deg']<=5 and row['hand_contact_path_m']>=.1 and row['both_ticks']>=6)
        row['supported_no_fall_success'] = row['moved_10cm_max_tilt_le5'] and row['body_fall']==0 and row['supported_contact_ticks']>=6
        row['hands_only_proxy_success'] = row['supported_no_fall_success'] and row['nonhand_near_box_ticks']==0
        rows.append(row)
    import numpy as np
    np.savez_compressed(out/'trace.npz',trace=torch.stack(trace).numpy(),columns=np.array(['tilt_deg','forward_m','midpoint_z_m','hand_horizontal_N','hand_vertical_N','both_contact','both_feet_support','root_height_m','nonhand_near_box']))
    report = dict(gpu=os.environ['TOKENHSI_GPU'],training=False,policy_used=False,external_force=False,
        num_cases=n,duration_s=ticks/60,physics_tick_s=1/60,internal_substeps=sim.substeps,
        box_friction=.6,floor_friction=1.,density_kg_m3=20/1.3**3,
        criterion='Final forward >=0.10m, hand-contact path >=0.10m, >=6 paired contact ticks and maximum tilt <=5deg over entire 4s; strict subset also no root<0.35m and >=6 both-foot support ticks (root>0.45m). Hands-only proxy additionally excludes nonhand bodies near box with horizontal net reaction>1N.',
        limits='Native articulated PD with Jacobian wrist targets and force-bias; no learned policy or external force. Wrist net reactions plus geometry are not pair-resolved. Whole-body collisions can also push the box; peak nonhand reactions are provided. Command force/height are not measured contact force/height. No universal guarantee outside these tested controls.',
        bank_unchanged=hashlib.sha256(bank_path.read_bytes()).hexdigest()==bank_hash,cases=rows)
    assert report['bank_unchanged']
    (out/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (out/'cases.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('HEIGHT_PD_RESULT',n,'moving <=5deg',sum(r['moved_10cm_max_tilt_le5'] for r in rows),'supported',sum(r['supported_no_fall_success'] for r in rows),flush=True)
finally:
    task.gym.destroy_sim(task.sim)

"""Read-only checkpoint evaluation of size-linked mass, yaw and tipping.
Run through the original run.py so checkpoint/normalizer/physics contracts apply.
"""
from isaacgym import gymapi  # must precede torch
import hashlib
import json
import math
from pathlib import Path
import runpy
import torch
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from env.tasks.multi_agent.open_push_reward import push_face_geometry
from learning.multi_agent.ma_players import MAPlayerContinuous

torch.set_num_threads(1)
_original_init = HumanoidMAOpenPush.__init__
def audit_init(self, cfg, *args, **kwargs):
    cfg['_size_audit'] = True  # existing audit-only uniform actor-size allocation
    return _original_init(self, cfg, *args, **kwargs)
HumanoidMAOpenPush.__init__ = audit_init

def yaw(q):
    x,y,z,w=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
def wrapped(a):
    return torch.atan2(a.sin(),a.cos())

def summarize(rows):
    n=len(rows)
    if not n:return None
    result={'trials':n,'scene_trials':len({(r['repeat'],r['env']) for r in rows})}
    for key in ['ever_success','final_success','ever_upright_success','final_upright_success','ever_yaw30_upright','ever_tilt20','ever_tilt45','ever_tilt80','overshoot_10cm','scene_fall','rsi']:
        result[key+'_rate']=sum(r[key] for r in rows)/n
    for key in ['forward_m','goal_reduction_m','final_goal_distance_m','initial_goal_distance_m','lateral_m','max_lateral_m','max_yaw_deg','max_upright_yaw_deg','yaw_path_deg','max_tilt_deg','steps','paired_contact_rate','contact_hand_force_N','contact_force_imbalance','mass_kg','side_m']:
        result[key+'_mean']=sum(r[key] for r in rows)/n
    for key in ['forward_m','max_upright_yaw_deg','max_tilt_deg']:
        vals=sorted(r[key] for r in rows)
        result[key+'_median']=vals[n//2]
        result[key+'_p90']=vals[int((n-1)*.9)]
    successful=[r for r in rows if r['ever_success']]
    result['successful_trials']=len(successful)
    for key in ['first_success_tilt_deg','max_success_tilt_deg']:
        vals=sorted(r[key] for r in successful)
        result[key+'_mean']=sum(vals)/len(vals) if vals else None
        result[key+'_median']=vals[len(vals)//2] if vals else None
        result[key+'_p90']=vals[int((len(vals)-1)*.9)] if vals else None
    return result

@torch.no_grad()
def evaluate(player):
    task=player.env.task
    args=task.cfg['args'];out=Path(args.output_path);out.mkdir(parents=True,exist_ok=True)
    bank=Path(task.cfg['env']['openPushRSI']['bank_file'])
    bank_hash=hashlib.sha256(bank.read_bytes()).hexdigest()
    assert task.num_envs%62==0
    assert player.is_determenistic
    # Read physical masses from simulator actors, independently of the formula.
    physical_mass=torch.zeros(task.num_envs,2,device=task.device)
    for e in range(task.num_envs):
        for lane in range(2):
            handle=task._box_handles[e*task.num_objects+lane+2]
            physical_mass[e,lane]=sum(p.mass for p in task.gym.get_actor_rigid_body_properties(task.envs[e],handle))
    trials=[]
    for mode in ('training_reset_mix','loco'):
        task._is_eval=mode=='loco'
        if not task._is_eval:
            task._skill=list(task.cfg['env']['skill'])
            task._skill_init_prob=torch.tensor(task.cfg['env']['skillInitProb'],device=task.device)
        for repeat in range(args.eval_repeats):
            obs=player.env_reset();player.get_batch_size(obs['obs'],1)
            boxes=task._owner_push_boxes();shape=boxes.shape[:2]
            start=boxes[...,:3].clone();goals=task._tar_pos.clone();side=task._owner_push_sizes()[...,0].clone()
            assert len(torch.unique(side))==31
            masses=task._owner_push_values(physical_mass).clone()
            torch.testing.assert_close(masses,20*(side/1.3)**3,atol=2e-4,rtol=1e-4)
            delta=goals[...,:2]-start[...,:2];initial_distance=delta.norm(dim=-1);axis=delta/initial_distance[...,None].clamp_min(1e-6)
            lateral_axis=torch.stack((-axis[...,1],axis[...,0]),-1)
            initial_yaw=yaw(boxes[...,3:7]);previous_yaw=initial_yaw.clone()
            final_pos=start.clone();alive=torch.ones(task.num_envs,device=task.device,dtype=torch.bool)
            initial_rsi=task._rsi_interaction.clone()
            floats=['ticks','steps','max_lateral','max_yaw','max_upright_yaw','yaw_path','max_tilt','contact_ticks','contact_force','force_imbalance']
            stats={k:torch.zeros(shape,device=task.device) for k in floats}
            first_success_tilt=torch.zeros(shape,device=task.device)
            max_success_tilt=torch.zeros(shape,device=task.device)
            flags={k:torch.zeros(shape,device=task.device,dtype=torch.bool) for k in ['ever_success','final_success','ever_upright_success','final_upright_success','ever_yaw30_upright','ever_tilt20','ever_tilt45','ever_tilt80','overshoot_10cm','scene_fall']}
            original=task.relation_runtime.accumulate_push_credit
            def record(boxes,targets,contact):
                nonlocal previous_yaw,final_pos
                original(boxes,targets,contact)
                valid=alive[:,None].expand(shape)
                pos=boxes[...,:3];q=boxes[...,3:7]
                tilt=torch.acos((1-2*(q[...,0]**2+q[...,1]**2)).clamp(-1,1))
                current_yaw=yaw(q);change=wrapped(current_yaw-initial_yaw).abs()
                upright=tilt<=.35
                offset=pos[...,:2]-start[...,:2]
                lat=(offset*lateral_axis).sum(-1).abs()
                forward=(offset*axis).sum(-1)
                stats['ticks']+=valid
                stats['max_lateral']=torch.maximum(stats['max_lateral'],lat*valid)
                stats['max_yaw']=torch.maximum(stats['max_yaw'],change*valid)
                stats['max_upright_yaw']=torch.maximum(stats['max_upright_yaw'],change*valid*upright)
                stats['yaw_path']+=wrapped(current_yaw-previous_yaw).abs()*valid
                stats['max_tilt']=torch.maximum(stats['max_tilt'],tilt*valid)
                for key,condition in [('ever_yaw30_upright',(change>math.pi/6)&upright),('ever_tilt20',tilt>.35),('ever_tilt45',tilt>math.pi/4),('ever_tilt80',tilt>math.radians(80)),('overshoot_10cm',forward>initial_distance+.1)]:
                    flags[key]|=condition&valid
                # Normal hand reactions on strict paired-contact ticks only.
                _,normal,_,_,_=push_face_geometry(boxes,task._owner_push_sizes(),targets,task._rigid_body_pos[:,:,task._key_body_ids[:2]])
                force=-(task._contact_forces[:,:,task._key_body_ids[:2]]*normal[:,:,None]).sum(-1)
                force=force.clamp_min(0)
                total=force.sum(-1);paired=contact&valid
                stats['contact_ticks']+=paired
                stats['contact_force']+=total*paired
                stats['force_imbalance']+=(force[...,0]-force[...,1]).abs()/total.clamp_min(1e-6)*paired
                final_pos=torch.where(valid[...,None],pos,final_pos)
                previous_yaw=current_yaw.clone()
            task.relation_runtime.accumulate_push_credit=record
            try:
                for step in range(task.max_episode_length+1):
                    action=player.get_action(obs,player.is_determenistic)
                    action=action.reshape(task.num_envs,task.num_agents,-1)
                    action[~alive]=0
                    obs,_,done,info=player.env_step(player.env,action.reshape(-1,task.num_actions))
                    obs=player.obs_to_torch(obs)
                    assert torch.isfinite(task.obs_buf).all()
                    valid=alive[:,None].expand(shape);stats['steps']+=valid
                    success=task.relation_runtime.done.bool()
                    b=task._owner_push_boxes();q=b[...,3:7]
                    upright=(1-2*(q[...,0]**2+q[...,1]**2))>=math.cos(.35)
                    tilt_deg=torch.acos((1-2*(q[...,0]**2+q[...,1]**2)).clamp(-1,1))*180/math.pi
                    first=success&valid&~flags['ever_success']
                    first_success_tilt=torch.where(first,tilt_deg,first_success_tilt)
                    max_success_tilt=torch.maximum(max_success_tilt,torch.where(success&valid,tilt_deg,torch.zeros_like(tilt_deg)))
                    flags['ever_success']|=success&valid
                    flags['ever_upright_success']|=success&upright&valid
                    flags['final_success']=torch.where(valid,success,flags['final_success'])
                    flags['final_upright_success']=torch.where(valid,success&upright,flags['final_upright_success'])
                    ending=done.reshape(shape).bool().any(-1)
                    fall=task._terminate_buf.reshape(shape).bool().any(-1)
                    flags['scene_fall']|=(fall&alive)[:,None]
                    alive&=~ending
                    if step%150==0:print('AUDIT_PROGRESS',mode,repeat,step,flush=True)
                    if not alive.any():break
                assert not alive.any(),'Episode horizon exceeded'
            finally:
                task.relation_runtime.accumulate_push_credit=original
            offset=final_pos[...,:2]-start[...,:2];distance=(final_pos[...,:2]-goals[...,:2]).norm(dim=-1)
            values=dict(side_m=side,mass_kg=masses,initial_goal_distance_m=initial_distance,final_goal_distance_m=distance,
                goal_reduction_m=initial_distance-distance,forward_m=(offset*axis).sum(-1),lateral_m=(offset*lateral_axis).sum(-1).abs(),
                max_lateral_m=stats['max_lateral'],max_yaw_deg=stats['max_yaw']*180/math.pi,max_upright_yaw_deg=stats['max_upright_yaw']*180/math.pi,
                yaw_path_deg=stats['yaw_path']*180/math.pi,max_tilt_deg=stats['max_tilt']*180/math.pi,steps=stats['steps'],
                paired_contact_rate=stats['contact_ticks']/stats['ticks'].clamp_min(1),contact_hand_force_N=stats['contact_force']/stats['contact_ticks'].clamp_min(1),
                contact_force_imbalance=stats['force_imbalance']/stats['contact_ticks'].clamp_min(1),rsi=initial_rsi,**flags)
            values.update(first_success_tilt_deg=first_success_tilt,max_success_tilt_deg=max_success_tilt)
            cpu={k:v.cpu().tolist() for k,v in values.items()}
            for e in range(shape[0]):
                for a in range(shape[1]):
                    row=dict(mode=mode,repeat=repeat,env=e,agent=a,**{k:v[e][a] for k,v in cpu.items()})
                    if not row['ever_success']:
                        row['first_success_tilt_deg']=None
                        row['max_success_tilt_deg']=None
                    trials.append(row)
            (out/'trials.json').write_text(json.dumps(trials,indent=2)+'\n')
            print('AUDIT_DONE',mode,repeat,'trials',len(trials),flush=True)
    summaries={}
    for mode in ['training_reset_mix','training_rsi','training_loco','loco']:
        rows=[r for r in trials if r['mode']==mode or mode=='training_rsi' and r['mode']=='training_reset_mix' and r['rsi'] or mode=='training_loco' and r['mode']=='training_reset_mix' and not r['rsi']]
        summaries[mode]={'all':summarize(rows),'size_bins':{},'per_size':{}}
        for name,low,high in [('small',1.099,1.195),('middle',1.195,1.305),('large',1.305,1.401)]:
            summaries[mode]['size_bins'][name]=summarize([r for r in rows if low<=r['side_m']<high])
        for i in range(31):
            s=round(1.1+.01*i,2)
            summaries[mode]['per_size'][str(s)]=summarize([r for r in rows if abs(r['side_m']-s)<.001])
    assert hashlib.sha256(bank.read_bytes()).hexdigest()==bank_hash
    report=dict(checkpoint=args.checkpoint,seed=args.seed,num_envs=task.num_envs,repeats=args.eval_repeats,episode_length=task.max_episode_length,
        deterministic=True,training=False,bank_sha256=bank_hash,bank_unchanged=True,summary=summaries,
        notes=['Mass read from actual actors; constant density couples size and mass, so this is not a mass-only causal test.',
        'Native AT success has no tilt gate; upright success additionally requires tilt <=0.35rad at the same control step.',
        'First-success and maximum-success tilt sampled at successful control steps; unsuccessful episodes excluded from conditional tilt statistics.',
        'Yaw/tilt/max lateral sampled at physics ticks; upright yaw omits frames with tilt >0.35rad.',
        'Scene terminates when either agent falls or 600-step horizon ends; both agent outcomes are correlated.',
        'Forward projection, actual goal-distance reduction and overshoot are separate; initial RSI displacement is excluded.',
        'Force is sum of two nonnegative normal hand reactions during strict paired-contact ticks; not commanded force or proof of sole hand causation.'])
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    for mode in ['training_reset_mix','loco']:
        print('AUDIT_SUMMARY',mode,json.dumps(summaries[mode]['size_bins']),flush=True)

MAPlayerContinuous.run_eval=evaluate
runpy.run_path('tokenhsi/run.py',run_name='__main__')

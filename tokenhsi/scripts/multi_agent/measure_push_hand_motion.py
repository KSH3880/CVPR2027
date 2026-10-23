"""Policy evaluation: direction-independent XY motion during hand contact.

Contact and displacement coincide at physics ticks; this is not causal force
attribution. Path length includes rocking/oscillation, so also report net motion.
"""
from isaacgym import gymapi  # before torch
import json
import os
from pathlib import Path
import runpy
import torch
from learning.multi_agent.ma_players import MAPlayerContinuous
from env.tasks.multi_agent.open_push_reward import push_any_surface_geometry


def run_motion_eval(player):
    task=player.env.task
    out=Path(task.cfg['args'].output_path);out.mkdir(parents=True,exist_ok=True)
    trials=[]
    for mode in ('training_reset_mix','loco'):
        task._is_eval=mode=='loco'
        if not task._is_eval:
            task._skill=list(task.cfg['env']['skill'])
            task._skill_init_prob=torch.tensor(task.cfg['env']['skillInitProb'],device=task.device)
        for repeat in range(task.cfg['args'].eval_repeats):
            obs=player.env_reset();player.get_batch_size(obs['obs'],1)
            start=task._owner_push_boxes()[...,:3].clone()
            previous=start.clone();final=start.clone()
            shape=start.shape[:2];alive=torch.ones(shape[0],device=task.device,dtype=torch.bool)
            stats={k:torch.zeros(shape,device=task.device) for k in
                   ('ticks','any_ticks','both_ticks','strict_ticks','path','hand_path','both_path','upright_hand_path','max_excursion')}
            rsi=task._rsi_interaction.clone();ever=torch.zeros(shape,device=task.device,dtype=torch.bool)
            original=task.relation_runtime.accumulate_push_credit

            def record(boxes,goals,strict_contact):
                nonlocal previous,final
                original(boxes,goals,strict_contact)
                hands=task._rigid_body_pos[:,:,task._key_body_ids[:2]]
                forces=task._contact_forces[:,:,task._key_body_ids[:2]]
                distance,normal=push_any_surface_geometry(hands,boxes,task._owner_push_sizes())
                # Any box surface; no goal direction/selected-face restriction.
                hand=(distance<=task._relation_cfg['push']['hand_distance']) & ((forces*normal).sum(-1)>=task._relation_cfg['push']['min_hand_force'])
                valid=alive[:,None].expand(shape)
                any_hand=hand.any(-1)&valid;both=hand.all(-1)&valid
                pos=boxes[...,:3];travel=(pos[...,:2]-previous[...,:2]).norm(dim=-1)
                q=boxes[...,3:7]
                upright=(1-2*(q[...,0].square()+q[...,1].square())>=torch.cos(pos.new_tensor(task._relation_cfg['push']['max_tilt']))) & ((pos[...,2]-start[...,2]).abs()<=task._relation_cfg['push']['ground_tolerance'])
                for key,value in [('ticks',valid),('any_ticks',any_hand),('both_ticks',both),('strict_ticks',strict_contact&valid),('path',travel*valid),('hand_path',travel*any_hand),('both_path',travel*both),('upright_hand_path',travel*any_hand*upright)]:
                    stats[key]+=value
                stats['max_excursion']=torch.maximum(stats['max_excursion'],(pos[...,:2]-start[...,:2]).norm(dim=-1)*valid)
                final=torch.where(valid[...,None],pos,final)
                previous=pos.clone()

            task.relation_runtime.accumulate_push_credit=record
            try:
                for step in range(task.max_episode_length+1):
                    action=player.get_action(obs,player.is_determenistic)
                    obs,_,done,_=player.env_step(player.env,action)
                    obs=player.obs_to_torch(obs)
                    assert torch.isfinite(task.obs_buf).all()
                    ever|=task.relation_runtime.done&alive[:,None]
                    alive&=~done.reshape(shape).bool().any(-1)
                    if not alive.any():break
                    if step%100==0:print('MOTION_PROGRESS',mode,repeat,step,flush=True)
                assert not alive.any(),'Evaluation exceeded episode horizon'
            finally:
                task.relation_runtime.accumulate_push_credit=original
            net=(final[...,:2]-start[...,:2]).norm(dim=-1)
            cpu={k:v.cpu().tolist() for k,v in stats.items()}
            for env in range(shape[0]):
                for agent in range(shape[1]):
                    row={k:v[env][agent] for k,v in cpu.items()}
                    row.update(mode=mode,repeat=repeat,env=env,agent=agent,rsi=bool(rsi[env,agent]),net_xy_m=float(net[env,agent]),ever_at_success=bool(ever[env,agent]))
                    trials.append(row)
            print('MOTION_DONE',mode,repeat,flush=True)
    summaries={}
    for group in ('training_reset_mix','training_rsi','training_loco','loco'):
        rows=[r for r in trials if (r['mode']==group or group=='training_rsi' and r['mode']=='training_reset_mix' and r['rsi'] or group=='training_loco' and r['mode']=='training_reset_mix' and not r['rsi'])]
        if not rows:continue
        total=lambda k:sum(r[k] for r in rows)
        n=len(rows);ticks=max(1,total('ticks'))
        summaries[group]=dict(trials=n,any_hand_tick_rate=total('any_ticks')/ticks,both_hands_tick_rate=total('both_ticks')/ticks,strict_contact_tick_rate=total('strict_ticks')/ticks,
            mean_hand_contact_path_m=total('hand_path')/n,mean_both_hands_path_m=total('both_path')/n,mean_upright_hand_path_m=total('upright_hand_path')/n,
            mean_total_path_m=total('path')/n,mean_net_xy_m=total('net_xy_m')/n,mean_max_excursion_m=total('max_excursion')/n,
            hand_path_10cm_rate=sum(r['hand_path']>=.1 for r in rows)/n,hand_path_50cm_rate=sum(r['hand_path']>=.5 for r in rows)/n,
            net_10cm_rate=sum(r['net_xy_m']>=.1 for r in rows)/n,ever_at_success_rate=sum(r['ever_at_success'] for r in rows)/n)
        summaries[group]['contact_path_and_net_10cm_rate']=sum(r['hand_path']>=.1 and r['net_xy_m']>=.1 for r in rows)/n
        for key in ('hand_path','net_xy_m'):
            values=sorted(r[key] for r in rows)
            summaries[group][key+'_median']=values[n//2]
            summaries[group][key+'_p90']=values[int(.9*(n-1))]
    report=dict(checkpoint=task.cfg['args'].checkpoint,seed=task.cfg['args'].seed,summary=summaries,
                num_envs=task.num_envs,repeats=task.cfg['args'].eval_repeats,episode_length=task.max_episode_length,deterministic=bool(player.is_determenistic),
                notes='XY path while hand surface distance <= configured threshold and outward reaction >= threshold, any direction. Includes rocking/oscillation; net displacement is separate. Contact coincidence does not prove sole hand causation.')
    (out/'trials.json').write_text(json.dumps(trials,indent=2)+'\n')
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


MAPlayerContinuous.run_eval=run_motion_eval
runpy.run_path('tokenhsi/run.py',run_name='__main__')

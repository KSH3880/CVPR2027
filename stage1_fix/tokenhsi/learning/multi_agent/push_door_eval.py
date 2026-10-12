"""Finite checkpoint evaluation for independently assigned push and door tasks."""
import json
import os
from pathlib import Path
import torch


@torch.no_grad()
def run_push_door_eval(player):
    task=player.env.task
    results=[]
    reward_interval=max(0,int(os.environ.get("REWARD_VIEW_EVERY","0")))
    output=Path(task.cfg['args'].output_path);output.mkdir(parents=True,exist_ok=True)
    for repeat in range(task.cfg['args'].eval_repeats):
        obs=player.env_reset();player.get_batch_size(obs['obs'],1)
        if task.viewer is not None and os.environ.get('VIEW_OVERVIEW','0')=='1':
            from isaacgym import gymapi
            points=torch.cat([task._humanoid_root_states[0,:,:3],task._box_states[0,:,:3],task._targets[0]],0)
            minimum=points[:,:2].amin(0);maximum=points[:,:2].amax(0)
            center=(minimum+maximum)/2;span=max(8.,float((maximum-minimum).max())+4.)
            task._camera_follow_flag=False
            task.gym.viewer_camera_look_at(task.viewer,task.envs[0],
                gymapi.Vec3(float(center[0])+span*.6,float(center[1])-span*.9,span*.7),
                gymapi.Vec3(float(center[0]),float(center[1]),1.))
        alive=torch.ones(task.num_envs,device=task.device,dtype=torch.bool)
        assignments=task._tasks.clone()
        success=torch.zeros_like(assignments,dtype=torch.bool)
        current=torch.zeros_like(success)
        peak_angle=torch.zeros_like(assignments,dtype=torch.float)
        for step in range(task.max_episode_length):
            if not isinstance(obs,dict):obs={"obs":obs}
            actions=player.get_action(obs,player.is_determenistic)
            obs,rew,done,info=player.env_step(player.env,actions)
            if reward_interval and step%reward_interval==0:
                rows=info['reward_terms'][:task.num_agents].detach().cpu().tolist()
                for agent,values in enumerate(rows):
                    terms=dict(zip(task.REWARD_TERM_NAMES,values))
                    names=('push_hand_state','push_hand_progress','push_hand_result','at_state','at_progress','at_result') if int(assignments[0,agent])==0 else ('door_progress','door_hold','success_bonus','door_approach','door_closing')
                    print('[view reward] step={} agent={} task={} {}'.format(step,agent,'PUSH' if int(assignments[0,agent])==0 else 'DOOR',
                        ' '.join('{}={:.4f}'.format(name,terms[name]) for name in names)),flush=True)
            success[alive]|=task._done_task[alive]
            current[alive]=(task._elapsed[alive]>=torch.where(assignments[alive].bool(),task._interaction['door']['hold_seconds'],task._interaction['push']['settle_seconds']))
            peak_angle[alive]=torch.maximum(peak_angle[alive],task._door_state_view[alive,:,0])
            alive &= ~done.view(task.num_envs,task.num_agents).any(-1).bool()
            if task.viewer is not None:
                task.render(sync_frame_time=True)
                if step in (0,30,90):
                    task.gym.write_viewer_image_to_file(task.viewer,str(output/'repeat_{}_step_{}.png'.format(repeat,step)))
            if not alive.any():break
        row={'repeat':repeat,'steps':step+1,'task_assignments':assignments.cpu().tolist(),
            'ever_success':success.cpu().tolist(),'current_success':current.cpu().tolist(),
            'peak_door_angle_degrees':torch.rad2deg(peak_angle).cpu().tolist()}
        for uid,name in enumerate(('push','door')):
            mask=assignments==uid
            row[name+'_trials']=int(mask.sum());row[name+'_successes']=int(success[mask].sum())
        results.append(row)
        print('[PushDoor eval] '+json.dumps(row),flush=True)
    (output/'push_door_eval.json').write_text(json.dumps(results,indent=2)+'\n')
    return results

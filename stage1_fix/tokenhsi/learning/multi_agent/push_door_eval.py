"""Finite checkpoint evaluation for independently assigned push and door tasks."""
import json
from pathlib import Path
import torch


@torch.no_grad()
def run_push_door_eval(player):
    task=player.env.task
    results=[]
    output=Path(task.cfg['args'].output_path);output.mkdir(parents=True,exist_ok=True)
    for repeat in range(task.cfg['args'].eval_repeats):
        obs=player.env_reset();player.get_batch_size(obs['obs'],1)
        alive=torch.ones(task.num_envs,device=task.device,dtype=torch.bool)
        assignments=task._tasks.clone()
        success=torch.zeros_like(assignments,dtype=torch.bool)
        current=torch.zeros_like(success)
        peak_angle=torch.zeros_like(assignments,dtype=torch.float)
        for step in range(task.max_episode_length):
            if not isinstance(obs,dict):obs={"obs":obs}
            actions=player.get_action(obs,player.is_determenistic)
            obs,rew,done,info=player.env_step(player.env,actions)
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

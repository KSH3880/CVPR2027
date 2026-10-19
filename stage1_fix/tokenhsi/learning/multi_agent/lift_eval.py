"""Finite evaluation of independent SUPPORT+LIFT episodes."""
import json
from pathlib import Path
import torch


@torch.no_grad()
def run_lift_eval(player):
    task=player.env.task
    output=Path(task.cfg['args'].output_path);output.mkdir(parents=True,exist_ok=True)
    results=[]
    for repeat in range(task.cfg['args'].eval_repeats):
        obs=player.env_reset();player.get_batch_size(obs['obs'],1)
        alive=torch.ones(task.num_envs,device=task.device,dtype=torch.bool)
        success=torch.zeros(task.num_envs,2,device=task.device,dtype=torch.bool)
        current=torch.zeros_like(success)
        peak_height=task._box_states[:,:,2].clone()
        peak_angle=torch.zeros_like(peak_height)
        final_distance=(task._box_states[:,:,:3]-task._tar_pos).norm(dim=-1)
        for step in range(task.max_episode_length):
            if not isinstance(obs,dict):obs={'obs':obs}
            actions=player.get_action(obs,player.is_determenistic)
            obs,_,done,info=player.env_step(player.env,actions)
            success[alive]|=task._lift_achieved[alive]
            current[alive]=task._lift_current_success[alive]
            peak_height[alive]=torch.maximum(peak_height[alive],task._box_states[alive,:,2])
            peak_angle[alive]=torch.maximum(peak_angle[alive],task._lift_scores['angle_rad'][alive])
            final_distance[alive]=task._lift_scores['distance'][alive]
            alive &= ~done.view(task.num_envs,2).any(-1).bool()
            if task.viewer is not None:
                task.render(sync_frame_time=True)
                if step in (0,30,90):
                    task.gym.write_viewer_image_to_file(task.viewer,str(output/f'repeat_{repeat}_step_{step}.png'))
            if not alive.any():break
        row=dict(repeat=repeat,steps=step+1,trials=success.numel(),successes=int(success.sum()),
            ever_success=success.cpu().tolist(),current_success=current.cpu().tolist(),
            peak_height_m=peak_height.cpu().tolist(),peak_angle_degrees=torch.rad2deg(peak_angle).cpu().tolist(),
            final_target_distance_m=final_distance.cpu().tolist())
        results.append(row)
        print('[Lift eval] '+json.dumps(row),flush=True)
    (output/'lift_eval.json').write_text(json.dumps(results,indent=2)+'\n')
    return results

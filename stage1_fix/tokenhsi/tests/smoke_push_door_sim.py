"""Actual-simulator contracts through the normal checkpoint/player path."""
import sys
import json
import math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import isaacgym
from isaacgym import gymtorch
import torch
import run as entry
from learning.multi_agent.ma_players import MAPlayerContinuous
from utils.unified_training import amp_family_ids,sample_family_matched,preserve_amp_labels


@torch.no_grad()
def check_simulator(player):
    task=player.env.task
    obs=player.env_reset();player.get_batch_size(obs['obs'],1)
    N,M=task.num_envs,task.num_agents
    assert N>=4
    assert task._humanoid_root_states[...,:2].abs().max()<4
    assert task._box_states[...,:2].abs().max()<4
    initial_handles=task._door_geometry()[4]
    door_slots=task._tasks.bool()
    assert torch.linalg.vector_norm(task._humanoid_root_states[...,:2]-initial_handles[...,:2],dim=-1)[door_slots].max()<.8
    # Commit the reset before making another indexed setter call.
    obs,_,_,info=player.env_step(player.env,player.get_action(obs,True))
    before=task._dof_pos[1:].clone();before_tasks=task._tasks[1:].clone()
    player.env_reset(torch.tensor([0],device=task.device))
    torch.testing.assert_close(before,task._dof_pos[1:])
    torch.testing.assert_close(before_tasks,task._tasks[1:])
    obs,_,_,info=player.env_step(player.env,player.get_action(player.env_reset(torch.empty(0,device=task.device,dtype=torch.long)),True))
    # Geometric observations must agree with the actual articulated body positions.
    _,_,_,_,front,back=task._door_geometry()
    real=task.doors.observe()
    error=float((front.flatten(0,1)-real['handle_position']).abs().max())
    assert error<1e-4,error
    # A carried box at the XY goal must earn neither push progress nor settling.
    saved_box=task._box_states[0,0].clone();saved_task=task._tasks[0,0].clone()
    saved_best=task._best_push[0,0].clone()
    task._tasks[0,0]=0;task._best_push[0,0]=-1
    task._box_states[0,0,:3]=task._targets[0,0]
    task._box_states[0,0,2]+=.2;task._box_states[0,0,7:13]=0
    task._compute_reward(task.actions)
    assert task.extras['reward_terms'][0,:2].abs().max()==0
    assert task._elapsed[0,0]==0
    task._box_states[0,0]=saved_box;task._tasks[0,0]=saved_task
    task._best_push[0,0]=saved_best
    # Trigger real angle-based AMP changes without fabricating hand contact.
    task._tasks[0:4]=torch.tensor([[0,0],[0,1],[1,0],[1,1]],device=task.device)
    task._door_state_view[0:4,:,0]=math.radians(90)
    task._door_state_view[0:4,:,1]=0
    actors=task.doors.actor_indices.view(N,M)[0:4].flatten().contiguous()
    task.gym.set_dof_state_tensor_indexed(task.sim,gymtorch.unwrap_tensor(task._dof_state),gymtorch.unwrap_tensor(actors),len(actors))
    task._compute_reward(task.actions)
    task._compute_amp_observations();task._compute_observations()
    family=task._family()[0:4]
    assert torch.equal(family,torch.tensor([[0,0],[0,2],[2,0],[2,2]],device=task.device))
    assert not task._done_task[0:4].any()
    assert not (task._elapsed[0:4]>0).any()  # Angle alone is not HOLD success.
    amp=task._amp_obs_buf.flatten(0,1).flatten(1)
    labels=amp.view(N*M,10,-1)[...,-3:]
    assert torch.equal(labels,labels[:,0:1].expand_as(labels))
    expert=task.fetch_amp_obs_demo(300)
    matched=sample_family_matched(expert,amp,10)
    assert torch.equal(amp_family_ids(matched,10),amp_family_ids(amp,10))
    torch.testing.assert_close(preserve_amp_labels(amp,amp*.5,10).view(N*M,10,-1)[...,-3:],labels)
    assert torch.isfinite(matched).all()
    stepobs=task.obs_buf.clone()
    for _ in range(32):
        action=player.get_action({'obs':stepobs},True)
        out,rew,done,info=player.env_step(player.env,action)
        stepobs=out['obs'] if isinstance(out,dict) else out
        assert torch.isfinite(stepobs).all() and torch.isfinite(rew).all()
        assert torch.isfinite(info['amp_obs']).all()
        if done.any():
            out=player.env_reset(done.nonzero().flatten()[::M]);stepobs=out['obs'] if isinstance(out,dict) else out
    report={'num_envs':N,'observation_size':task.get_obs_size(),'amp_size':task.get_num_amp_obs(),
        'partial_reset':True,'env_local_reset_and_near_door':True,'handle_geometry_max_error_m':error,'all_task_pairs':True,
        'airborne_box_not_push_success':True,'angle_only_not_hold_success':True,'phase_and_history_labels':True,'expert_matching':True,'finite_rollout_steps':32}
    destination=Path(task.cfg['args'].output_path);destination.mkdir(parents=True,exist_ok=True)
    (destination/'sim_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS '+json.dumps(report),flush=True)

MAPlayerContinuous.run_eval=check_simulator
entry.main()

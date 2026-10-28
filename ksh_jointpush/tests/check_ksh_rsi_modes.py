"""CPU-only reset/rollout check for group and independent-agent PUSH RSI."""
import json
from pathlib import Path
import torch


def install():
    from learning.multi_agent.ma_players import MAPlayerContinuous
    from utils.stage2_jointpush import group_sizes
    from ksh_jointpush.ksh_goal_cone import goal_in_cone

    def check(player):
        task=player.env.task
        ids=torch.arange(task.num_envs,device=task.device)
        mode=task.cfg['env']['stage2JointPush']['coupled_rsi_selection']
        rows=[]
        for preset in ['all_coupled','all_solo','random_scenario']:
            task._task_graph_preset=preset
            obs=player.env_reset();player.get_batch_size(obs['obs'],task.num_envs)
            chosen=task._interaction.clone()
            same=task._groups[:,:,None]==task._groups[:,None,:]
            equal=chosen[:,:,None]==chosen[:,None,:]
            if mode=='group':assert (equal|~same).all()
            elif mode=='independent_agents_v1' and preset=='all_coupled':assert (same&~equal).any()
            assert not task._ontop_scene_infeasible(ids).any()
            assigned=task._agent_box_assignment
            assert torch.equal(assigned[:,:,None]==assigned[:,None,:],same)
            for skill,(es,owner) in task._reset_ref_slots.items():
                if not len(es):continue
                assert skill in ('push','loco')
                assert (chosen[es,owner] == (skill=='push')).all()
                lib=task._motion_lib['push_loco' if skill=='loco' else skill]
                mids=task._reset_ref_motion_ids[skill];times=task._reset_ref_motion_times[skill]
                torch.testing.assert_close(task._dof_pos[es,owner],lib.get_motion_state(mids,times)[2])
            boxes=task._assigned_box_values(task._box_states)
            shared=task._members>1
            assert torch.where(shared,goal_in_cone(boxes,task._tar_pos,0.),goal_in_cone(boxes,task._tar_pos,45.)).all()
            d=(task._tar_pos[...,:2]-boxes[...,:2]).norm(dim=-1)
            assert ((d>=2-1e-4)&(d<=10+1e-4)).all()
            before=chosen.clone();goals=task._tar_pos.clone()
            for _ in range(8):
                action=player.get_action(obs if isinstance(obs,dict) else {'obs':obs},True)
                obs,reward,_,_=player.env_step(player.env,action)
                state=obs['obs'] if isinstance(obs,dict) else obs
                assert torch.isfinite(state).all() and torch.isfinite(reward).all()
                torch.testing.assert_close(task._tar_pos,goals)
            # Partial reset must not change untouched environments' mode/objects.
            objects=task._box_states.clone()
            player.env_reset(ids[::2]*task.num_agents)
            torch.testing.assert_close(task._box_states[1::2],objects[1::2])
            assert torch.equal(task._interaction[1::2],before[1::2])
            rows.append(dict(preset=preset,push=chosen.sum().item(),loco=(~chosen).sum().item(),mixed_within_group=bool((same&~equal).any())))
        report=dict(mode=mode,gpu_used=False,optimizer_steps=0,resets=rows,reference_dofs_verified=True,shared_boxes_verified=True,partial_reset_verified=True)
        path=Path(task.cfg['args'].output_path);path.mkdir(parents=True,exist_ok=True)
        (path/'verification.json').write_text(json.dumps(report,indent=2));print('PASS_KSH_RSI_MODES',json.dumps(report),flush=True)
    MAPlayerContinuous.run=check

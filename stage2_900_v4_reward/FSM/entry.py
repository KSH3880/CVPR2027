import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tokenhsi'),str(ROOT/'eval')]
import isaacgym
import numpy as np
import torch
from learning.multi_agent import formal_eval
from utils import evaluation_protocol
import stage2_scalability_layout
import stage2_xy100
import stage2_full_observation
import table2_sampling
from stage1_protocol_policy import Stage1Policy
from controller import StateFreeze


original_configure = evaluation_protocol.configure
def configure_viewer(args,cfg,name):
    headless = args.headless
    args.headless = True
    try:
        return original_configure(args,cfg,name)
    finally:
        args.headless = headless

evaluation_protocol.configure = configure_viewer
stage2_scalability_layout.install()
stage2_xy100.install()
original_eval = formal_eval.run_formal_eval
table2_sampling.install(os.environ['FSM_SETTING'])


@torch.no_grad()
def evaluate(player):
    task = player.env.task
    task._design = True
    player._amp_debug = lambda info: None
    task.cfg['env']['episodeLength'] = int(os.environ['FSM_STEPS'])
    destination = Path(task.cfg['args'].output_path)
    reset,step,action = player.env_reset,player.env_step,player.get_action
    policy = controller = None
    motor_properties = [[task.gym.get_actor_dof_properties(task.envs[env],handle).copy()
                         for handle in handles] for env,handles in enumerate(task.humanoid_handles_all)]
    episodes = []
    for episode in range(int(os.environ['FSM_EPISODES'])):
        path = destination/f'episode_{episode:03d}'
        path.mkdir(parents=True,exist_ok=True)
        task.cfg['args'].output_path = str(path)
        for env,handles in enumerate(task.humanoid_handles_all):
            for person,handle in enumerate(handles):
                task.gym.set_actor_dof_properties(task.envs[env],handle,motor_properties[env][person])
        shots = 0
        def capture(name):
            nonlocal shots
            if task.viewer is not None and os.environ.get('FSM_CAPTURE')=='1' and shots<8:
                task.render()
                task.gym.write_viewer_image_to_file(task.viewer,str(path/f'{controller.frame:04d}_{name}.png'))
                shots += 1

        def reset_scene(*args,**kwargs):
            nonlocal controller
            obs = reset(*args,**kwargs)
            import stage2_protocol_sampling
            stage2_protocol_sampling.capture(task)
            torch.save(dict(roots=task._humanoid_root_states.cpu().clone(),dof_pos=task._dof_pos.cpu().clone(),
                            dof_vel=task._dof_vel.cpu().clone(),boxes=task._evaluation_box_values(task._box_states).cpu().clone()),
                       path/'initial_before_fsm.pt')
            controller = StateFreeze(task)
            task._compute_observations()
            return reset(torch.empty(0,device=task.device,dtype=torch.long))

        def native_action(obs,is_determenistic=False):
            nonlocal policy
            if policy is None:
                policy = Stage1Policy(player,os.environ['FSM_STAGE1_CHECKPOINT'],obs['obs'])
            if controller.frame>0 and controller.update():
                task._compute_observations()
                obs = reset(torch.empty(0,device=task.device,dtype=torch.long))
                capture('transition')
            result = action(obs,is_determenistic).reshape(task.num_envs,task.num_agents,-1)
            return result.masked_fill(controller.frozen[...,None],0).flatten(0,1)

        def controlled_step(*args,**kwargs):
            result = step(*args,**kwargs)
            controller.frame += 1
            controller.score_alive = controller.alive.clone()
            controller.alive &= ~task._terminate_buf.reshape(task.num_envs,task.num_agents).bool()
            if controller.frame in (1,300,600,900):
                capture('scene')
            return result

        player.env_reset,player.env_step,player.get_action = reset_scene,controlled_step,native_action
        try:
            result = original_eval(player)
        finally:
            player.env_reset,player.env_step,player.get_action = reset,step,action
            proof = controller.finish() if controller is not None else None
        np.savez_compressed(path/'fsm_trace.npz',**{key:np.asarray(value) for key,value in controller.trace.items()})
        proof.update(episode=episode,setting=os.environ['FSM_SETTING'],policy='SCT_26k_plus_FSM')
        if proof['max_frozen_root_error'] != 0 or proof['max_frozen_dof_error'] != 0:
            raise ValueError('Frozen humanoid state changed after projection')
        (path/'fsm.json').write_text(json.dumps(proof,indent=2))
        result.update(fsm=proof,policy='SCT_26k_plus_FSM',experiment='SCT_26k_plus_FSM',
                      xy_extent_percent=100,distractor_attention='all_observed_tokens',
                      distractor_physics='removed_from_arena')
        (path/'result.json').write_text(json.dumps(result,indent=2))
        episodes.append(dict(episode=episode,frames=proof['frames'],events=proof['events'],
                             interaction_group_scores=result['interaction_group_scores']))
        controller = None
    task.cfg['args'].output_path = str(destination)
    proof = policy.finish()
    proof.update(external_wait_gate=True,physical_scene_changed=True,
                 controller='FSM/state.py; FSM/controller.py',physical_change='frozen humanoid state projection; task boxes unchanged')
    (destination/'stage1_policy.json').write_text(json.dumps(proof,indent=2))
    (destination/'viewer_summary.json').write_text(json.dumps(dict(completed=True,episodes=episodes,
        stage1_policy=proof,xy_extent_percent=100,distractor_physics='removed',distractor_attention='all_observed_tokens'),indent=2))
    return episodes


if os.environ.get('FSM_COLLISION_PROBE') == '1':
    from collision_probe import run_probe
    formal_eval.run_formal_eval = run_probe
else:
    formal_eval.run_formal_eval = evaluate
runpy.run_path(str(ROOT/'tokenhsi/run.py'),run_name='__main__')

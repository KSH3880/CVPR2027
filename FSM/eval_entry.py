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
from utils.evaluation_protocol import predecessors
from utils.task_role_spec import task_graph_packet
from utils.stage2_spec import evaluation_groups
from controller import StateFreeze
from stage1_protocol_policy import Stage1Policy

base_eval = formal_eval.run_formal_eval
import stage2_scalability_layout
import stage2_xy100
import stage2_full_observation
stage2_scalability_layout.install()
stage2_xy100.install()
import stage2_protocol_sampling as sampling

setting = os.environ.get('TABLE2_SETTING')
if setting:
    import table2_sampling
    table2_sampling.install(setting)
from table2_metrics import Table2Metrics
import stage2_distractor_physics as physics


@torch.no_grad()
def evaluate(player):
    task = player.env.task
    path = Path(task.cfg['args'].output_path)
    task.cfg['env']['episodeLength'] = int(os.environ.get('FSM_STEPS','900'))
    reset,step,action = player.env_reset,player.env_step,player.get_action
    controller = policy = tracker = None
    initial_roots = None
    hands,physical = [],[]
    physical_sizes = None
    task._ghost_checks = 0
    calls = dict(actor=0,critic=0)

    def attention_hook(name):
        def check(module,args,kwargs):
            if kwargs.get('object_valid') is not None:
                raise ValueError('FSM evaluation masked observed distractors')
            calls[name] += 1
        return check

    network = player.model.a2c_network
    hooks = [encoder.register_forward_pre_hook(attention_hook(name),with_kwargs=True)
             for name,encoder in (('actor',network.actor_encoder),('critic',network.critic_encoder))]

    def fresh_observation():
        task._compute_observations()
        return player.obs_to_torch(player.env.reset(torch.empty(0,device=task.device,dtype=torch.long)))

    def reset_scene(*args,**kwargs):
        nonlocal controller,initial_roots
        obs = reset(*args,**kwargs)
        reference = physics.REFERENCE/f'general_seed{task.cfg["args"].seed}'
        if not torch.equal(obs['obs'].cpu(),torch.load(reference/'observed_scene.pt',map_location='cpu')):
            raise ValueError('FSM initial observation differs from reference')
        n,m = task.num_envs,task.num_agents
        packet = task_graph_packet(task.relation_runtime.graph,n).reshape(n,m,5).long().clone()
        groups = evaluation_groups(packet,obs['obs'][:,-m:].long())
        values = task._evaluation_box_values
        initial = dict(groups=groups.cpu(),packet=packet.cpu(),
            predecessor=predecessors(task.relation_runtime.graph,n,task.device).cpu(),
            sizes=values(task._box_size).cpu().clone(),goals=task._tar_pos.cpu().clone(),
            roots=task._humanoid_root_states.cpu().clone(),boxes=values(task._box_states).cpu().clone(),
            dof_pos=task._dof_pos.cpu().clone(),dof_vel=task._dof_vel.cpu().clone(),
            logical_box_order=task._logical_box_order.cpu().clone(),
            shared_center_goals=task._center_tar_pos.cpu().clone(),
            shared_centers=task._assigned_box_values(task._box_states)[...,:3].cpu().clone())
        torch.save(initial,path/'initial_before_fsm.pt')
        initial_roots = task._humanoid_root_states[...,:3].clone()
        torch.save(dict(mask=task._ghost_mask.cpu(),sampled=task._ghost_state.cpu(),
            physical=task._logical_box_values(task._box_states).cpu(),
            observed=physics.observed_boxes(task,task._logical_box_values(task._box_states),None).cpu(),
            observed_scene=obs['obs'].cpu()),path/'physics_only_reset.pt')
        sampling.capture(task)
        same = task._groups[:,:,None] == task._groups[:,None,:]
        centers = torch.bmm(same.float(),task._humanoid_root_states[...,:2])/same.sum(-1,keepdim=True)
        gap = None
        if task.num_agents>4:
            gap = float((centers[:,:,None]-centers[:,None]).norm(dim=-1).masked_fill(same,float('inf')).min())
            if gap<4.99:
                raise ValueError('FSM group center gap below five meters')
            mapping = task._protocol_token_groups
            logical = physics.observed_boxes(task,task._logical_box_values(task._box_states),None)
            target = task._group_spawn_targets.gather(1,mapping[...,None].expand(-1,-1,2))
            if ((logical[...,:2]-target).norm(dim=-1)[~task._stage2_task_slots]>4.501).any():
                raise ValueError('FSM distractor outside assigned group disc')
        if not setting and os.environ['STAGE2_SCENE_MODE']=='independent':
            if (initial['predecessor']>=0).any() or (same.sum(-1)!=1).any():
                raise ValueError('Independent scene contains a relation')
        (path/'layout.json').write_text(json.dumps(dict(minimum_initial_group_gap=gap,
            group_spawn_gap=5. if task.num_agents>4 else None)))
        physics.check_step(task)
        controller = StateFreeze(task)
        return fresh_observation()

    def native_action(obs,is_determenistic=False):
        nonlocal policy
        if policy is None:
            policy = Stage1Policy(player,os.environ['FSM_STAGE1_CHECKPOINT'],obs['obs'])
        if controller.frame>0 and controller.update():
            obs = fresh_observation()
        value = action(obs,is_determenistic).reshape(task.num_envs,task.num_agents,-1)
        return value.masked_fill(controller.frozen[...,None],0).flatten(0,1)

    def controlled_step(*args,**kwargs):
        nonlocal tracker,physical_sizes
        result = step(*args,**kwargs)
        controller.frame += 1
        if controller.frame==1 and (task._humanoid_root_states[...,:3]-initial_roots).norm(dim=-1).max()>.5:
            raise ValueError('First physics frame lost sampled humanoid initialization')
        controller.score_alive = controller.alive.clone()
        if setting:
            if tracker is None:
                tracker = Table2Metrics(torch.load(path/'initial.pt',map_location='cpu'),setting,task.device)
            options = {}
            if setting.startswith('coupled_'):
                box = task._assigned_box_values(task._box_states)[:,0,:7]
                hand = task._rigid_body_pos[:,:,task._key_body_ids[:2]]
                if physical_sizes is None:
                    physical_sizes = task._assigned_box_values(task._box_size)[:,0].clone()
                physical.append(box.cpu().numpy().copy())
                hands.append(hand.cpu().numpy().copy())
                options = dict(physical=box,physical_sizes=physical_sizes,hands=hand)
            tracker.update(task._humanoid_root_states[...,:3],task._evaluation_box_values(task._box_states)[...,:7],
                           controller.score_alive,**options)
        controller.alive &= ~task._terminate_buf.reshape(task.num_envs,task.num_agents).bool()
        physics.check_step(task)
        return result

    player.env_reset,player.env_step,player.get_action = reset_scene,controlled_step,native_action
    try:
        result = base_eval(player)
    finally:
        player.env_reset,player.env_step,player.get_action = reset,step,action
        for hook in hooks:
            hook.remove()
        freeze = controller.finish() if controller is not None else None
    proof = policy.finish()
    proof.update(external_wait_gate=True,physical_scene_changed=True,
                 controller='FSM/controller.py',physical_change='humanoid state projection; task boxes unchanged')
    if freeze['max_frozen_root_error'] or freeze['max_frozen_dof_error']:
        raise ValueError('FSM frozen state changed after projection')
    np.savez_compressed(path/'fsm_trace.npz',**{key:np.asarray(value) for key,value in controller.trace.items()})
    (path/'fsm.json').write_text(json.dumps(freeze,indent=2))
    (path/'stage1_policy.json').write_text(json.dumps(proof,indent=2))
    (path/'physics_only_verification.json').write_text(json.dumps(dict(checked_frames=task._ghost_checks,
        physical_removal_only=True,full_observation_unchanged_at_parking=True,
        all_object_tokens_retained=True,num_distractor_tokens=int(task._ghost_mask.sum())),indent=2))
    result.update(policy='SCT_26k_plus_FSM',experiment='SCT_26k_plus_FSM',stage1_policy=proof,fsm=freeze,
        xy_extent_percent=100,distractor_physics='removed_from_arena',distractor_attention='all_observed_tokens',
        encoder_attention_checks=calls,scene_mode=setting or os.environ['STAGE2_SCENE_MODE'],
        enable_size_mode='common_intersection',scene_carrier_checkpoint=task.cfg['args'].checkpoint,
        group_sampler=('table2_fixed_'+setting if setting else
            ('all_independent_uniform_skills' if os.environ['STAGE2_SCENE_MODE']=='independent'
             else 'sequential_uniform_feasible_I_E_C')))
    if task.num_agents>4:
        result['group_spawn_over4'].update(unused_distractors=True,sampler=result['group_sampler'])
    if setting:
        table = tracker.summary(task.dt)
        table.update(seed=task.cfg['args'].seed,num_agents=task.num_agents,num_envs=task.num_envs,
                     simulated_steps=tracker.step,policy='SCT_26k_plus_FSM',tcoop_status='provisional; raw trajectories saved')
        metric = ('Enabling Interaction Group Success Rate' if setting.startswith('enable_')
                  else 'Coupled Interaction Group Success Rate')
        formal = result['interaction_group_scores']['region'][metric]
        if (formal['success'],formal['groups'])!=(table['successes'],table['trials']):
            raise ValueError('FSM Table2 success differs from main scoring')
        np.savez_compressed(path/'table2_trials.npz',**tracker.arrays())
        if hands:
            np.savez_compressed(path/'cooperation.npz',hands=np.stack(hands),physical_boxes=np.stack(physical),
                                physical_sizes=physical_sizes.cpu().numpy())
        (path/'table2.json').write_text(json.dumps(table,indent=2))
        result['table2'] = table
    (path/'result.json').write_text(json.dumps(result,indent=2))
    return result


formal_eval.run_formal_eval = evaluate
runpy.run_path(str(ROOT/'tokenhsi/run.py'),run_name='__main__')

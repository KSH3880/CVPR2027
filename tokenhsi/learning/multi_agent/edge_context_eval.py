"""Evaluate explicit required-goal edges, without a two-edges-per-agent assumption."""
import datetime
import json
import os
import torch
from env.tasks.multi_agent.edge_context_reward import scene_success, owner_sum
from utils.edge_ontop_spec import batched


@torch.no_grad()
def run_edge_context_eval(player):
    task = player.env.task
    N, M = task.num_envs, task.num_agents
    graph = task.relation_runtime.graph
    results = {}
    distance_audit = os.environ.get('TOKENHSI_PUSH_DISTANCE_EVAL') == '1' and getattr(task, '_open_push', False)
    if distance_audit and os.environ.get('TOKENHSI_EVAL_TRAIN_RSI') == '1':
        raise ValueError('Distance evaluation requires locomotion starts, without training RSI')
    for repeat in range(task.cfg['args'].eval_repeats):
        obs = player.env_reset(); player.get_batch_size(obs['obs'], 1)
        alive = torch.ones(N, dtype=torch.bool, device=player.device)
        final = torch.zeros(N, M, dtype=torch.bool, device=player.device)
        ever = torch.zeros_like(final)
        final_scene = torch.zeros_like(alive); ever_scene = torch.zeros_like(alive)
        final_edges = torch.zeros(N, len(graph.ids), dtype=torch.bool, device=player.device)
        required_owners = owner_sum(batched(graph.required_goal & graph.edge_valid, N).long(), graph).bool().clone()
        edge_valid = batched(graph.edge_valid, N).clone()
        edge_relation = batched(graph.edge_relation, N).clone()
        terminated = torch.zeros_like(alive); resets = []
        if distance_audit:
            if task._rsi_interaction.any():
                raise ValueError('Distance evaluation unexpectedly started from interaction RSI')
            initial_distance = (task._owned_push_states()[..., :2]-task._tar_pos[..., :2]).norm(dim=-1).clone()
            initial_approach = (task._owned_push_states()[..., :2]-task._humanoid_root_states[..., :2]).norm(dim=-1).clone()
            final_distance = torch.zeros_like(initial_distance)
            first_success = torch.full_like(initial_distance, -1.)
        face_audit = None
        if os.environ.get('TOKENHSI_PUSH_FACE_AUDIT') == '1' and getattr(task, '_open_push', False):
            from utils.push_face_audit import PushFaceAudit
            face_audit = PushFaceAudit(task._owned_push_states().detach().cpu().tolist(),
                task._tar_pos.detach().cpu().tolist(), task.cfg['env']['pushBox']['size'], task.dt)
        audit = os.environ.get('TOKENHSI_PUSH_EVAL_AUDIT') == '1' and getattr(task, '_open_push', False)
        if audit:
            original_goals = task._tar_pos.clone()
            sums = {}
            ever_contact = torch.zeros_like(final)
            initial_interaction_rate = task._rsi_interaction.float().mean().item()
            goal_drift = torch.zeros((), device=player.device)
            def count(name, value):
                sums[name] = sums.get(name, 0) + value.sum()
        for step in range(task.max_episode_length + 1):
            obs = player.env_reset(resets)
            action = player.get_action(obs, player.is_determenistic)
            obs, _, done, info = player.env_step(player.env, action)
            current = task.relation_runtime.done
            if distance_audit:
                first_success[(first_success < 0) & current & alive[:, None]] = (step+1)*task.dt
            if face_audit is not None:
                face_audit.step(task._owned_push_states().detach().cpu().tolist(),
                    task._tar_pos.detach().cpu().tolist(),
                    task._rigid_body_pos[:, :, task._key_body_ids[:2]].detach().cpu().tolist(),
                    current.detach().cpu().tolist(), alive.detach().cpu().tolist(),
                    task._contact_forces[:, :, task._key_body_ids[:2]].detach().cpu().tolist(),
                    task._zone_values(task._box_forces).detach().cpu().tolist())
            if audit:
                active = alive[:, None].expand(N, M)
                contact = task.relation_runtime.push_contact
                moving = (task._owned_push_states()[...,7:10].norm(dim=-1) > .02) & ~current
                ever_contact |= contact & active
                for phase, selected in (('all',active),('not_completed',active & ~current),
                                        ('moving',active & moving),('completed',active & current)):
                    count(phase+'_steps', selected)
                    count(phase+'_contact_steps', selected & contact)
                count('completed_without_contact_steps', active & current & ~contact)
                terms = info['reward_terms'].reshape(N,M,-1)[...,:3]
                count('completed_reward_saturation_mismatch_steps', active & current &
                    ((terms-.4).abs().amax(-1) > 1e-5))
                goal_drift = torch.maximum(goal_drift,
                    ((task._tar_pos-original_goals).norm(dim=-1)*active).amax())
            scene = scene_success(task.relation_runtime.own_success, graph)
            ever |= current & alive[:, None]; ever_scene |= scene & alive
            ending = done.reshape(N, M).bool().any(-1)
            collect = ending & alive
            if distance_audit:
                final_distance[collect] = (task._owned_push_states()[..., :2]-task._tar_pos[..., :2]).norm(dim=-1)[collect]
            final[collect] = current[collect]; final_scene[collect] = scene[collect]
            final_edges[collect] = task.relation_runtime.own_success[collect]
            terminated[collect] = info['terminate'].reshape(N, M)[collect].bool().any(-1)
            alive &= ~ending
            resets = ending.nonzero(as_tuple=False).flatten() * M
            if not alive.any():
                break
        if alive.any():
            raise RuntimeError('Evaluation exceeded horizon')
        metrics = dict(num_scene_trials=N, num_edges=len(graph.ids),
            final_agent_goal_success=final[required_owners].float().mean().item() if required_owners.any() else None,
            ever_agent_goal_success=ever[required_owners].float().mean().item() if required_owners.any() else None,
            final_scene_goal_success=final_scene.float().mean().item(), ever_scene_goal_success=ever_scene.float().mean().item(),
            final_edge_own_success={name: (final_edges & edge_valid & (edge_relation == rel)).sum().item() / max(1, (edge_valid & (edge_relation == rel)).sum().item()) for rel,name in ([(6,'holding'),(7,'at'),(8,'ontop'),(9,'sit'),(10,'climb')] if getattr(task,'_edge_interaction',False) else [(6,'holding'),(7,'at'),(8,'ontop')])} if getattr(task,'_edge_ontop',False) else dict(zip(graph.ids,final_edges.float().mean(0).tolist())),
            scene_termination_rate=terminated.float().mean().item())
        results['repeat_' + str(repeat)] = metrics
        if distance_audit:
            data = [x.detach().cpu().tolist() for x in
                    (initial_distance, initial_approach, final_distance, first_success, ever, final)]
            metrics['push_distance_audit'] = dict(start_mode='locomotion', agents=[dict(
                scene=i, agent=j, initial_goal_distance_m=data[0][i][j],
                initial_agent_box_distance_m=data[1][i][j], final_goal_distance_m=data[2][i][j],
                first_success_seconds=data[3][i][j] if data[3][i][j] >= 0 else None,
                ever_success=data[4][i][j], final_success=data[5][i][j])
                for i in range(N) for j in range(M)])
        if face_audit is not None:
            metrics['push_face_audit'] = face_audit.result()
        if audit:
            metrics['push_audit'] = {name: value.item() for name,value in sums.items()}
            metrics['push_audit'].update(
                ever_paired_contact_agent_rate=ever_contact.float().mean().item(),
                ever_goal_without_any_paired_contact_agent_rate=(ever & ~ever_contact).float().mean().item(),
                max_goal_drift_m=goal_drift.item(),
                contact_definition='same-physics-tick paired XY proximity and goal-direction force proxy; per-control-step OR',
                initial_interaction_agent_rate=initial_interaction_rate,
                start_mode='training RSI mixture' if os.environ.get('TOKENHSI_EVAL_TRAIN_RSI') == '1' else 'evaluation locomotion; training interaction RSI is disabled in eval')
        print('[edge context evaluation]', metrics, flush=True)
        if distance_audit:
            from utils.push_distance_eval import summarize_distance_results
            directory = os.path.join(task.cfg['args'].output_path, 'metrics')
            os.makedirs(directory, exist_ok=True)
            for name, value in (('distance_completed_results.json', results),
                                ('distance_summary.json', summarize_distance_results(results))):
                path = os.path.join(directory, name)
                temporary = path + '.tmp'
                with open(temporary, 'w') as f:
                    json.dump(value, f, indent=2)
                os.replace(temporary, path)
    directory = os.path.join(task.cfg['args'].output_path, 'metrics'); os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, 'edge_context_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved edge context evaluation:', path)
    return results

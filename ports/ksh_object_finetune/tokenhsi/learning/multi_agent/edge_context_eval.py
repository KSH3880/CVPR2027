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
        for _ in range(task.max_episode_length + 1):
            obs = player.env_reset(resets)
            action = player.get_action(obs, player.is_determenistic)
            obs, _, done, info = player.env_step(player.env, action)
            current = task.relation_runtime.done
            scene = scene_success(task.relation_runtime.own_success, graph)
            ever |= current & alive[:, None]; ever_scene |= scene & alive
            ending = done.reshape(N, M).bool().any(-1)
            collect = ending & alive
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
        print('[edge context evaluation]', metrics, flush=True)
    directory = os.path.join(task.cfg['args'].output_path, 'metrics'); os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, 'edge_context_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved edge context evaluation:', path)
    return results

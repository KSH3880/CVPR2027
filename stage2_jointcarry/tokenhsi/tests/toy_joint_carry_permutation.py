import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path

import isaacgym
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.multi_agent.ma_players import MAPlayerContinuous


def swap_observations(obs):
    result = obs.clone()
    result[:, :446] = obs[:, :446].reshape(-1, 2, 223).flip(1).flatten(1)
    result[:, 566:568] = obs[:, 566:568].flip(1)
    order = torch.tensor([1, 0, 2, 3, 4, 5, 7, 6], device=obs.device)
    result[:, 568:624] = obs[:, 568:624].reshape(-1, 8, 7)[:, order].flatten(1)
    packet = obs[:, 624:].reshape(-1, 2, 5).flip(1).clone()
    packet[..., 2:] = order[packet[..., 2:].long()]
    result[:, 624:] = packet.flatten(1)
    return result


def rebuilt_swap(task):
    arrays = [task._rigid_body_pos, task._rigid_body_rot, task._rigid_body_vel,
              task._rigid_body_ang_vel, task._tar_pos]
    saved = [value.clone() for value in arrays]
    graph = task.relation_runtime.graph
    order = torch.tensor([1, 0, 2, 3, 4, 5, 7, 6], device=task.device)
    try:
        for value, original in zip(arrays, saved):
            value.copy_(original.flip(1))
        task.relation_runtime.graph = replace(graph, edge_src=order[graph.edge_src],
            edge_dst=order[graph.edge_dst], edge_owner=1-graph.edge_owner)
        return task._compute_clean_scene_obs().clone()
    finally:
        for value, original in zip(arrays, saved):
            value.copy_(original)
        task.relation_runtime.graph = graph


@torch.no_grad()
def evaluate(player):
    task = player.env.task
    n = task.num_envs
    obs = player.env_reset(); player.get_batch_size(obs['obs'], 1)
    net = player.model.a2c_network
    encoder = net.actor_encoder
    forward = encoder.forward
    shuffle = net.coordination.shuffle_task_order
    rms_before = {k:v.clone() for k,v in player.running_mean_std.state_dict().items()}
    scores = {}
    identity = torch.arange(8, device=task.device)
    human_order = torch.tensor([1, 0, 2, 3, 4, 5, 7, 6], device=task.device)
    state_count = 0
    alive = torch.ones(n, dtype=torch.bool, device=task.device)
    initial_distance = (task._logical_box_values(task._box_states)[:, 0, :2]
                        - task._tar_pos[:, 0, :2]).norm(dim=-1)

    def compare(name, actual, expected):
        delta = actual - expected
        if not torch.isfinite(delta).all():
            raise RuntimeError('Nonfinite comparison: '+name)
        old = scores.setdefault(name, dict(max_abs=0., sum_squared=0., elements=0))
        old['max_abs'] = max(old['max_abs'], delta.abs().max().item())
        old['sum_squared'] += delta.double().square().sum().item()
        old['elements'] += delta.numel()

    def action(raw, order):
        encoder.forward = lambda value, **kwargs: forward(value, token_order=order, **kwargs)
        return player.get_action({'obs':raw.clone()}, True).reshape(n, 2, -1)

    try:
        for step in range(181):
            if step % 15 == 0:
                raw = task._compute_clean_scene_obs().clone()
                if raw.shape[1] != 634:
                    raise ValueError('This diagnostic expects the 2-human/4-object schema-9 scene layout')
                swapped = swap_observations(raw)
                compare('rebuilt_observation', rebuilt_swap(task), swapped)
                norm = player._preproc_obs(raw.clone())
                swapped_norm = player._preproc_obs(swapped.clone())
                compare('normalization', swapped_norm, swap_observations(norm))
                net.coordination.shuffle_task_order = False
                base_h, base_nodes = forward(norm, return_all=True, token_order=identity)
                swap_h, swap_nodes = forward(swapped_norm, return_all=True, token_order=identity)
                compare('encoder_humans', swap_h, base_h.flip(1))
                compare('encoder_all_nodes', swap_nodes, base_nodes[:, human_order])
                base_context = net.coordination(base_h, base_nodes, norm[:, 624:], encoder.edge_encoder)
                swap_context = net.coordination(swap_h, swap_nodes, swapped_norm[:, 624:], encoder.edge_encoder)
                compare('ca_context', swap_context, base_context.flip(1))
                base_mu = net.action_head(torch.cat((base_h, base_context), -1))
                swap_mu = net.action_head(torch.cat((swap_h, swap_context), -1))
                compare('unclipped_mean_action', swap_mu, base_mu.flip(1))
                baseline = action(raw, identity)
                compare('full_player_human_swap', action(swapped, identity), baseline.flip(1))
                compare('sa_token_order', action(raw, torch.randperm(8, device=task.device)), baseline)
                packet_swap = raw.clone()
                packet_swap[:, 624:] = raw[:, 624:].reshape(n, 2, 5).flip(1).flatten(1)
                compare('task_packet_order', action(packet_swap, identity), baseline)
                net.coordination.shuffle_task_order = True
                for _ in range(3):
                    order = torch.randperm(8, device=task.device)
                    compare('sa_ca_shuffles', action(raw, order), baseline)
                    compare('human_swap_with_shuffles', action(swapped, order), baseline.flip(1))
                state_count += n
                encoder.forward = forward
                net.coordination.shuffle_task_order = shuffle
            if step == 180:
                break
            actions = player.get_action(obs, True)
            obs, _, done, _ = player.env_step(player.env, actions)
            obs = player.obs_to_torch(obs)
            alive &= ~done.reshape(n, 2).bool().any(-1)
    finally:
        encoder.forward = forward
        net.coordination.shuffle_task_order = shuffle

    for name, values in scores.items():
        values['rmse'] = (values.pop('sum_squared') / values['elements'])**.5
        values['below_5e_5'] = values['max_abs'] < 5e-5
    distance = (task._logical_box_values(task._box_states)[:, 0, :2]
                - task._tar_pos[:, 0, :2]).norm(dim=-1)
    report = dict(checkpoint=task.cfg['args'].checkpoint, environments=n, steps=180,
        seconds=180*task.dt, observed_scene_states=state_count,
        joint_environments=int(task._joint_env_mask.sum()),
        independent_environments=int((~task._joint_env_mask).sum()),
        normalized_policy_comparisons=scores,
        rms_unchanged=all(torch.equal(v, player.running_mean_std.state_dict()[k]) for k,v in rms_before.items()),
        finite_model=all(torch.isfinite(v).all().item() for v in player.model.state_dict().values()),
        surviving_scenes=int(alive.sum()),
        joint_initial_distance=float(initial_distance[task._joint_env_mask].mean()),
        joint_final_distance=float(distance[task._joint_env_mask].mean()),
        scope='Counterfactual policy queries on identical simulator states; not swapped-physics rollouts or broad generalization validation')
    output = Path(task.cfg['args'].output_path); output.mkdir(parents=True, exist_ok=True)
    (output/'permutation_check.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PERMUTATION_CHECK', json.dumps(report), flush=True)


if __name__ == '__main__':
    MAPlayerContinuous.run_eval = evaluate
    runpy.run_path(str(ROOT/'run.py'), run_name='__main__')

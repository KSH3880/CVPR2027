import hashlib
import json
import runpy
import sys
from pathlib import Path

import isaacgym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning.multi_agent.ma_players import MAPlayerContinuous
from env.tasks.multi_agent.joint_carry_reward import anchor_scores, cpa_collision_penalty
from env.tasks.multi_agent.edge_context_reward import owner_sum
from utils.edge_context_spec import HOLDING


def model_digest(player):
    digest = hashlib.sha256()
    for value in player.model.state_dict().values():
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def summarize(a, dt):
    alive = np.broadcast_to(a['alive'][..., None], a['holding'].shape)
    unheld = ~a['holding']
    before_joint = ~a['coupled'][..., None]
    phases = {
        'all': alive,
        'first_2_seconds': alive & (np.arange(len(alive))[:, None, None]*dt < 2.),
        'unheld_approaching': alive & unheld & (a['toward_box'] > .15),
        'unheld_near_approaching': alive & unheld & (a['toward_box'] > .15) & (a['root_box_distance'] < 1.5),
        'before_joint_near_stopped': alive & before_joint & (a['root_speed'] < .1) & (a['root_box_distance'] < 1.5),
        'unheld_hand_near': alive & unheld & (a['hand_anchor_distance'] < .4),
        'coupled': alive & a['coupled'][..., None],
    }
    result = {}
    for name, mask in phases.items():
        if not mask.any():
            result[name] = {'agent_steps': 0}
            continue
        penalty = -a['collision'][mask]
        dense = (a['holding_state'] + a['holding_progress'])[mask]
        row = dict(agent_steps=int(mask.sum()), penalized_fraction=float((penalty > .001).mean()),
                   cpa_mean=float(penalty.mean()), cpa_p95=float(np.quantile(penalty, .95)),
                   cpa_max=float(penalty.max()), cpa_exceeds_holding_dense_fraction=float((penalty > dense).mean()),
                   scenes_with_cpa_over_0_2=int((mask & (-a['collision'] > .2)).any((0, 2)).sum()),
                   negative_dense_after_cpa_fraction=float((dense-penalty < 0).mean()))
        for key in ('holding_state', 'holding_progress', 'holding_success', 'root_box_distance',
                    'hand_anchor_distance', 'task_on', 'task_off', 'combined_on', 'combined_off'):
            row[key+'_mean'] = float(a[key][mask].mean())
        result[name] = row
    active = a['alive']
    ever = (a['coupled'] & active).any(0)
    first = np.argmax(a['coupled'] & active, axis=0)*dt
    result['scenes'] = dict(count=active.shape[1], ever_coupled=int(ever.sum()),
        first_coupled_seconds_mean=float(first[ever].mean()) if ever.any() else None,
        any_individual_holding=int((a['holding'] & active[..., None]).any((0, 2)).sum()),
        mean_observed_seconds=float(active.sum(0).mean()*dt),
        observed_root_distance_below_0_7_fraction=float((a['root_separation'][active] < .7).mean()))
    return result


@torch.no_grad()
def evaluate(player):
    task = player.env.task
    before = model_digest(player)
    rms = {k:v.clone() for k,v in player.running_mean_std.state_dict().items()}
    obs = player.env_reset()
    player.get_batch_size(obs['obs'], 1)
    n, m = task.num_envs, task.num_agents
    if not task._joint_env_mask.all() or m != 2:
        raise ValueError('Use a joint preset with two humans')
    config = task._relation_cfg['joint_carry']['collision']
    coefficient = task._agent_collision_coeff
    collision_index = task.REWARD_TERM_NAMES.index('collision')
    output = Path(task.cfg['args'].output_path)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(checkpoint=task.cfg['args'].checkpoint, coefficient=coefficient,
                  distance=task._agent_collision_dist, discount=config['ttc_discount'],
                  dt=task.dt, steps=600, seed=4242, presets={})
    for preset in ('joint_carry_at', 'joint_carry_ontop'):
        torch.manual_seed(4242)
        np.random.seed(4242)
        task._task_graph_preset = preset
        task._reset_envs(torch.arange(n, device=task.device))
        if not (task._joint_reset.skill_index == task._skill.index('loco')).all():
            raise ValueError('This diagnostic requires loco-only initial states')
        obs = player.obs_to_torch(player.env._policy_observation())
        alive = torch.ones(n, device=task.device, dtype=torch.bool)
        record = {}
        def save(key, value):
            record.setdefault(key, []).append(value.cpu().numpy().copy())
        for step in range(600):
            action = player.get_action(obs, True)
            obs, reward, done, info = player.env_step(player.env, action)
            obs = player.obs_to_torch(obs)
            runtime = task.relation_runtime
            graph, result = runtime.graph, runtime.last_result
            terms = info['reward_terms'].reshape(n, m, -1)
            roots = task._humanoid_root_states
            collision = -coefficient*cpa_collision_penalty(roots[..., :3], roots[..., 7:10],
                task._agent_collision_dist, config['ttc_discount'], task.dt)
            torch.testing.assert_close(collision, terms[..., collision_index])
            torch.testing.assert_close(terms[..., :-1].sum(-1), terms[..., -1], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(reward.reshape(n, m), terms[..., -1])
            mask = graph.edge_valid & (graph.edge_relation == HOLDING)
            holding = owner_sum(result['own_success'].float()*mask, graph).bool()
            boxes = task._logical_box_values(task._box_states)
            sizes = task._logical_box_values(task._box_size)
            hands = task._rigid_body_pos[..., task._key_body_ids[[0, 1]], :]
            _, anchors = anchor_scores(hands, boxes[:, 0], sizes[:, 0], .15)
            hand_distance = (hands.mean(-2)[:, :, None]-anchors[:, None]).norm(dim=-1).amin(-1)
            delta = boxes[:, 0, None, :2]-roots[..., :2]
            distance = delta.norm(dim=-1)
            toward = (delta*roots[..., 7:9]).sum(-1)/distance.clamp_min(1e-6)
            amp = player._calc_amp_rewards(info['amp_obs'])['disc_rewards'].reshape(n, m)
            on = terms[..., -1]
            off = on-collision
            for key, value in dict(alive=alive, holding=holding, coupled=runtime.coupled_holding,
                    collision=collision, task_on=on, task_off=off,
                    combined_on=player._task_reward_w*on+player._disc_reward_w*amp,
                    combined_off=player._task_reward_w*off+player._disc_reward_w*amp,
                    root_box_distance=distance, toward_box=toward,
                    root_speed=roots[..., 7:9].norm(dim=-1), hand_anchor_distance=hand_distance,
                    root_separation=(roots[:, 0, :2]-roots[:, 1, :2]).norm(dim=-1)).items():
                save(key, value)
            for key in ('state', 'progress', 'success'):
                save('holding_'+key, owner_sum(result[key+'_component']*mask, graph))
            alive &= ~done.reshape(n, m).bool().any(-1)
            if step % 150 == 149:
                print(preset, 'steps', step+1, 'alive', int(alive.sum()), flush=True)
        arrays = {k:np.stack(v) for k,v in record.items()}
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise RuntimeError('Nonfinite diagnostic')
        np.savez_compressed(output/(preset+'.npz'), **arrays)
        report['presets'][preset] = summarize(arrays, task.dt)
    probes = []
    for separation in (2., 1.5, 1.2):
        for speed in (.5, .1, 0.):
            position = torch.tensor([[[-separation/2, 0., 0.], [separation/2, 0., 0.]]], device=task.device)
            velocity = torch.tensor([[[speed, 0., 0.], [-speed, 0., 0.]]], device=task.device)
            penalty = coefficient*cpa_collision_penalty(position, velocity, task._agent_collision_dist,
                                                       config['ttc_discount'], task.dt)
            probes.append(dict(root_distance=separation, inward_speed_each=speed, penalty_each=float(penalty[0, 0])))
    report['constant_velocity_probes_not_physics_rollouts'] = probes
    report['model_unchanged'] = before == model_digest(player)
    report['rms_unchanged'] = all(torch.equal(v, player.running_mean_std.state_dict()[k]) for k,v in rms.items())
    if not report['model_unchanged'] or not report['rms_unchanged']:
        raise RuntimeError('Evaluation changed model or normalization')
    report['scope'] = 'Identical-state reward counterfactual only; no CPA-off training or causal policy improvement test. Samples after first termination excluded.'
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('CPA_CHECK', json.dumps(report), flush=True)


if __name__ == '__main__':
    MAPlayerContinuous.run_eval = evaluate
    runpy.run_path(str(ROOT/'run.py'), run_name='__main__')

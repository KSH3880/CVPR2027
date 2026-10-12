import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

import isaacgym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tokenhsi'))

from utils import config as runtime_config
from utils import mixed_carry_spec
from env.tasks.multi_agent.joint_carry_reset import JointCarryReset
from env.tasks.multi_agent.joint_carry_reward import anchor_scores, opposite_gate
from learning.multi_agent.ma_agent import MAAgent
from learning.multi_agent.ma_players import MAPlayerContinuous


def heading(quaternion):
    x, y, z, w = quaternion.unbind(-1)
    yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return torch.stack((yaw.cos(), yaw.sin()), -1)


def digest(values):
    result = hashlib.sha256()
    for value in values:
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def install(toy):
    original_load = runtime_config.load_cfg
    original_sample = mixed_carry_spec.sample_graph
    original_init = JointCarryReset.__init__
    original_reset = JointCarryReset.reset
    original_train = MAAgent.train
    original_stats = MAAgent.get_stats_weights
    spec = dict(version=1, task='joint_at', rsi='held_carryWith', goal_distance=1.5,
                goal_directions='initial_human_A_forward_backward_balanced',
                zero_initial_velocities=True, amp_reward_weight=toy.toy_amp_weight,
                task_reward_weight=0.5, updates=toy.toy_updates)

    def load_cfg(args):
        cfg, train, logdir = original_load(args)
        cfg['env']['jointCarryToy'] = spec
        cfg['env']['debug']['reward'] = False
        train['params']['config']['disc_reward_w'] = toy.toy_amp_weight
        train['params']['config']['task_reward_w'] = 0.5
        train['params']['config']['save_frequency'] = 50
        return cfg, train, logdir

    def environment_mask(n, device, preset='random_scenario'):
        return torch.ones(n, device=device, dtype=torch.bool)

    def sample_graph(n, graph_spec, device='cpu', preset='random_scenario',
                     role_swap=False, generator=None, joint_mask=None):
        return original_sample(n, graph_spec, device, 'joint_carry_at', role_swap,
                               generator, environment_mask(n, device))

    def initialize(reset, task):
        original_init(reset, task)
        pool = reset.pools['carryWith']
        hands = pool['body_state'][..., task._key_body_ids[:2], :3]
        sizes = pool['box_state'].new_tensor([.52, .80, .40]).expand(len(hands), -1)
        scores, _ = anchor_scores(hands, pool['box_state'][:, 0], sizes, .15)
        facing = heading(pool['root_state'][..., 3:7])
        mask = opposite_gate(scores, .9) & ((facing[:, 0] * facing[:, 1]).sum(-1) < -.9)
        if not mask.any():
            raise RuntimeError('No paired carryWith snapshots satisfy opposite HOLDING')
        reset.pools['carryWith'] = {key: value[mask] for key, value in pool.items()}
        task._toy_reset_count = torch.zeros(task.num_envs, device=task.device, dtype=torch.long)
        task._toy_sign = torch.zeros_like(task._toy_reset_count)
        task._toy_initial_checked = False
        print('TOY_POOL', int(mask.sum()), '/', len(mask), flush=True)

    def reset_scene(reset, ids, finalize=True):
        if not len(ids):
            return
        task = reset.task
        index = task._skill.index('carryWith')
        if task._is_eval:
            task._skill_init_prob[:] = 0
            task._skill_init_prob[index] = 1
        else:
            task._template_rsi_weights[3:5] = 0
            task._template_rsi_weights[3:5, index] = 1
        original_reset(reset, ids, finalize=False)
        sign = 1 - 2 * ((ids + task._toy_reset_count[ids]) % 2)
        task._toy_sign[ids] = sign
        task._toy_reset_count[ids] += 1
        facing = heading(task._humanoid_root_states[ids, 0, 3:7])
        target = task._box_states[ids, 0, :3].clone()
        target[:, :2] += 1.5 * sign[:, None] * facing
        target[:, 2] = task._box_size[ids, 0, 2] / 2
        task._tar_pos[ids] = target[:, None]
        for state in (task._humanoid_root_states, task._kinematic_humanoid_rigid_body_states,
                      task._box_states):
            state[ids, ..., 7:13] = 0
        task._dof_vel[ids] = 0
        # Keep unused props clear of either transport direction.
        for box, xy in ((1, (8., 8.)), (2, (-8., 8.)), (3, (8., -8.))):
            task._box_states[ids, box, :2] = task._env_origins[ids, :2] + target.new_tensor(xy)
        if not task._toy_initial_checked and len(ids) == task.num_envs:
            hands = task._kinematic_humanoid_rigid_body_states[ids][..., task._key_body_ids[:2], :3]
            scores, _ = anchor_scores(hands, task._box_states[ids, 0], task._box_size[ids, 0], .15)
            valid = opposite_gate(scores, .9)
            if not valid.all():
                raise RuntimeError('Toy reset failed simultaneous HOLDING')
            graph = task.relation_runtime.graph
            state_hash = digest([task._humanoid_root_states, task._box_states,
                task._dof_pos, task._dof_vel, task._tar_pos, graph.edge_relation, graph.edge_owner])
            report = dict(spec=spec, initial_state_sha256=state_hash, environments=task.num_envs,
                          holding_fraction=float(valid.float().mean()),
                          forward=int((sign > 0).sum()), backward=int((sign < 0).sum()))
            out = Path(task.cfg['args'].output_path)
            out.mkdir(parents=True, exist_ok=True)
            (out / 'initial_check.json').write_text(json.dumps(report, indent=2))
            task._toy_initial_checked = True
            print('TOY_INITIAL', json.dumps(report), flush=True)
        if finalize:
            reset.commit(ids)
            task._refresh_sim_tensors()
            task._reset_relation_history(ids)
            task._compute_observations(ids)
            task._compute_amp_observations(ids)
            task._hist_amp_obs_buf[ids] = task._curr_amp_obs_buf[ids, :, None]

    def train(agent):
        start_epoch = agent.epoch_num
        agent.max_epochs = start_epoch + toy.toy_updates - 1
        agent.config['max_epochs'] = agent.max_epochs
        before = {key: value.clone() for key, value in agent.model.state_dict().items()
                  if key.startswith('a2c_network.actor_encoder.')}
        rms = {key: value.clone() for key, value in agent.running_mean_std.state_dict().items()}
        result = original_train(agent)
        after = agent.model.state_dict()
        unchanged = all(torch.equal(value, after[key]) for key, value in before.items())
        rms_unchanged = all(torch.equal(value, agent.running_mean_std.state_dict()[key]) for key, value in rms.items())
        report = dict(start_epoch=start_epoch, end_epoch=agent.epoch_num,
                      updates=agent.epoch_num-start_epoch, encoder_unchanged=unchanged,
                      observation_rms_unchanged=rms_unchanged,
                      model_finite=all(torch.isfinite(v).all().item() for v in after.values()), spec=spec)
        out = Path(agent.vec_env.env.task.cfg['args'].output_path)
        (out / 'training_check.json').write_text(json.dumps(report, indent=2))
        print('TOY_TRAINING', json.dumps(report), flush=True)
        return result

    def stats(agent):
        result = original_stats(agent)
        result['joint_carry_toy'] = spec
        return result

    runtime_config.load_cfg = load_cfg
    mixed_carry_spec.environment_mask = environment_mask
    mixed_carry_spec.sample_graph = sample_graph
    JointCarryReset.__init__ = initialize
    JointCarryReset.reset = reset_scene
    MAAgent.train = train
    MAAgent.get_stats_weights = stats


@torch.no_grad()
def evaluate(player):
    task = player.env.task
    torch.manual_seed(4242)
    np.random.seed(4242)
    task._toy_initial_checked = False
    obs = player.env_reset()
    player.get_batch_size(obs['obs'], 1)
    n, m = task.num_envs, task.num_agents
    sign = task._toy_sign.clone()
    target = task._tar_pos[:, 0, :3].clone()
    origin = task._env_origins.clone()
    alive = torch.ones(n, device=task.device, dtype=torch.bool)
    initial_distance = (task._box_states[:, 0, :2] - target[:, :2]).norm(dim=-1)
    final_distance = initial_distance.clone()
    holding = torch.zeros(n, device=task.device)
    steps = torch.zeros_like(holding)
    ever = torch.zeros_like(alive)
    distance, gates, bodies, boxes, live = [], [], [], [], []
    for step in range(min(task.max_episode_length, 180)):
        action = player.get_action(obs, True)
        obs, _, done, info = player.env_step(player.env, action)
        obs = player.obs_to_torch(obs)
        current = (task._box_states[:, 0, :2] - target[:, :2]).norm(dim=-1)
        final_distance[alive] = current[alive]
        holding += task.relation_runtime.coupled_holding & alive
        steps += alive
        ever |= task.relation_runtime.done.all(-1) & alive
        distance.append(current.cpu().numpy())
        gates.append(task.relation_runtime.coupled_holding.cpu().numpy())
        bodies.append((task._rigid_body_pos - origin[:, None, None]).cpu().numpy())
        boxes.append((task._box_states[:, 0, :3] - origin).cpu().numpy())
        live.append(alive.cpu().numpy())
        alive &= ~done.reshape(n, m).bool().any(-1)
    report = dict(checkpoint=task.cfg['args'].checkpoint, control_dt=task.dt, steps=len(distance), groups={})
    for value, name in ((1, 'A_forward_B_backward'), (-1, 'A_backward_B_forward')):
        mask = sign == value
        gain = initial_distance - final_distance
        report['groups'][name] = dict(n=int(mask.sum()),
            distance_reduction_mean=float(gain[mask].mean()),
            distance_reduction_median=float(gain[mask].median()),
            approached_30cm=float((gain[mask] >= .3).float().mean()),
            holding_fraction=float((holding / steps.clamp_min(1))[mask].mean()),
            completed_horizon_fraction=float((steps[mask] >= min(task.max_episode_length-1, 180)).float().mean()),
            ever_success=float(ever[mask].float().mean()))
    out = Path(task.cfg['args'].output_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'evaluation.json').write_text(json.dumps(report, indent=2))
    names = task.gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles_all[0][0])
    np.savez_compressed(out / 'traces.npz', distance=np.stack(distance), holding=np.stack(gates),
        body=np.stack(bodies), box=np.stack(boxes), alive=np.stack(live),
        target=(target-origin).cpu().numpy(), sign=sign.cpu().numpy(), body_names=np.array(names))
    print('TOY_EVALUATION', json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--toy-amp-weight', type=float, choices=(0., .5), default=.5)
    parser.add_argument('--toy-updates', type=int, default=100)
    toy, remaining = parser.parse_known_args()
    install(toy)
    MAPlayerContinuous.run_eval = evaluate
    sys.argv = [str(ROOT / 'tokenhsi/run.py')] + remaining
    runpy.run_path(str(ROOT / 'tokenhsi/run.py'), run_name='__main__')

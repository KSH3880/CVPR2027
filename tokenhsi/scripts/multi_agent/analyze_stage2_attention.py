"""Measure checkpoint trends on matched rollout starts and a shared observation bank.

All graph/attention instrumentation is confined to diagnostic subprocesses.
Production attention, reward, training configuration and running jobs are untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = ROOT / 'output/approach_stage2_rescue_klclimb50/ApproachStage2RescueKLClimb50_01-20-37-44'
CONDITIONS = [('place_climb', 0), ('place_climb', 1), ('place_sit', 0),
              ('place_sit', 1), ('place_stack', 0), ('place_stack', 1), ('independent', 0)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--output', type=Path, default=ROOT / 'output/stage2_attention_trends')
    p.add_argument('--envs-per-condition', type=int, default=16)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--stride', type=int, default=10)
    p.add_argument('--seed', type=int, default=20261002)
    p.add_argument('--epochs', type=int, nargs='+', default=[500, 2000, 4000, 6000, 8000, 10000])
    p.add_argument('--worker', action='store_true')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--replay-all', action='store_true')
    p.add_argument('--render-only', action='store_true')
    p.add_argument('--intervention', choices=['none', 'ca_off', 'uniform', 'peer_removed'], default='none')
    a = p.parse_args()
    if min(a.envs_per_condition, a.repeats, a.steps, a.stride) < 1:
        p.error('environment/repeat/step/stride counts must be positive')
    return a


def worker(a):
    sys.path[:0] = [str(ROOT / 'tokenhsi'), str(ROOT)]
    from isaacgym import gymapi  # Must precede torch.
    import numpy as np
    np.float = float
    np.int = int
    import torch
    from learning.multi_agent.attention_probe import inspect_coordination, alter_context
    from learning.multi_agent import edge_context_eval
    import env.tasks.multi_agent.edge_ontop_task as integration
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    from env.tasks.multi_agent.edge_context_reward import scene_success
    from utils.edge_stage2_spec import sample_graph, _empty_graph, validate_graph
    from utils.edge_ontop_spec import copy_graph_rows, select_graph
    n = len(CONDITIONS) * a.envs_per_condition
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    epoch = int(checkpoint['epoch'])
    del checkpoint
    target = a.output / ('epoch_%05d' % epoch)
    target.mkdir(parents=True, exist_ok=True)
    original_sample = HumanoidMACarry._sample_episode_graph

    def sample_conditions(self, ids):
        if not hasattr(self, '_attention_probe_graph'):
            graph = _empty_graph(n, 2, 4, 4, self.device)
            generator = torch.Generator(device=self.device).manual_seed(a.seed + 71)
            for condition, (family, role) in enumerate(CONDITIONS):
                block = torch.arange(condition * a.envs_per_condition,
                                     (condition + 1) * a.envs_per_condition, device=self.device)
                sampled = sample_graph(len(block), self._relation_graph_spec, self.device,
                                       family, bool(role), generator)
                copy_graph_rows(graph, block, sampled)
            validate_graph(graph)
            self._attention_probe_graph = graph
        previous = integration.sample_stage1_graph
        integration.sample_stage1_graph = lambda *args, **kwargs: select_graph(self._attention_probe_graph, ids)
        try:
            original_sample(self, ids)
        finally:
            integration.sample_stage1_graph = previous

    HumanoidMACarry._sample_episode_graph = sample_conditions

    @torch.no_grad()
    def evaluate(player):
        task = player.env.task
        net = player.model.a2c_network
        captured = {}
        def hook(module, inputs, outputs):
            captured['humans'], captured['edges'] = inputs[:2]
            captured['context'] = outputs[0]
            if a.intervention != 'none':
                graph = task._attention_probe_graph
                changed = alter_context(module, *inputs[:2], graph.edge_valid, graph.edge_owner, a.intervention)
                return changed, outputs[1]
        handle = net.coordination.attention.register_forward_hook(hook)
        player.env_reset()
        graph = task._attention_probe_graph
        valid = graph.edge_valid
        dt = float(task.dt)
        samples, initial, trials = [], [], []
        max_error = 0.
        start = time.time()
        for repeat in range(a.repeats):
            seed = a.seed + repeat * 101
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            obs = player.env_reset(torch.arange(n, device=task.device) * 2)
            player.get_batch_size(obs['obs'], 1)
            initial.append(obs['obs'].cpu().numpy().copy())
            alive = torch.ones(n, dtype=torch.bool, device=task.device)
            ever = torch.zeros(n, 4, dtype=torch.bool, device=task.device)
            final = ever.clone()
            joint_ever = torch.zeros_like(alive)
            terminated = torch.zeros_like(alive)
            joint_run = torch.zeros(n, dtype=torch.long, device=task.device)
            max_joint_run = joint_run.clone()
            success_run = torch.zeros(n, 4, dtype=torch.long, device=task.device)
            max_success_run = success_run.clone()
            stable_run = joint_run.clone()
            first = torch.full((n, 4), -1., device=task.device)
            stable_at = torch.full((n,), -1., device=task.device)
            length = torch.zeros(n, dtype=torch.long, device=task.device)
            for step in range(a.steps):
                action = player.get_action(obs, True)
                if not torch.isfinite(action).all():
                    raise RuntimeError('Nonfinite rollout action')
                if step % a.stride == 0:
                    probe = inspect_coordination(net.coordination.attention, net.action_head,
                        captured['humans'], captured['edges'], valid, captured['context'])
                    max_error = max(max_error, probe.pop('reconstruction_error'))
                    if max_error > 2e-5:
                        raise RuntimeError('Attention reconstruction mismatch: %g' % max_error)
                    boxes = task._logical_box_values(task._box_states)
                    row = torch.arange(n, device=task.device)
                    source = (graph.edge_src[:, 1] - 2).clamp(0, 3)
                    goal = (graph.edge_dst[:, 1] - 6).clamp(0, 1)
                    state = dict(obs=obs['obs'], alive=alive, phi=task.relation_runtime.phi,
                                 success=task.relation_runtime.own_success,
                                 box_speed=boxes[row, source, 7:10].norm(dim=-1),
                                 goal_distance=(boxes[row, source, :3]-task._tar_pos[row, goal]).norm(dim=-1))
                    state.update(probe)
                    samples.append({key: value.cpu().numpy().copy() for key, value in state.items()})
                obs, _, done, info = player.env_step(player.env, action)
                if not isinstance(obs, dict):
                    obs = {'obs': obs}
                current = task.relation_runtime.own_success & alive[:, None] & valid
                joint = scene_success(current, graph) & alive
                ever |= current; joint_ever |= joint
                success_run = torch.where(current, success_run + 1, 0)
                max_success_run = torch.maximum(max_success_run, success_run)
                joint_run = torch.where(joint, joint_run + 1, 0)
                max_joint_run = torch.maximum(max_joint_run, joint_run)
                first = torch.where(current & (first < 0), (step + 1) * dt, first)
                boxes = task._logical_box_values(task._box_states)
                row = torch.arange(n, device=task.device)
                source = (graph.edge_src[:, 1] - 2).clamp(0, 3)
                ready = current[:, 1] & (boxes[row, source, 7:10].norm(dim=-1) < .1)
                stable_run = torch.where(ready, stable_run + 1, 0)
                stable_at = torch.where((stable_run * dt >= .5) & (stable_at < 0),
                                       (step + 1) * dt, stable_at)
                ending = done.reshape(n, 2).bool().any(-1) | (step == a.steps - 1)
                collect = ending & alive
                final[collect] = current[collect]
                length[collect] = step + 1
                terminated[collect] = info['terminate'].reshape(n, 2).bool().any(-1)[collect]
                alive &= ~ending
                # Keep completed scenes unreset; they are excluded from later samples.
                if not alive.any():
                    # Maintain a rectangular trace with explicit invalid padding.
                    for padding in range(step + 1, a.steps):
                        if padding % a.stride == 0:
                            blank = {key: np.zeros_like(value) for key, value in samples[-1].items()}
                            blank['alive'][:] = False
                            samples.append(blank)
                    break
            for env in range(n):
                condition = env // a.envs_per_condition
                family, role = CONDITIONS[condition]
                required = graph.required_goal[env] & valid[env]
                trials.append(dict(epoch=epoch, repeat=repeat, env=env, condition=condition,
                    family=family, carrier=role, joint_ever=bool(joint_ever[env]),
                    joint_final=bool((final[env] | ~required).all()), terminated=bool(terminated[env]),
                    joint_1s=bool(max_joint_run[env] * dt >= 1),
                    max_joint_seconds=float(max_joint_run[env] * dt), length_steps=int(length[env]),
                    edge_ever=ever[env].cpu().tolist(), edge_final=final[env].cpu().tolist(),
                    edge_max_seconds=(max_success_run[env] * dt).cpu().tolist(),
                    first_edge_seconds=first[env].cpu().tolist(), stable_at_seconds=float(stable_at[env])))
            print('[attention rollout] epoch=%d repeat=%d joint ever=%.3f elapsed=%.1fs' %
                  (epoch, repeat, joint_ever.float().mean(), time.time()-start), flush=True)
        per_repeat = len(range(0, a.steps, a.stride))
        arrays = {key: np.stack([sample[key] for sample in samples]).reshape(
            a.repeats, per_repeat, *samples[0][key].shape) for key in samples[0]}
        arrays.update(initial_obs=np.stack(initial), valid=valid.cpu().numpy(),
                      src=graph.edge_src.cpu().numpy(), dst=graph.edge_dst.cpu().numpy(),
                      owner=graph.edge_owner.cpu().numpy(), relation=graph.edge_relation.cpu().numpy(),
                      required=graph.required_goal.cpu().numpy(),
                      box_sizes=task._box_size.cpu().numpy())
        np.savez_compressed(target / 'traces.npz', **arrays)
        metadata = dict(epoch=epoch, checkpoint=str(a.checkpoint), sha256=digest(a.checkpoint),
                        intervention=a.intervention,
                        dt=dt, stride=a.stride, seed=a.seed, repeats=a.repeats,
                        envs_per_condition=a.envs_per_condition, steps=a.steps,
                        reconstruction_max_error=max_error, conditions=CONDITIONS, trials=trials)
        (target / 'results.json').write_text(json.dumps(metadata, indent=2))

        if a.replay_all:
            # Fixed-state comparison removes differing state visitation across epochs.
            mask = arrays['alive'].reshape(-1)
            bank = arrays['obs'].reshape(-1, arrays['obs'].shape[-1])[mask]
            indices = np.argwhere(arrays['alive'])
            bank_valid = arrays['valid'][indices[:, 2]]
            np.savez_compressed(a.output / 'observation_bank.npz', obs=bank, indices=indices,
                                valid=bank_valid)
            paths = sorted((a.run / 'nn').glob('*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].pth'))
            # Include the frozen requested pointer even if newer numbered files appear.
            paths = [path for path in paths if int(path.stem.rsplit('_', 1)[1]) <= epoch]
            replay_meta = []
            for path in paths:
                player.restore(str(path))
                player.model.eval(); player.running_mean_std.eval()
                outputs = []
                error = 0.
                for begin in range(0, len(bank), 256):
                    raw_obs = torch.tensor(bank[begin:begin+256], device=player.device)
                    norm_obs = player._preproc_obs(raw_obs)
                    net.eval_actor(norm_obs)
                    values = inspect_coordination(net.coordination.attention, net.action_head,
                        captured['humans'], captured['edges'],
                        torch.tensor(bank_valid[begin:begin+256], device=player.device), captured['context'])
                    error = max(error, values.pop('reconstruction_error'))
                    outputs.append({key: value.cpu().numpy() for key, value in values.items()})
                if error > 2e-5:
                    raise RuntimeError('Replay reconstruction mismatch')
                replay_epoch = int(path.stem.rsplit('_', 1)[1])
                replay = {key: np.concatenate([out[key] for out in outputs]) for key in outputs[0]}
                np.savez_compressed(a.output / ('replay_%05d.npz' % replay_epoch), **replay)
                replay_meta.append(dict(epoch=replay_epoch, checkpoint=str(path), sha256=digest(path),
                                        observations=len(bank), reconstruction_max_error=error))
                print('[attention replay] epoch=%d fixed observations=%d' % (replay_epoch, len(bank)), flush=True)
            (a.output / 'replay_manifest.json').write_text(json.dumps(replay_meta, indent=2))
        handle.remove()
        return metadata

    edge_context_eval.run_edge_context_eval = evaluate
    sys.argv = [str(ROOT / 'tokenhsi/run.py'), '--task', 'HumanoidMACarry',
        '--cfg_train', 'tokenhsi/data/cfg/train/rlg/amp_ma_stage2_rescue_klclimb50.yaml',
        '--cfg_env', 'tokenhsi/data/cfg/multi_agent/approach_stage2_rescue_klclimb50.yaml',
        '--motion_file', 'tokenhsi/data/dataset_loco_sit_carry_climb.yaml',
        '--checkpoint', str(a.checkpoint), '--num_agents', '2', '--num_envs', str(n),
        '--num_objects', '4', '--episode_length', str(a.steps), '--eval_skills', 'loco',
        '--eval_skill_probs', '1.0', '--output_path', str(target), '--seed', str(a.seed),
        '--task_graph', 'place_climb', '--headless', '--no_video', '--test', '--eval']
    runpy.run_path(str(ROOT / 'tokenhsi/run.py'), run_name='__main__')


def orchestrate(a):
    a.output.mkdir(parents=True, exist_ok=True)
    pointer = a.run / 'nn/ApproachStage2RescueKLClimb50.pth'
    frozen = a.output / 'requested_checkpoint.pth'
    if not frozen.exists():
        shutil.copyfile(pointer, frozen)
    paths = [(frozen, 'none')] + [(a.run / ('nn/ApproachStage2RescueKLClimb50_%08d.pth' % epoch), 'none') for epoch in a.epochs]
    paths += [(frozen, mode) for mode in ['ca_off', 'uniform', 'peer_removed']]
    for i, (path, mode) in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        destination = a.output if mode == 'none' else a.output / 'ablations' / mode
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--run', str(a.run),
                   '--output', str(destination), '--checkpoint', str(path), '--intervention', mode,
                   '--envs-per-condition', str(a.envs_per_condition), '--repeats', str(a.repeats),
                   '--steps', str(a.steps), '--stride', str(a.stride), '--seed', str(a.seed)]
        if i == 0:
            command.append('--replay-all')
        logfile = a.output / ('worker_%s.log' % ('latest' if i == 0 else path.stem+'_'+mode))
        print('[attention worker]', path.name, flush=True)
        with logfile.open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    args = parse_args()
    if args.worker:
        worker(args)
    else:
        if not args.render_only:
            orchestrate(args)
        sys.path.insert(0, str(Path(__file__).parent))
        from render_stage2_attention import render
        render(args.output)

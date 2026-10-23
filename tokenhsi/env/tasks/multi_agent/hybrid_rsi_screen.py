"""Startup PD screening against the actual immutable box assets (no policy)."""
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import yaml

from utils.hybrid_late_rsi import CARRY_SKILLS, phase_samples, profile_key
from utils.reference_rsi import digest, geometry_digest, interpolate_object_velocity


def screen_states(root, boxes, initial_root, initial_boxes, spec):
    return (torch.isfinite(root).all(-1) & torch.isfinite(boxes).all((-1, -2)) &
        ((root[:, 7:10]-initial_root[:, 7:10]).norm(dim=-1) <= spec['root_velocity_change']) &
        (boxes[..., 7:10].norm(dim=-1).amax(-1) <= spec['box_speed']) &
        ((root[:, :3]-initial_root[:, :3]).norm(dim=-1) <= spec['root_displacement']) &
        ((boxes[..., :3]-initial_boxes[..., :3]).norm(dim=-1).amax(-1) <= spec['box_displacement']))


def build_physical_records(env, spec, dataset, asset):
    """Cache keys include source+support size, data, physics and screening code."""
    from isaacgym import gymtorch
    identity = dict(version=1, spec=spec, sim=env.cfg['sim'],
        dataset=digest(dataset), asset=geometry_digest(asset),
        density=env._box_density.detach().cpu().tolist(),
        character={k: env.cfg['env'][k] for k in
            ('asset', 'plane', 'powerScale', 'enableSelfCollisionDetection')})
    motion_config = yaml.safe_load(Path(dataset).read_text())['motions']
    for skill in ('pickUp', 'carryWith', 'putDown'):
        lib = env._motion_lib[skill]
        identity[skill] = [(digest(f), digest(Path(dataset).parent / entry['obj_file']))
            for f, entry in zip(lib._motion_files, motion_config[skill])]
    identity['code'] = digest(__file__)
    identity['sampler'] = digest(Path(__file__).parents[3] / 'utils/hybrid_late_rsi.py')
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(spec['cache_directory']) / (fingerprint[:24] + '.json')
    records = {}
    if path.exists():
        payload = json.loads(path.read_text())
        if payload['fingerprint'] != fingerprint:
            raise ValueError('Hybrid physical cache fingerprint mismatch')
        for item in payload['profiles']:
            key = (item['skill'], tuple(item['source']))
            if item['support'] is not None:
                key += (tuple(item['support']),)
            records[key] = [tuple(candidate) for candidate in item['candidates']]
    sizes = env._box_size.detach().cpu().tolist()
    work = []
    pending = set(records)
    for kind in CARRY_SKILLS:
        for source in range(env.num_objects):
            targets = [None] if kind != 'putDownOnTop' else [
                i for i in range(env.num_objects) if i != source]
            for support in targets:
                profiles = {}
                for row in range(env.num_envs):
                    key = profile_key(kind, sizes[row][source],
                        sizes[row][support] if support is not None else None)
                    if key not in pending:
                        profiles[key] = row
                        pending.add(key)
                if profiles:
                    work.append((kind, source, support, profiles))
    if not work:
        print('[hybrid RSI] physical cache hit:', path, flush=True)
        return records, fingerprint
    print('[hybrid RSI] screening', sum(len(p[3]) for p in work),
          'actual source/support profiles; all four bands retained for policy fallback', flush=True)
    original_roots, original_dofs = env._root_states.clone(), env._dof_state.clone()
    # Screening must not consume training/reset RNG state.
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(env.device)
    try:
        for index, (kind, source, support, profiles) in enumerate(work):
            rows = torch.tensor(list(profiles.values()), device=env.device, dtype=torch.long)
            keys = list(profiles)
            result = [[] for _ in keys]
            lib = env._motion_lib['putDown' if kind == 'putDownOnTop' else kind]
            offset = env._agent_spawn_offsets[0] + env._env_origins[rows]
            for clip in range(lib.num_motions()):
                ids = torch.full_like(rows, clip)
                for band in range(4):
                    for phase in phase_samples(spec, band):
                        times = lib.get_motion_length(ids)*phase
                        root, rot, dofs, vel, ang, dof_vel, _ = lib.get_motion_state(ids, times)
                        box_pos, box_rot = lib.get_obj_motion_state(ids, times)
                        box_pos = box_pos + offset
                        floor = env._box_size[rows, source, 2]/2
                        clamped = box_pos[:, 2] < floor
                        box_pos[:, 2] = torch.maximum(box_pos[:, 2], floor)
                        linear, angular = interpolate_object_velocity(lib, ids, times)
                        linear[clamped, 2] = 0.
                        env._root_states.copy_(original_roots)
                        env._dof_state.copy_(original_dofs)
                        env._humanoid_root_states[..., :2] = env._env_origins[:, None, :2] + 15.
                        env._humanoid_root_states[..., 0] += torch.arange(env.num_agents, device=env.device)*3
                        env._humanoid_root_states[..., 7:13] = 0.
                        env._box_states[..., :2] = env._env_origins[:, None, :2] - 15.
                        env._box_states[..., 0] -= torch.arange(env.num_objects, device=env.device)*2
                        env._box_states[..., 2] = env._box_size[..., 2]/2
                        env._box_states[..., 3:7] = env._box_states.new_tensor([0., 0., 0., 1.])
                        env._box_states[..., 7:13] = 0.
                        env._set_env_state(rows, torch.zeros_like(rows), root+offset,
                                           rot, dofs, vel, ang, dof_vel)
                        env._box_states[rows, source, :3] = box_pos
                        env._box_states[rows, source, 3:7] = box_rot
                        env._box_states[rows, source, 7:10] = linear
                        env._box_states[rows, source, 10:13] = angular
                        checked = [source]
                        if support is not None:
                            end, _ = lib.get_obj_motion_state(ids, lib.get_motion_length(ids))
                            env._box_states[rows, support, :2] = end[:, :2]+offset[:, :2]
                            env._box_states[rows, support, 2] = env._box_size[rows, support, 2]/2
                            checked.append(support)
                        initial_root = env._humanoid_root_states[rows, 0].clone()
                        initial_boxes = env._box_states[rows][:, checked].clone()
                        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env._root_states))
                        env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env._dof_state))
                        env.gym.set_dof_position_target_tensor(env.sim,
                            gymtorch.unwrap_tensor(env._dof_pos.contiguous().clone()))
                        valid = torch.ones(len(rows), device=env.device, dtype=torch.bool)
                        for _ in range(math.ceil(spec['screen']['seconds']/env.sim_params.dt)):
                            env.gym.simulate(env.sim); env.gym.fetch_results(env.sim, True)
                            env._refresh_sim_tensors()
                            valid &= screen_states(env._humanoid_root_states[rows, 0],
                                env._box_states[rows][:, checked], initial_root, initial_boxes, spec['screen'])
                            valid &= torch.isfinite(env._dof_state.view(env.num_envs, env.num_agents, -1)[rows, 0]).all(-1)
                        for record, accepted in zip(result, valid.cpu().tolist()):
                            if accepted:
                                record.append((clip, phase, band))
            records.update(zip(keys, result))
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(fingerprint=fingerprint, profiles=[dict(skill=k[0],
                source=k[1], support=k[2] if len(k)>2 else None, candidates=v)
                for k, v in sorted(records.items())])
            temporary = path.with_suffix('.tmp.' + str(os.getpid()))
            temporary.write_text(json.dumps(payload)); os.replace(temporary, path)
            print('[hybrid RSI]', index+1, '/', len(work), kind,
                  'profiles', len(keys), 'nonempty', sum(bool(v) for v in result), flush=True)
    finally:
        env._root_states.copy_(original_roots); env._dof_state.copy_(original_dofs)
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env._root_states))
        env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env._dof_state))
        env.gym.set_dof_position_target_tensor(env.sim,
            gymtorch.unwrap_tensor(env._dof_pos.contiguous().clone()))
        env._refresh_sim_tensors()
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng, env.device)
    return records, fingerprint

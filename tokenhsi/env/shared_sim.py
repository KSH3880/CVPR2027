"""One Isaac Gym simulator with independently indexed reference/native partitions."""
import torch
import random
import numpy as np
from contextlib import contextmanager
from isaacgym import gymapi, gymtorch

TENSORS = {'actor_root_state': 'actor', 'dof_state': 'dof', 'dof_force': 'dof',
           'rigid_body_state': 'rigid_body', 'net_contact_force': 'rigid_body', 'force_sensor': 'force_sensor'}


class SharedSim:
    def __init__(self, native_cfg, native_params, args, reference_num_envs):
        self.gym = gymapi.acquire_gym()
        self.sim = None
        self.native_cfg, self.native_params, self.args = native_cfg, native_params, args
        self.reference_num_envs = reference_num_envs
        self.parts = [Partition(self, 0), Partition(self, 1)]
        self.buffers, self.controls = {}, {}
        self.pending_states, self.applied_states = {}, {}
        self.pending_controls = set()
        self.refreshed = {}
        self.dof_actors = []
        self.native_rng = None
        self.native = None
        self.prepared = False
        self.steps = 0
        self.simulation_calls = 0

    def counts(self):
        return {kind: getattr(self.gym, 'get_sim_'+kind+'_count')(self.sim)
                for kind in ('actor','dof','rigid_body','force_sensor')}

    def prepare(self, partition):
        if partition.index == 0:
            self.parts[0].end = self.counts()
            self.parts[1].start = dict(self.parts[0].end)
            from env.tasks.multi_task.humanoid_traj_sit_carry_climb import HumanoidTrajSitCarryClimb
            cfg = self.native_cfg
            cfg['_shared_gym'] = self.parts[1]
            a = self.args
            # Native asset creation nests here, before either task acquires tensors.
            with self.native_context():
                self.native = HumanoidTrajSitCarryClimb(cfg, self.native_params, a.physics_engine,
                                                        a.device, a.device_id, a.headless)
        else:
            self.parts[1].end = self.counts()
            self.gym.prepare_sim(self.sim)
            self.prepared = True
            print('SHARED_SIM prepared:', [p.end for p in self.parts], flush=True)

    @staticmethod
    def _rng_state():
        return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)

    @staticmethod
    def _restore_rng(state):
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])

    @contextmanager
    def native_context(self):
        # Match the old worker's independent RNG without changing learner/PUSH RNG.
        parent = self._rng_state()
        try:
            if self.native_rng is None:
                seed = int(self.native_cfg.get('seed', 42))
                seed = seed if seed >= 0 else 42
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
            else:
                self._restore_rng(self.native_rng)
            yield
        finally:
            self.native_rng = self._rng_state()
            self._restore_rng(parent)

    def reset_native(self, ids):
        with self.native_context():
            return self.native.reset(ids)

    def tensor(self, name):
        if name not in self.buffers:
            self.buffers[name] = gymtorch.wrap_tensor(getattr(self.gym, 'acquire_'+name+'_tensor')(self.sim))
        return self.buffers[name]

    def refresh(self, name):
        # Refreshing the same global tensor from the other partition must not
        # overwrite resets staged since the last physics step.
        if self.refreshed.get(name) != self.simulation_calls:
            getattr(self.gym, name)(self.sim)
            self.refreshed[name] = self.simulation_calls

    def queue_state(self, partition, key, local, ids):
        if key not in self.pending_states:
            self.pending_states[key] = (self.tensor(key).clone(), [])
        state, indices = self.pending_states[key]
        kind = TENSORS[key]
        state[partition.start[kind]:partition.end[kind]].copy_(gymtorch.wrap_tensor(local))
        indices.append(ids.clone())
        return True

    def flush(self):
        # Isaac Gym's GPU pipeline permits one call per setter per simulate.
        # Passing only the last partition's indices loses the other resets.
        for key, (state, pieces) in self.pending_states.items():
            ids = torch.unique(torch.cat(pieces)).to(dtype=torch.int32)
            self.applied_states[key] = (state, ids)
            getattr(self.gym, 'set_'+key+'_tensor_indexed')(
                self.sim, gymtorch.unwrap_tensor(state), gymtorch.unwrap_tensor(ids), ids.numel())
        self.pending_states.clear()
        for name in sorted(self.pending_controls):
            getattr(self.gym, name)(self.sim, gymtorch.unwrap_tensor(self.controls[name]))
        self.pending_controls.clear()

    def step(self, reference, ref_actions, native_actions):
        native = self.native
        if reference.dr_randomizations or native.dr_randomizations:
            raise ValueError('Shared simulator does not support domain randomization yet')
        reference.pre_physics_step(ref_actions)
        with self.native_context():
            native.pre_physics_step(native_actions)
        # Reference loop accumulates PUSH contact at each physics step. Both pools advance here.
        reference._physics_step()
        if reference.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        reference.post_physics_step()
        with self.native_context():
            native.post_physics_step()
        self.steps += 1

    def close(self):
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None


class Partition:
    def __init__(self, owner, index):
        self.owner, self.index = owner, index
        self.start = {k:0 for k in ('actor','dof','rigid_body','force_sensor')}
        self.end = None
        self.keep = {}

    def __deepcopy__(self, memo):
        return self

    def __getattr__(self, name):
        if name.startswith('refresh_') and name.endswith('_tensor'):
            return lambda sim: self.owner.refresh(name)
        if name.startswith('acquire_') and name.endswith('_tensor'):
            key = name[len('acquire_'):-len('_tensor')]
            if key in TENSORS:
                return lambda sim: self.acquire(key)
        return getattr(self.owner.gym, name)

    def create_sim(self, *args):
        if self.owner.sim is None:
            self.owner.sim = self.owner.gym.create_sim(*args)
        return self.owner.sim

    def add_ground(self, sim, plane):
        if self.index == 0:
            self.owner.gym.add_ground(sim, plane)

    def prepare_sim(self, sim):
        self.owner.prepare(self)

    def simulate(self, sim):
        self.owner.flush()
        self.owner.simulation_calls += 1
        return self.owner.gym.simulate(sim)

    def create_actor(self, env, asset, pose, name, group, *args):
        # Collision groups must be unique across both environment partitions.
        offset = self.owner.reference_num_envs if self.index else 0
        handle = self.owner.gym.create_actor(env, asset, pose, name, group+offset if group >= 0 else group, *args)
        if self.owner.gym.get_asset_dof_count(asset):
            self.owner.dof_actors.append(self.owner.gym.get_actor_index(env, handle, gymapi.DOMAIN_SIM))
        return handle

    def acquire(self, key):
        kind = TENSORS[key]
        view = self.owner.tensor(key)[self.start[kind]:self.end[kind]]
        self.keep[key] = view
        return gymtorch.unwrap_tensor(view)

    def _set_indexed(self, name, key, local, ids, count):
        global_ids = gymtorch.wrap_tensor(ids)[:count]+self.start['actor']
        return self.owner.queue_state(self, key, local, global_ids)

    def set_actor_root_state_tensor_indexed(self, sim, local, ids, count):
        return self._set_indexed('set_actor_root_state_tensor_indexed','actor_root_state',local,ids,count)

    def set_dof_state_tensor_indexed(self, sim, local, ids, count):
        return self._set_indexed('set_dof_state_tensor_indexed','dof_state',local,ids,count)

    def _set_state(self, name, key, local):
        # RSI cache screening also uses non-indexed setters during initialization.
        state = gymtorch.wrap_tensor(local)
        ids = range(self.start['actor'], self.end['actor']) if key == 'actor_root_state' else (
            x for x in self.owner.dof_actors if self.start['actor'] <= x < self.end['actor'])
        ids = torch.tensor(list(ids), device=state.device, dtype=torch.int32)
        return self.owner.queue_state(self, key, local, ids)

    def set_actor_root_state_tensor(self, sim, local):
        return self._set_state('set_actor_root_state_tensor', 'actor_root_state', local)

    def set_dof_state_tensor(self, sim, local):
        return self._set_state('set_dof_state_tensor', 'dof_state', local)

    def _set_control(self, name, local):
        source = gymtorch.wrap_tensor(local).flatten()
        if name not in self.owner.controls:
            self.owner.controls[name] = source.new_zeros(self.owner.parts[1].end['dof'])
        full = self.owner.controls[name]
        full[self.start['dof']:self.end['dof']].copy_(source)
        self.owner.pending_controls.add(name)
        return True

    def set_dof_position_target_tensor(self, sim, local):
        return self._set_control('set_dof_position_target_tensor',local)

    def set_dof_actuation_force_tensor(self, sim, local):
        return self._set_control('set_dof_actuation_force_tensor',local)

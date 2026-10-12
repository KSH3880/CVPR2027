"""Early current-policy measurements and completed XYZ/task episode counters."""
import csv
import os
from pathlib import Path

import torch

from utils.edge_scenario_spec import classify_templates


TASKS = ('holding', 'sit', 'climb', 'holding_at', 'holding_ontop')
FIELDS = ('starts', 'steps', 'completed', 'final_success', 'ever_success',
          'initial_success', 'reference_starts', 'loco_starts')


def grid_index(sizes):
    cm = (sizes*100).round().long()
    index = (cm-20)//5
    if ((cm-20) % 5 != 0).any() or (index < 0).any() or (index > 9).any():
        raise ValueError('Hybrid diagnostics require XYZ20..65cm in 5cm steps')
    return (index[..., 0]*10 + index[..., 1])*10 + index[..., 2]


class HybridRSIRuntime:
    def __init__(self, env, bank):
        self.env, self.bank = env, bank
        shape = (env.num_envs, env.num_agents)
        self.pending = torch.full(shape, -1, device=env.device, dtype=torch.long)
        self.age = torch.full_like(self.pending, 2)
        self.valid = torch.ones(shape, device=env.device, dtype=torch.bool)
        self.source = torch.zeros_like(self.pending)
        self.root = torch.zeros(*shape, 13, device=env.device)
        self.box = torch.zeros_like(self.root)
        self.prev_velocity = torch.zeros(*shape, 3, device=env.device)
        self.hand_distance = torch.zeros(shape, device=env.device)
        self.episode_bin = torch.zeros_like(self.pending)
        self.live = torch.zeros_like(self.valid)
        self.ever = torch.zeros_like(self.valid)
        self.counts = torch.zeros(5000, len(FIELDS), device=env.device, dtype=torch.long)
        self.last_dump_epoch = 0
        self.session = str(os.getpid())
        self.hands = torch.tensor([env.gym.find_actor_rigid_body_handle(env.envs[0],
            env.humanoid_handles_all[0][0], name) for name in ('right_hand', 'left_hand')],
            device=env.device)

    def begin_slots(self, rows, owners):
        # Called on every reset attempt; an abandoned attempt is never observed.
        self.pending[rows, owners] = -1

    def mark_slots(self, rows, owners, tokens):
        self.pending[rows, owners] = tokens

    def commit_reset(self, env_ids):
        env = self.env
        rows = env_ids.repeat_interleave(env.num_agents)
        owners = torch.arange(env.num_agents, device=env.device).repeat(len(env_ids))
        physical = env._scenario_physical_object(rows, owners)
        self.source[rows, owners] = physical
        templates = classify_templates(env.relation_runtime.graph, rows, owners, True)
        bins = templates*1000 + grid_index(env._box_size[rows, physical])
        self.episode_bin[rows, owners] = bins
        self.live[env_ids] = True
        self.ever[env_ids] = env.relation_runtime.done[env_ids]
        self.root[env_ids] = env._humanoid_root_states[env_ids]
        self.box[rows, owners] = env._box_states[rows, physical]
        self.prev_velocity[rows, owners] = self.box[rows, owners, 7:10]
        hand = env._kinematic_humanoid_rigid_body_states[rows, owners][:, self.hands, :3].mean(-2)
        self.hand_distance[rows, owners] = (hand-self.box[rows, owners, :3]).norm(dim=-1)
        self.valid[env_ids] = True
        self.age[env_ids] = 0
        self.counts[:, 0].index_add_(0, bins, torch.ones_like(bins))
        self.counts[:, 5].index_add_(0, bins, env.relation_runtime.done[rows, owners].long())
        advanced_slots = torch.zeros_like(self.pending)
        for name, slots in env._reset_ref_slots.items():
            if name != 'loco':
                r, a = slots
                advanced_slots[r, a] = 1
        advanced = advanced_slots[rows, owners]
        self.counts[:, 6].index_add_(0, bins, advanced)
        self.counts[:, 7].index_add_(0, bins, 1-advanced)

    def step(self):
        env, spec = self.env, self.bank.spec['screen']
        rows, owners = ((self.age < 2) & (self.pending >= 0) & self.live).nonzero(as_tuple=True)
        if len(rows):
            box = env._box_states[rows, self.source[rows, owners]]
            root = env._humanoid_root_states[rows, owners]
            initial_root, initial_box = self.root[rows, owners], self.box[rows, owners]
            from env.tasks.multi_agent.hybrid_rsi_screen import screen_states
            good = screen_states(root, box[:, None], initial_root, initial_box[:, None], spec)
            good &= (box[:, 7:10]-self.prev_velocity[rows, owners]).norm(dim=-1) <= spec['box_delta_v']
            hands = env._rigid_body_pos.reshape(env.num_envs, env.num_agents, env.num_bodies, 3)[rows, owners][:, self.hands].mean(-2)
            distance = (hands-box[:, :3]).norm(dim=-1)
            tokens = self.pending[rows, owners]
            carry = self.bank.is_carry[tokens]
            good &= ~carry | (distance-self.hand_distance[rows, owners] <= spec['carry_hand_increase'])
            good &= torch.isfinite(env._dof_state.view(env.num_envs, env.num_agents, -1)[rows, owners]).all(-1)
            self.valid[rows, owners] &= good
            self.prev_velocity[rows, owners] = box[:, 7:10]
            self.age[rows, owners] += 1
            finished = (self.age[rows, owners] == 2) | env.reset_buf[rows].bool()
            if finished.any():
                self.bank.observe(tokens[finished], self.valid[rows[finished], owners[finished]])
                self.age[rows[finished], owners[finished]] = 2
        bins = self.episode_bin[self.live]
        self.counts[:, 1].index_add_(0, bins, torch.ones_like(bins))
        self.ever |= env.relation_runtime.done & self.live
        ending = env.reset_buf.bool()[:, None].expand_as(self.live) & self.live
        if ending.any():
            bins = self.episode_bin[ending]
            self.counts[:, 2].index_add_(0, bins, torch.ones_like(bins))
            self.counts[:, 3].index_add_(0, bins, env.relation_runtime.done[ending].long())
            self.counts[:, 4].index_add_(0, bins, self.ever[ending].long())
            self.live[ending] = False

    def dump(self, epoch):
        if epoch < self.last_dump_epoch+500:
            return
        self.last_dump_epoch = epoch
        directory = Path(getattr(self.env, '_relation_output_directory',
                                 self.env.cfg['args'].output_path)) / 'diagnostics'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ('hybrid_xyz_episodes_' + self.session + '.csv')
        exists = path.exists()
        with path.open('a', newline='') as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(['epoch', 'task', 'x_cm', 'y_cm', 'z_cm', *FIELDS])
            for index, values in enumerate(self.counts.cpu().tolist()):
                if not any(values):
                    continue
                task, size = divmod(index, 1000)
                x, remainder = divmod(size, 100); y, z = divmod(remainder, 10)
                writer.writerow([epoch, TASKS[task], 20+5*x, 20+5*y, 20+5*z, *values])
        candidate_path = directory / ('hybrid_candidates_' + self.session + '.csv')
        exists = candidate_path.exists()
        with candidate_path.open('a', newline='') as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(['epoch', 'generation', 'skill', 'source_xyz_cm',
                    'support_xyz_cm', 'motion_id', 'phase', 'band', 'observations', 'failures'])
            for desc, n, bad in zip(self.bank.descriptors, self.bank.counts.cpu().tolist(),
                                    self.bank.failures.cpu().tolist()):
                if not n:
                    continue
                key, mid, phase, band = desc
                writer.writerow([epoch, self.bank.generation, key[0], key[1],
                    key[2] if len(key)>2 else '', mid, phase, band, n, bad])

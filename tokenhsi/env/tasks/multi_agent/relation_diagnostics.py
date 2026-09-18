"""Reward-independent placement metrics and bounded per-step CSV traces."""
import csv
import os
import uuid

import torch


# Order is encoded in tags so TensorBoard's tag sorting keeps the useful cards
# together. Each metric is emitted once; raw diagnostic/CSV keys stay unchanged.
RELATION_TB_GROUPS = (
    ('00_main', (
        'placement/episode_final_rate', 'placement/episode_ever_rate',
        'placement/post_first_retention', 'current_success_state', 'holding/satisfied',
    )),
    ('01_placement', (
        'goal_xy_error', 'goal_z_error', 'at/satisfied',
        'placement/first_seconds', 'placement/longest_hold_seconds',
        'placement/post_first_seconds',
    )),
    ('02_reward', (
        'saturation_active', 'current_success_reward', 'task_relation_total',
        'holding/state_reward', 'at/state_reward',
        'holding/progress_reward', 'at/progress_reward',
        'holding/raw_state_reward', 'at/raw_state_reward',
        'holding/raw_progress_reward', 'at/raw_progress_reward',
        'success_bonus', 'first_success_bonus',
    )),
    ('03_samples', (
        'placement/eligible_completed_count', 'placement/reached_count',
        'placement/initially_placed_count', 'placement/initially_placed_fraction',
        'placement/completed_count', 'placement/initially_placed_final_rate',
    )),
    ('04_state', (
        'holding/phi', 'at/phi', 'holding/gate', 'at/gate',
        'holding/achieved', 'at/achieved', 'current_target_valid',
        'scene_current_all_valid', 'done', 'scene_all_done', 'first_success',
        'holding/threshold_up', 'holding/threshold_down',
        'at/threshold_up', 'at/threshold_down', 'active',
    )),
    ('05_motion', (
        'root_box_distance_xy', 'hand_midpoint_distance',
        'right_hand_center_distance', 'left_hand_center_distance',
        'box_speed', 'box_bottom_height_proxy', 'target_xy_crossing',
        'progress_holding', 'progress_at',
        'holding/progress_bar', 'at/progress_bar', 'holding/approach', 'at/approach',
    )),
)
_RELATION_TB_TAGS = {
    key: 'relation/{}/{:02d}_{}'.format(group, index, key.replace('/', '_'))
    for group, keys in RELATION_TB_GROUPS
    for index, key in enumerate(keys, 1)
}


def relation_tensorboard_tag(key):
    """Presentation only; preserve unlisted metrics in a trailing debug group."""
    return _RELATION_TB_TAGS.get(key, 'relation/90_debug/' + key)


def placement_valid(xy_error, z_error):
    """One fixed geometric criterion shared by every reward experiment."""
    return (xy_error <= .1) & (z_error.abs() <= .001)


class PlacementEpisodeMetrics:
    """Aggregate completed agent-episodes, not rollout-time averages.

    Initially placed episodes are excluded from the main metrics. Conditional
    metrics use only reached episodes; the first placed step is included in the
    post-first window. An explicit scene termination finalizes each agent once.
    """

    def __init__(self, num_envs, num_agents, dt, device):
        self.dt = dt
        shape = (num_envs, num_agents)
        self.active = torch.zeros(shape, dtype=torch.bool, device=device)
        self.initial = torch.zeros_like(self.active)
        self.current = torch.zeros_like(self.active)
        self.steps = torch.zeros(shape, dtype=torch.long, device=device)
        self.first = torch.full_like(self.steps, -1)
        self.post_steps = torch.zeros_like(self.steps)
        self.post_placed = torch.zeros_like(self.steps)
        self.streak = torch.zeros_like(self.steps)
        self.longest = torch.zeros_like(self.steps)
        # total, initial, initial-final, eligible, reached, final,
        # retention sum, first-time sum, longest-hold sum, post-first seconds sum
        self.sums = torch.zeros(10, device=device)

    def reset(self, env_ids, placed):
        # A reset without a preceding terminal event discards an unfinished
        # episode (including initialization/resampling); it is not a completion.
        self.active[env_ids] = True
        self.initial[env_ids] = placed
        self.current[env_ids] = placed
        self.steps[env_ids] = 0
        self.first[env_ids] = torch.where(placed, 0, -1)
        for value in (self.post_steps, self.post_placed, self.streak, self.longest):
            value[env_ids] = 0

    def step(self, placed):
        self.steps += self.active.long()
        self.current.copy_(torch.where(self.active, placed, self.current))
        first_now = self.active & placed & (self.first < 0)
        self.first.copy_(torch.where(first_now, self.steps, self.first))
        post = self.active & (self.first >= 0)
        self.post_steps += post.long()
        self.post_placed += (post & placed).long()
        streak = torch.where(placed, self.streak + 1, torch.zeros_like(self.streak))
        self.streak.copy_(torch.where(self.active, streak, self.streak))
        self.longest.copy_(torch.maximum(self.longest, self.streak))

    def finish(self, scene_done):
        completed = self.active & scene_done.bool()[:, None] & (self.steps > 0)
        initial = completed & self.initial
        eligible = completed & ~self.initial
        reached = eligible & (self.first >= 0)
        retention = self.post_placed.float() / self.post_steps.clamp_min(1)
        self.sums += torch.stack([
            completed.sum(), initial.sum(), (initial & self.current).sum(),
            eligible.sum(), reached.sum(), (eligible & self.current).sum(),
            (retention * reached).sum(), (self.first * reached).sum() * self.dt,
            (self.longest * reached).sum() * self.dt,
            (self.post_steps * reached).sum() * self.dt,
        ])
        self.active &= ~completed

    def consume(self):
        sums = self.sums.clone()
        self.sums.zero_()
        # One small host transfer per rollout; no per-step GPU synchronization.
        total, initial, _, eligible, reached = sums[:5].tolist()
        result = dict(completed_count=sums[0], initially_placed_count=sums[1],
                      eligible_completed_count=sums[3], reached_count=sums[4])
        if total:
            result['initially_placed_fraction'] = sums[1] / sums[0]
        if initial:
            result['initially_placed_final_rate'] = sums[2] / sums[1]
        if eligible:
            result['episode_ever_rate'] = sums[4] / sums[3]
            result['episode_final_rate'] = sums[5] / sums[3]
        if reached:
            result['post_first_retention'] = sums[6] / sums[4]
            result['first_seconds'] = sums[7] / sums[4]
            result['longest_hold_seconds'] = sums[8] / sums[4]
            result['post_first_seconds'] = sums[9] / sums[4]
        return {'placement/' + key: value for key, value in result.items()}


class RelationTimeline:
    """Trace env 0's first and then every 100th reset episode by default.

    All agents in selected scenes are recorded every step, for at most 600
    steps/episode. Rows are buffered; each process writes a fresh file so resumed
    runs and schema changes never append incompatible rows to an existing CSV.
    """

    def __init__(self, directory, num_envs, config):
        enabled = config.get('enabled', True) and config.get('timeline_enabled', True)
        self.count = min(num_envs, config.get('timeline_sample_envs', 1)) if enabled else 0
        self.every = config.get('timeline_every_episodes', 100)
        self.max_steps = config.get('timeline_max_steps', 600)
        self.directory = directory
        self.path = None
        self.episode_counts = [0] * self.count
        self.has_stepped = [False] * self.count
        self.selected = set()
        self.recorded = [0] * self.count
        self.columns = None
        self.rows = []

    def reset(self, env_ids):
        if not self.count:
            return
        # Only sampled scene IDs cross to CPU, and only at reset.
        for env in env_ids[env_ids < self.count].detach().cpu().tolist():
            if env in self.selected:
                self.flush()
            # Environment construction / learner initialization may reset twice
            # before any simulation step. Do not skip the first sampled episode.
            if self.episode_counts[env] == 0 or self.has_stepped[env]:
                self.episode_counts[env] += 1
            self.has_stepped[env] = False
            self.recorded[env] = 0
            if (self.episode_counts[env] - 1) % self.every == 0:
                self.selected.add(env)
            else:
                self.selected.discard(env)

    def record(self, global_step, episode_ids, episode_steps, dt, diag):
        if not self.selected:
            return
        envs = sorted(self.selected)
        names = list(diag)
        columns = ['global_step', 'env', 'agent', 'episode', 'episode_step', 'seconds'] + names
        if self.columns is None:
            self.columns = columns
        elif self.columns != columns:
            raise ValueError('Relation timeline columns changed within one run')
        num_agents = next(iter(diag.values())).shape[1]
        steps = episode_steps[envs, None].expand(-1, num_agents)
        values = torch.stack([
            episode_ids[envs, None].expand(-1, num_agents), steps, steps * dt,
        ] + [diag[name][envs] for name in names], -1).detach().cpu().tolist()
        for env, agents in zip(envs, values):
            self.has_stepped[env] = True
            for agent, values in enumerate(agents):
                self.rows.append([global_step, env, agent] + values)
            self.recorded[env] += 1
            if self.recorded[env] >= self.max_steps:
                self.selected.discard(env)
        if len(self.rows) >= 256:
            self.flush()

    def finish(self, scene_done):
        for env in range(self.count):
            self.has_stepped[env] = True
        if self.selected:
            envs = sorted(self.selected)
            ended = scene_done[envs].bool().detach().cpu().tolist()
            for env, done in zip(envs, ended):
                if done:
                    self.selected.discard(env)
                    self.flush()

    def flush(self):
        if not self.rows:
            return
        new = self.path is None
        if new:
            os.makedirs(self.directory, exist_ok=True)
            self.path = os.path.join(self.directory, 'relation_timeline_' + uuid.uuid4().hex[:12] + '.csv')
        with open(self.path, 'a', newline='') as handle:
            writer = csv.writer(handle)
            if new:
                writer.writerow(self.columns)
            writer.writerows(self.rows)
        self.rows.clear()

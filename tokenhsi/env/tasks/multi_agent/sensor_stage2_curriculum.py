import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from utils.bridge_curriculum import span_probabilities, walk_layout
from utils.sensor_stage2_spec import BRIDGE


class SensorStage2Curriculum:
    def _init_bridge_curriculum(self):
        n, m, device = self.num_envs, self.num_agents, self.device
        self._bridge_curriculum_update = 0
        self._bridge_walk = torch.zeros(n, dtype=torch.bool, device=device)
        self._bridge_span = torch.ones(n, dtype=torch.long, device=device)
        self._crosser_mask = torch.zeros(n, m, dtype=torch.bool, device=device)
        self._cross_phase = torch.full((n, m), -1, dtype=torch.long, device=device)
        self._cross_progress = torch.zeros(n, m, device=device)
        self._cross_entered = torch.zeros(n, m, dtype=torch.bool, device=device)
        self._cross_completed = torch.zeros_like(self._cross_entered)
        self._cross_stats = torch.zeros(2, 3, 3, 8, device=device)
        self._cross_scene_stats = torch.zeros(2, 2, device=device)
        self._cross_exposure = torch.zeros(4, device=device)

    def _sample_bridge_curriculum(self, ids, counts):
        spec = self._bridge_curriculum
        walk = (torch.rand(len(ids), device=self.device) < spec['walk_probability']) & (self._scene_ids[ids] == BRIDGE)
        if self._is_eval:
            walk[:] = self._curriculum_eval_mode != 'cooperative'
            walk &= self._scene_ids[ids] == BRIDGE
        spans = counts.clone()
        update = spec['warmup_updates']+spec['anneal_updates'] if self._is_eval else self._bridge_curriculum_update
        probability = torch.tensor(span_probabilities(spec, update), device=self.device)
        if walk.any():
            spans[walk] = torch.multinomial(probability, int(walk.sum()), replacement=True)+1
        counts = torch.where(walk, 1, counts)
        self._bridge_walk[ids], self._bridge_span[ids] = walk, spans
        self._cross_entered[ids], self._cross_completed[ids] = False, False
        return counts, spans

    def _curriculum_roles(self, ids, groups, pre, tasks):
        selected = self._bridge_walk[ids]
        rows = selected.nonzero().flatten()
        env = ids[rows]
        if len(rows):
            people = self._roles[env, 1:]
            groups[rows] = torch.arange(self.num_agents, device=self.device)
            pre[rows] = -1
            tasks[rows[:, None], people] = torch.randint(1, 5, people.shape, device=self.device)
        return groups, pre, tasks

    def _curriculum_surfaces(self, ids):
        bridge = self._scene_ids[ids] == BRIDGE
        self._crosser_mask[ids] = False
        self._cross_phase[ids] = -1
        env = ids[bridge]
        self._crosser_mask[env, self._roles[env, 1]] = True
        self._cross_phase[env, self._roles[env, 1]] = 0
        env = ids[self._bridge_walk[ids]]
        if not len(env):
            return
        people = self._roles[env, 1:]
        start = torch.randint(1, 3, (len(env), 1), device=self.device)
        self._start_surface[env[:, None], people] = start
        self._surface[env[:, None], people] = 3-start
        self._crosser_mask[env[:, None], people] = True
        self._cross_phase[env[:, None], people] = 0
        phase = torch.randint(1, 3, (len(env),), device=self.device)
        if self._is_eval and self._curriculum_eval_mode == 'walk_entry':
            phase.zero_()
        self._cross_phase[env, people[:, 2]] = phase
        lo = torch.where(phase == 1, .25, .65)
        self._cross_progress[env, people[:, 2]] = lo+.2*torch.rand(len(env), device=self.device)
        self._bridge_start_mode[env] = 0
        self._bridge_near[env] = False
        self._bridge_aligned[env] = True
        self._bridge_carry_from_start[env] = False

    def _place_curriculum_walk(self, env, rows, occupied):
        m, n = self.num_agents, len(env)
        people = self._roles[env, 1:]
        start = self._start_surface[env, people[:, 0]]-1
        dest = 1-start
        centers, halves = self._surface_center[env, BRIDGE], self._surface_half[env, BRIDGE]
        index = torch.arange(n, device=self.device)
        source = self._agent_box_assignment[env[:, None], people]
        support = 3*m+people
        sizes = self._box_size[env]
        source_radius = sizes[index[:, None], source, :2].norm(dim=-1)/2
        second_radius = sizes[index[:, None], support, :2].norm(dim=-1)/2
        tasks = self._templates[env[:, None], people]
        second_radius = torch.where(tasks == 4, second_radius, source_radius)
        bridge = self._box_size[env, self._bridge_slot[env]]
        payload, second, humans, failed = walk_layout(centers[index, start], halves[index, start],
            centers[index, dest], halves[index, dest], self._bridge_landing[env], bridge,
            self._bridge_gap[env], tasks, torch.stack((source_radius, second_radius), -1),
            self._cross_phase[env[:, None], people], self._cross_progress[env[:, None], people], self._bridge_curriculum)
        self._place_failed[env[failed]] = True
        origin = self._env_origins[env]
        yaw, shift = self._structure_yaw[env, BRIDGE], self._structure_shift[env, BRIDGE]
        world = lambda xy: scene.to_world(xy, yaw, shift, origin)
        height = self._surface_height[env, BRIDGE]
        for column in range(3):
            person, primary, extra = people[:, column], source[:, column], support[:, column]
            task = tasks[:, column]
            box_xy, second_xy, human_xy = world(payload[:, column]), world(second[:, column]), world(humans[:, column])
            self._box_states[env, primary, :2] = box_xy
            self._box_states[env, primary, 2] = origin[:, 2]+height+sizes[index, primary, 2]/2
            at = task == 3
            self._tar_pos[env[at], person[at], :2] = second_xy[at]
            self._tar_pos[env[at], person[at], 2] = (origin[:, 2]+height+sizes[index, primary, 2]/2)[at]
            top = task == 4
            self._box_states[env[top], extra[top], :2] = second_xy[top]
            self._box_states[env[top], extra[top], 2] = (origin[:, 2]+height+sizes[index, extra, 2]/2)[top]
            lift = torch.where(self._cross_phase[env, person] > 0, bridge[:, 2], height)
            direction = box_xy-human_xy
            heading = torch.atan2(direction[:, 1], direction[:, 0])
            moved = self._place_person(env, person, human_xy, heading, lift)
            self._move_neutral_goal(env, person, moved, lift)

    def _record_crossing_curriculum(self, fallen, ending, roots):
        bridge = self._scene_ids == BRIDGE
        eligible = self._crosser_mask & bridge[:, None]
        rows = torch.arange(self.num_envs, device=self.device)
        boxes = self._box_states[rows, self._bridge_slot]
        size = self._box_size[rows, self._bridge_slot]
        yaw = torch.atan2(2*(boxes[:, 6]*boxes[:, 5]+boxes[:, 3]*boxes[:, 4]), 1-2*(boxes[:, 4]**2+boxes[:, 5]**2))
        xy = scene.rotate_xy(roots[..., :2]-boxes[:, None, :2], -yaw[:, None])
        self._cross_entered |= eligible & (xy.abs() <= size[:, None, :2]/2).all(-1) & (roots[..., 2] > boxes[:, None, 2]+size[:, None, 2]/2+.4) & ~fallen
        dest = (self._surface-1).clamp_min(0)
        center = self._surface_center[rows[:, None], BRIDGE, dest]
        half = self._surface_half[rows[:, None], BRIDGE, dest]
        all_local = scene.rotate_xy(roots[..., :2]-self._env_origins[:, None, :2]-self._structure_shift[:, BRIDGE, None], -self._structure_yaw[:, BRIDGE, None])
        self._cross_completed |= eligible & ((all_local-center).abs() <= half).all(-1) & (roots[..., 2] > self._env_origins[:, None, 2]+self._surface_height[:, BRIDGE, None]+.6) & ~fallen
        self._cross_exposure += torch.stack((eligible.sum(), eligible.sum().new_tensor(self.num_envs*self.num_agents),
            (self._unreachable & (self._scene_ids == 0)[:, None]).sum(),
            ((self._start_surface > 0) & (self._scene_ids == 2)[:, None]).sum())).float()
        ended = eligible & ending[:, None]
        other_fall = fallen.any(-1)[:, None] & ~fallen
        scenes = (ending & bridge).nonzero().flatten()
        scene_values = torch.stack((torch.ones_like(scenes), (self._cross_completed | ~eligible).all(-1)[scenes]), -1).float()
        self._cross_scene_stats.index_add_(0, self._bridge_walk[scenes].long(), scene_values)
        index = (self._bridge_walk.long()*3+self._bridge_span-1)[:, None]*3+self._cross_phase
        values = torch.stack((torch.ones_like(eligible), self._cross_entered, self._cross_completed,
            fallen, other_fall, self._bridge_arrived[:, None].expand_as(eligible),
            (self.progress_buf >= self.max_episode_length)[:, None].expand_as(eligible),
            other_fall & ~self._cross_completed), -1).float()
        self._cross_stats.view(-1, 8).index_add_(0, index[ended], values[ended])

    def _consume_crossing_curriculum(self):
        result = {}
        for branch, name in enumerate(('cooperative', 'walk')):
            for kind, sums in [('all', self._cross_stats[branch].sum((0, 1)))]+[
                    ('span'+str(i+1), self._cross_stats[branch, i].sum(0)) for i in range(3)]+[
                    (name, self._cross_stats[branch, :, i].sum(0)) for i, name in enumerate(('entry', 'middle', 'exit'))]:
                prefix = 'crossing/'+name+'/'+kind+'/'
                result[prefix+'episodes'] = sums[0].clone()
                for i, metric in enumerate(('entered', 'crossed', 'falls', 'other_agent_fall', 'arrived', 'timeout', 'interrupted_before_crossing'), 1):
                    result[prefix+metric] = sums[i]/sums[0].clamp_min(1)
            scene = self._cross_scene_stats[branch]
            result['crossing/'+name+'/scene_episodes'] = scene[0].clone()
            result['crossing/'+name+'/all_crossed'] = scene[1]/scene[0].clamp_min(1)
        for i, name in enumerate(('crosser_agent_steps', 'all_agent_steps', 'cliff_edge_required_agent_steps', 'room_inside_agent_steps')):
            result['crossing/exposure/'+name] = self._cross_exposure[i].clone()
        result['crossing/exposure/crosser_fraction'] = self._cross_exposure[0]/self._cross_exposure[1].clamp_min(1)
        result['crossing/update'] = torch.tensor(float(self._bridge_curriculum_update), device=self.device)
        for i, p in enumerate(span_probabilities(self._bridge_curriculum, self._bridge_curriculum_update)):
            result['crossing/span_probability'+str(i+1)] = torch.tensor(p, device=self.device)
        self._cross_stats.zero_()
        self._cross_scene_stats.zero_()
        self._cross_exposure.zero_()
        return result

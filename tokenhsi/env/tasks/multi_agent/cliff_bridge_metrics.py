import torch

from env.tasks.multi_agent import square_sensor_scene as scene
from env.tasks.multi_agent.humanoid_ma_stage2 import HumanoidMAStage2
from utils.bridge_curriculum import span_probabilities
from utils.cliff_bridge_spec import BRANCHES
from utils.sensor_stage2_spec import paired_people
from utils import torch_utils


METRICS = ('episodes', 'entered', 'crossed', 'falls', 'other_agent_fall', 'arrived',
           'timeout', 'interrupted_before_crossing')


class CliffBridgeMetrics:
    def _record_scene_outcomes(self, fallen, done):
        ending = self.reset_buf.bool()
        roots = self._rigid_body_pos[:, :, 0]
        for index in (0, 1):
            chosen = self._scene_ids == index
            self._scene_outcomes[index] += torch.stack(((chosen & ending).sum(), chosen.sum(),
                fallen[chosen].sum(), (chosen & done).sum())).float()
        self._record_variant_crossing(fallen, ending, roots)
        cliff = self._scene_ids == 0
        local = scene.rotate_xy(roots[..., :2]-self._env_origins[:, None, :2]-self._structure_shift[:, 0, None],
            -self._structure_yaw[:, 0, None])
        clearance = (self._surface_half[:, 0, 0, None]-local.abs()).amin(-1)
        outside = (clearance < 0) & cliff[:, None]
        self._cliff_exited |= outside
        self._cliff_dropped |= outside & (roots[..., 2] < self._env_origins[:, None, 2]+self._surface_height[:, 0, None]+.5)
        near = cliff[:, None] & (clearance >= 0) & (clearance < 1.)
        speed = self._rigid_body_vel[:, :, 0, :2].norm(dim=-1)
        ended = cliff & ending
        self._cliff_stats += torch.stack((ended.sum(), 4*ended.sum(), self._cliff_exited[ended].sum(),
            fallen[ended].sum(), self._cliff_dropped[ended].sum(), (cliff & done).sum(), near.sum(),
            (near & (speed < .15)).sum(), 4*cliff.sum())).float()

    def _record_variant_crossing(self, fallen, ending, roots):
        rows = torch.arange(self.num_envs, device=self.device)
        eligible = self._crosser_mask & (self._scene_ids == 1)[:, None]
        box = self._box_states[rows, self._bridge_slot]
        size = self._box_size[rows, self._bridge_slot]
        yaw = torch_utils.calc_heading(box[:, 3:7])
        local = scene.rotate_xy(roots[..., :2]-box[:, None, :2], -yaw[:, None])
        self._cross_entered |= eligible & (local.abs() <= size[:, None, :2]/2).all(-1) & (
            roots[..., 2] > box[:, None, 2]+size[:, None, 2]/2+.4) & ~fallen
        destination = (self._surface-1).clamp_min(0)
        center = self._surface_center[rows[:, None], 1, destination]
        half = self._surface_half[rows[:, None], 1, destination]
        local = scene.rotate_xy(roots[..., :2]-self._env_origins[:, None, :2]-self._structure_shift[:, 1, None],
            -self._structure_yaw[:, 1, None])
        self._cross_completed |= eligible & ((local-center).abs() <= half).all(-1) & (
            roots[..., 2] > self._env_origins[:, None, 2]+self._surface_height[:, 1, None]+.6) & ~fallen
        distance = (box[:, :3]-self._bridge_landing_world).norm(dim=-1)
        aligned = (yaw-self._structure_yaw[:, 1]).sin().abs() < .1
        connected = (self._scene_ids == 1) & (distance < .2) & aligned
        self._bridge_arrived |= connected
        self._cross_exposure += torch.stack((eligible.sum(), eligible.new_tensor(self.num_envs*4, dtype=torch.long),
            4*(self._scene_ids == 0).sum(), eligible.new_tensor(0, dtype=torch.long))).float()
        other = fallen.any(-1)[:, None] & ~fallen
        entered = self._cross_entered
        if hasattr(self, '_compact_kind'):
            entered = entered & (self._compact_kind < 2)
        values = torch.stack((torch.ones_like(eligible), entered, self._cross_completed, fallen,
            other, self._bridge_arrived[:, None].expand_as(eligible),
            (self.progress_buf >= self.max_episode_length)[:, None].expand_as(eligible),
            other & ~self._cross_completed), -1).float()
        ended = eligible & ending[:, None]
        index = ((self._bridge_branch*3+self._bridge_span-1)*3)[:, None]+self._cross_phase
        self._cross_stats.view(-1, 8).index_add_(0, index[ended], values[ended])
        before = paired_people(self._before_predecessor)
        role = torch.where(self._before_predecessor >= 0, 2, torch.where(before, 1, 0))
        self._cross_role_stats.index_add_(0, role[ended], values[ended])
        scenes = ((self._scene_ids == 1) & ending).nonzero().flatten()
        scene_values = torch.stack((torch.ones_like(scenes),
            (self._cross_completed | ~eligible).all(-1)[scenes]), -1).float()
        self._cross_scene_stats.index_add_(0, self._bridge_branch[scenes], scene_values)

    def _consume_sampling_diagnostics(self):
        result = HumanoidMAStage2._consume_sampling_diagnostics(self)
        total = self._scene_sampling[:, 0].sum().clamp_min(1)
        for index, name in enumerate(('cliff', 'bridge')):
            sample, outcome = self._scene_sampling[index], self._scene_outcomes[index]
            prefix = 'scene/'+name+'/'
            result[prefix+'sampled_episodes'] = sample[0].clone()
            result[prefix+'sampling_fraction'] = sample[0]/total
            result[prefix+'ended_episodes'] = outcome[0].clone()
            result[prefix+'agent_falls'] = outcome[2].clone()
            result[prefix+'timeouts'] = outcome[3].clone()
        for branch, name in enumerate(BRANCHES):
            samples = self._branch_samples[branch]
            prefix = 'sampling/bridge/'+name+'/'
            for i, metric in enumerate(('episodes', 'before_pairs', 'before_people', 'crossers', 'coupled_people')):
                result[prefix+metric] = samples[i].clone()
            breakdown = [('all', self._cross_stats[branch].sum((0, 1)))]
            breakdown += [('span'+str(i+1), self._cross_stats[branch, i].sum(0)) for i in range(3)]
            breakdown += [(phase, self._cross_stats[branch, :, i].sum(0))
                for i, phase in enumerate(('entry', 'middle', 'exit'))]
            for label, sums in breakdown:
                self._crossing_metrics(result, f'crossing/{name}/{label}/', sums)
            counts = self._cross_scene_stats[branch]
            result[f'crossing/{name}/scene_episodes'] = counts[0].clone()
            result[f'crossing/{name}/all_crossed'] = counts[1]/counts[0].clamp_min(1)
        for role, sums in zip(('independent', 'before_leader', 'before_follower'), self._cross_role_stats):
            self._crossing_metrics(result, 'crossing/role/'+role+'/', sums)
        result['sampling/before_agent_fraction'] = 2*self._scene_sampling[:, 1].sum()/(4*total)
        result['sampling/crosser_agent_fraction'] = self._branch_samples[:, 3].sum()/(4*total)
        result['sampling/coupled_agent_fraction'] = self._branch_samples[:, 4].sum()/(4*total)
        for i, key in enumerate(('crosser_agent_steps', 'all_agent_steps', 'cliff_edge_required_agent_steps')):
            result['crossing/exposure/'+key] = self._cross_exposure[i].clone()
        result['crossing/exposure/crosser_fraction'] = self._cross_exposure[0]/self._cross_exposure[1].clamp_min(1)
        c = self._cliff_stats
        for i, key in enumerate(('ended_episodes', 'ended_agent_episodes', 'exited_agents', 'fallen_agents',
                                'dropped_agents', 'timeouts', 'near_edge_agent_steps', 'stopped_agent_steps', 'agent_steps')):
            result['cliff/'+key] = c[i].clone()
        result['cliff/exit_fraction'] = c[2]/c[1].clamp_min(1)
        result['cliff/drop_fraction'] = c[4]/c[1].clamp_min(1)
        result['cliff/timeout_fraction'] = c[5]/c[0].clamp_min(1)
        result['cliff/stopped_near_edge_fraction'] = c[7]/c[6].clamp_min(1)
        result['controller/environment_steps'] = self._controller_error[0].clone()
        result['controller/tracking_error_mean'] = self._controller_error[1]/self._controller_error[0].clamp_min(1)
        result['controller/tracking_error_max'] = self._controller_error[2].clone()
        result['crossing/update'] = torch.tensor(float(self._bridge_curriculum_update), device=self.device)
        for i, p in enumerate(span_probabilities(self._bridge_curriculum, self._bridge_curriculum_update)):
            result['crossing/span_probability'+str(i+1)] = torch.tensor(p, device=self.device)
        for values in (self._scene_sampling, self._scene_outcomes, self._branch_samples, self._cross_stats,
                       self._cross_scene_stats, self._cross_role_stats, self._cross_exposure,
                       self._cliff_stats, self._controller_error, self._bridge_start_samples):
            values.zero_()
        return result

    def _crossing_metrics(self, result, prefix, sums):
        result[prefix+'episodes'] = sums[0].clone()
        for i, metric in enumerate(METRICS[1:], 1):
            result[prefix+metric] = sums[i]/sums[0].clamp_min(1)

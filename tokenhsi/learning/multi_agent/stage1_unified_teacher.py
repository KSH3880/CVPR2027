from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from env.tasks.humanoid import compute_humanoid_observations_max
from env.tasks.basic_interaction_skills.humanoid_carry import compute_location_observations as carry_observations
from env.tasks.basic_interaction_skills.humanoid_sit import compute_location_observations as sit_observations
from env.tasks.basic_interaction_skills.humanoid_climb import compute_location_observations as climb_observations
from env.tasks.multi_agent.edge_ontop_reward import ontop_geometry, vertical_extent
from learning.amp_models import ModelAMPContinuous
from learning.transformer.amp_network_builder_transformer import AMPTransformerMultiTaskBuilder
from learning.transformer.trans_players import TransPlayerContinuous
from utils.edge_scenario_spec import classify_templates, agent_object_indices, agent_goal_indices
from utils.edge_ontop_spec import ON_TOP


ROOT = Path(__file__).resolve().parents[3]


class Stage1UnifiedTeacher:
    def __init__(self, checkpoint, device):
        train_path = ROOT / 'tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml'
        env_path = ROOT / 'tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml'
        with train_path.open() as f:
            params = yaml.safe_load(f)['params']
        with env_path.open() as f:
            self.env_cfg = yaml.safe_load(f)['env']

        self.multi_task_info = {
            'onehot_size': 4,
            'tota_subtask_obs_size': 127,
            'each_subtask_obs_size': [20, 38, 42, 27],
            'each_subtask_obs_indx': [0, 20, 58, 100, 127],
            'each_subtask_name': ['traj', 'sit', 'carry', 'climb'],
            'enable_task_mask_obs': True,
        }
        builder = AMPTransformerMultiTaskBuilder()
        builder.load(params['network'])
        config = dict(actions_num=32, input_shape=(354,), num_seqs=1,
                      amp_input_shape=(1330,), self_obs_size=223, task_obs_size=131,
                      multi_task_info=self.multi_task_info, device=device)
        self.model = ModelAMPContinuous.Network(builder.build('amp', **config)).to(device)
        self.rms = RunningMeanStd((354,)).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model.load_state_dict(state['model'], strict=True)
        self.rms.load_state_dict(state['running_mean_std'], strict=True)
        self.model.eval().requires_grad_(False)
        self.rms.eval().requires_grad_(False)

        self.player = TransPlayerContinuous.__new__(TransPlayerContinuous)
        self.player.model = self.model
        self.player.running_mean_std = self.rms
        self.player.normalize_input = params['config']['normalize_input']
        self.player.env = SimpleNamespace(task=SimpleNamespace(
            _enable_task_mask_obs=True,
            get_multi_task_info=lambda: self.multi_task_info))
        self.counts = torch.zeros(5, dtype=torch.long, device=device)

    @torch.no_grad()
    def observation(self, task, reset_ids=None):
        n, m = task.num_envs, task.num_agents
        graph = task.relation_runtime.graph
        scene = torch.arange(n, device=task.device).repeat_interleave(m)
        agent = torch.arange(m, device=task.device).repeat(n)
        template = classify_templates(graph, scene, agent, with_climb=True)
        self.counts += torch.bincount(template, minlength=5)

        body = torch.cat([task._rigid_body_pos, task._rigid_body_rot,
                          task._rigid_body_vel, task._rigid_body_ang_vel], dim=-1)
        if reset_ids is not None and len(reset_ids):
            body[reset_ids] = task._kinematic_humanoid_rigid_body_states[reset_ids]
        body = body.flatten(0, 1)
        human = compute_humanoid_observations_max(
            body[..., :3], body[..., 3:7], body[..., 7:10], body[..., 10:13],
            self.env_cfg['localRootObsPolicy'], self.env_cfg['rootHeightObsPolicy'])
        root = body[:, 0, :]

        primary = agent_object_indices(graph, scene, agent)
        boxes = task._logical_box_values(task._box_states)
        corners = task._logical_box_values(task._box_bps)
        sizes = task._logical_box_values(task._box_size)
        box = boxes[scene, primary]
        bps = corners[scene, primary]
        size = sizes[scene, primary]
        top = box[:, 2] + vertical_extent(box[:, 3:7], size / 2)

        task_obs = human.new_zeros(n * m, 131)
        carry = (template == 0) | (template == 3) | (template == 4)
        if carry.any():
            target = box[:, :3].clone()
            goal = agent_goal_indices(graph, scene, agent)
            target[template == 3] = task._tar_pos[scene[template == 3], goal[template == 3]]
            top_edge = graph.edge_valid[scene] & (graph.edge_owner[scene] == agent[:, None]) & \
                (graph.edge_relation[scene] == ON_TOP)
            support_id = torch.where(top_edge, graph.edge_dst[scene] - m, 0).amax(-1)
            support = boxes[scene, support_id]
            support_size = sizes[scene, support_id]
            _, geometry = ontop_geometry(box, support, size, support_size)
            target[template == 4] = geometry['target'][template == 4]
            task_obs[carry, 58:100] = carry_observations(
                root[carry], box[carry], bps[carry], target[carry], True)
            task_obs[carry, 127 + 2] = 1

        sit = template == 1
        if sit.any():
            target = box[:, :3].clone()
            target[:, 2] = top + task._relation_cfg['sit']['pelvis_clearance']
            facing = root.new_zeros(len(root), 3)
            facing[:, 0] = 1
            task_obs[sit, 20:58] = sit_observations(
                root[sit], target[sit], box[sit], bps[sit], facing[sit])
            task_obs[sit, 127 + 1] = 1

        climb = template == 2
        if climb.any():
            target = box[:, :3].clone()
            target[:, 2] = top + task._char_h
            task_obs[climb, 100:127] = climb_observations(
                root[climb], box[climb], bps[climb], target[climb])
            task_obs[climb, 127 + 3] = 1

        return torch.cat([human, task_obs], dim=-1)

    @torch.no_grad()
    def distribution(self, obs):
        normalized = self.player._preproc_obs(obs)
        mu, log_std = self.model.a2c_network.eval_actor(normalized, obs)
        return mu, log_std.exp().expand_as(mu)

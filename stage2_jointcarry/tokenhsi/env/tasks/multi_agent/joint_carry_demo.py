import copy

import torch
from isaacgym import gymapi

from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from utils.joint_carry_demo import demo_graph_spec, demo_layout
from utils.joint_carry_rsi import rotate_vectors, transform_states
from utils.joint_carry_spec import validate_env
from utils.motion_lib import MotionLib
from utils import torch_utils


class JointCarryDemo(HumanoidMACarry):
    _AGENT_COLOR_PALETTE = HumanoidMACarry._AGENT_COLOR_PALETTE + (
        (.15, .80, .75), (.90, .30, .25))

    def __init__(self, cfg, *args, **kwargs):
        if not cfg['args'].test or not cfg['args'].eval:
            raise ValueError('Mixed-scene viewer is inference-only')
        env = cfg['env']
        if (env['numAgents'], env['numObjects'], env['numEnvs']) not in ((6, 8, 1), (4, 8, 1)):
            raise ValueError('Mixed-scene viewer requires 4 or 6 humans / 8 objects / 1 env')
        self.demo_joint_teams = (env['numAgents'] - 2) // 2
        self.demo_solo_start = env['numAgents'] - 2
        self.demo_source_env = copy.deepcopy(env)
        self.demo_source_env.update(numAgents=2, numObjects=4)
        validate_env(self.demo_source_env)
        env['relationGraph'] = demo_graph_spec(env['numAgents'])
        env['debug']['reward'] = False
        env['relationReward']['diagnostics']['enabled'] = False
        super().__init__(cfg, *args, **kwargs)
        if self.viewer:
            self._update_camera()
            self._camera_follow_flag = False

    def _validate_joint_carry_env(self, env):
        validate_env(self.demo_source_env)

    def _load_motion(self, motion_file):
        self._skill_categories = ['loco']
        self._motion_lib = {'loco': MotionLib(motion_file=motion_file, skill='loco',
            dof_body_ids=self._dof_body_ids, dof_offsets=self._dof_offsets,
            key_body_ids=self._key_body_ids.cpu().numpy(), device=self.device)}

    def _load_box_asset(self):
        self._size_aware_rsi = False
        sizes = [[.52, .8, .4], [.52, .8, .4], [.4, .4, .4], [.4, .4, .4],
                 [.52, .8, .3], [.4, .4, .4], [.4, .4, .4], [.4, .4, .4]]
        self._box_size = torch.tensor(sizes, device=self.device)
        self._box_scale = self._box_size / self._box_size.new_tensor(self._build_base_size)
        self._box_density = torch.full((8,), 100., device=self.device)
        self._box_assets = []
        for size in sizes:
            options = gymapi.AssetOptions()
            options.angular_damping = .01
            options.linear_damping = .01
            options.max_angular_velocity = 100.
            options.density = 100.
            options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            self._box_assets.append(self.gym.create_box(self.sim, *size, options))
        self._box_size = self._box_size.reshape(1, 8, 3)
        self._box_scale = self._box_scale.reshape(1, 8, 3)

    def _reset_envs(self, ids):
        if not len(ids):
            return
        centers, goals = demo_layout(self.device)
        origin = self._env_origins[0]
        self._logical_box_order[ids] = torch.arange(8, device=self.device)
        assignment = [0, 0, 1, 1, 2, 3] if self.num_agents == 6 else [0, 0, 2, 3]
        self._agent_box_assignment[ids] = torch.tensor(assignment, device=self.device)
        boxes = torch.zeros(8, 13, device=self.device)
        boxes[:, 6] = 1.
        boxes[:, 2] = self._box_size[0, :, 2] / 2
        boxes[:4, :2] = centers[:, :2]
        boxes[4, :2] = goals[3, :2]
        boxes[5:, :2] = boxes.new_tensor([[-3.8, -2.8], [-3.8, 2.8], [3.8, 0.]])
        pool = self._joint_reset.pools['carryWith']
        rows = torch.randint(len(pool['root_state']), (self.demo_joint_teams,), device=self.device)
        self.demo_source_frames = pool['source_frame'][rows].tolist()
        for team, row in enumerate(rows):
            pair = slice(team * 2, team * 2 + 2)
            root, body, box = [pool[key][row].clone() for key in
                              ('root_state', 'body_state', 'box_state')]
            center = box[0, :3].clone()
            center[2] = 0.
            yaw = boxes.new_tensor(0.)
            shift = centers[team] - center + origin
            self._humanoid_root_states[0, pair] = transform_states(root, yaw, shift)
            self._kinematic_humanoid_rigid_body_states[0, pair] = transform_states(body, yaw, shift)
            self._dof_pos[0, pair] = pool['dof_position'][row]
            self._dof_vel[0, pair] = pool['dof_velocity'][row]
            boxes[team] = transform_states(box, yaw, centers[team] - center)[0]
        lib = self._motion_lib['loco']
        motions = lib.sample_motions(2)
        times = lib.sample_time_rsi(motions)
        pos, rot, dof, vel, ang, dof_vel, _ = lib.get_motion_state(motions, times)
        desired = centers[2:].clone()
        desired[:, 0] -= 1.5
        yaw = -torch_utils.calc_heading(rot)
        center = pos.clone()
        center[:, 2] = 0.
        shift = desired - rotate_vectors(center, yaw) + origin
        root = torch.cat((pos, rot, vel, ang), -1)
        body = torch.cat(lib.get_motion_state_max(motions, times), -1)
        self._humanoid_root_states[0, self.demo_solo_start:] = transform_states(root, yaw, shift)
        self._kinematic_humanoid_rigid_body_states[0, self.demo_solo_start:] = transform_states(body, yaw, shift)
        self._dof_pos[0, self.demo_solo_start:], self._dof_vel[0, self.demo_solo_start:] = dof, dof_vel
        self._every_env_init_dof_pos[ids] = self._dof_pos[ids]
        boxes[:, :3] += origin
        self._box_states[ids] = boxes
        goal_pos = torch.zeros(self.num_agents, 3, device=self.device)
        for team in range(self.demo_joint_teams):
            goal_pos[team * 2:team * 2 + 2] = goals[team]
        goal_pos[self.demo_solo_start:] = goals[2:]
        goal_pos[:, 2] = self._box_size[0, assignment, 2] / 2
        goal_pos[-1, 2] += self._box_size[0, 4, 2]
        self._tar_pos[ids] = goal_pos + origin
        self._joint_reset.commit(ids)
        self._refresh_sim_tensors()
        self._compute_observations(ids)
        self._compute_amp_observations(ids)
        self._hist_amp_obs_buf[ids] = self._curr_amp_obs_buf[ids, :, None]
        self._update_box_assignment_colors(ids)

    def _compute_reward(self, actions):
        self.rew_buf.zero_()

    def _compute_metrics_evaluation(self):
        pass

    def _draw_ontop_context(self):
        if self.viewer is None:
            return
        from env.tasks.multi_agent.scene_features import task_marker_vertices
        from utils.edge_ontop_spec import ON_TOP
        import numpy as np
        self.gym.clear_lines(self.viewer)
        target = self._box_states[0, 4, :3].clone()
        target[2] += (self._box_size[0, 3, 2] + self._box_size[0, 4, 2]) / 2
        vertices = task_marker_vertices(ON_TOP, target.cpu().numpy())
        color = self._agent_color(self.num_agents - 1)
        colors = np.tile((color.x, color.y, color.z), (len(vertices) // 2, 1)).astype(np.float32)
        self.gym.add_lines(self.viewer, self.envs[0], len(vertices) // 2, vertices, colors)

    def _update_camera(self):
        if self._camera_follow_flag:
            self.gym.viewer_camera_look_at(self.viewer, None,
                gymapi.Vec3(5., -6.5, 6.5), gymapi.Vec3(0., 0., .6))

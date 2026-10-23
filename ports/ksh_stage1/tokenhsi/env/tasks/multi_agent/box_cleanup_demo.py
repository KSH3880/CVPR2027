import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_mul, quat_rotate

from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from env.tasks.multi_agent.edge_ontop_reward import vertical_extent
from utils import torch_utils
from utils.box_cleanup_spec import BOX_SIZE, cleanup_graph_spec, cleanup_layout
from utils.motion_lib import MotionLib


class BoxCleanupDemo(HumanoidMACarry):
    def __init__(self, cfg, *args, **kwargs):
        if not cfg['args'].test or not cfg['args'].eval:
            raise ValueError('Box cleanup is an inference-only demo')
        env = cfg['env']
        if (env['numAgents'], env['numObjects'], env['numEnvs']) != (4, 16, 1):
            raise ValueError('Box cleanup requires 4 agents, 16 boxes, 1 environment')
        env['relationGraph'] = cleanup_graph_spec()
        env['stateInit'] = 'Start'
        env['agentSpawnRadius'] = 2.65
        env['box']['build'].update(baseSize=list(BOX_SIZE), randomSize=True,
            randomModeEqualProportion=False, scaleRangeX=[1., 1.25],
            scaleRangeY=[1., 1.25], scaleRangeZ=[.75, 1.], scaleSampleInterval=.125)
        env['box']['reset']['randomAssignment'] = False
        env['box']['reset']['randomRot'] = False
        env['debug']['reward'] = False
        env['relationReward']['diagnostics']['enabled'] = False
        super().__init__(cfg, *args, **kwargs)
        self.cleanup_round = 0
        self.cleanup_stable = torch.zeros(4, dtype=torch.long, device=self.device)
        self.cleanup_waiting = torch.zeros(4, dtype=torch.bool, device=self.device)
        self.cleanup_completed = torch.zeros(8, dtype=torch.bool, device=self.device)
        self.cleanup_events = []
        self.cleanup_finished = False

    def _load_motion(self, motion_file):
        self._skill_categories = ['loco']
        self._motion_lib = {'loco': MotionLib(motion_file=motion_file, skill='loco',
            dof_body_ids=self._dof_body_ids, dof_offsets=self._dof_offsets,
            key_body_ids=self._key_body_ids.cpu().numpy(), device=self.device)}

    def _reset_actors(self, ids):
        env, agent = self._expand_slots(ids)
        lib = self._motion_lib['loco']
        motions = lib.sample_motions(len(env))
        times = torch.zeros(len(env), device=self.device)
        pos, rot, dof, vel, ang_vel, dof_vel, _ = lib.get_motion_state(motions, times)
        offsets = self._agent_spawn_offsets[agent] + self._env_origins[env]
        self._set_env_state(env, agent, pos + offsets, rot, dof, vel, ang_vel, dof_vel)
        body_pos, body_rot, body_vel, body_ang = lib.get_motion_state_max(motions, times)
        self._kinematic_humanoid_rigid_body_states[env, agent] = torch.cat(
            (body_pos + offsets[:, None], body_rot, body_vel, body_ang), -1)
        self._reset_ref_slots['loco'] = (env, agent)
        self._reset_ref_motion_ids['loco'] = motions
        self._reset_ref_motion_times['loco'] = times
        self._every_env_init_dof_pos[env, agent] = dof

    def _update_box_assignment_colors(self, ids):
        if self.viewer is None and not self._video_enabled:
            return
        for env_id in ids.cpu().tolist():
            for object_id in range(self.num_objects):
                handle = self._box_handles[env_id * self.num_objects + object_id]
                self.gym.set_rigid_body_color(self.envs[env_id], handle, 0,
                                              gymapi.MESH_VISUAL, gymapi.Vec3(.45, .45, .45))

    def _reset_ontop_context_envs(self, ids):
        if not len(ids):
            return
        self.cleanup_round = 0
        self.cleanup_stable.zero_()
        self.cleanup_waiting.zero_()
        self.cleanup_completed.zero_()
        self.cleanup_events = []
        self.cleanup_finished = False
        self._relation_graph_spec = cleanup_graph_spec()
        self._sample_episode_graph(ids)
        self._reset_default_slots = None
        self._reset_ref_slots = {}
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}
        self._reset_actors(ids)
        positions, boxes, rotations, self.cleanup_goals = cleanup_layout(self._box_size[0])
        roots = self._humanoid_root_states[0].clone()
        kin = self._kinematic_humanoid_rigid_body_states[0].clone()
        heading = torch_utils.calc_heading(roots[:, 3:7])
        target = torch.atan2(positions[:, 1], positions[:, 0]) + torch.pi
        delta = target - heading
        rotation = torch.zeros(4, 4, device=self.device)
        rotation[:, 2] = torch.sin(delta / 2)
        rotation[:, 3] = torch.cos(delta / 2)
        new_roots = roots[:, :3].clone()
        new_roots[:, :2] = positions + self._env_origins[0, :2]
        q = rotation[:, None].expand(-1, self.num_bodies, -1).reshape(-1, 4)
        relative = (kin[..., :3] - roots[:, None, :3]).reshape(-1, 3)
        kin[..., :3] = quat_rotate(q, relative).view(4, self.num_bodies, 3) + new_roots[:, None]
        kin[..., 3:7] = quat_mul(q, kin[..., 3:7].reshape(-1, 4)).view(4, self.num_bodies, 4)
        kin[..., 7:13] = 0
        roots[:, :3] = new_roots
        roots[:, 3:7] = quat_mul(rotation, roots[:, 3:7])
        roots[:, 7:13] = 0
        self._humanoid_root_states[0] = roots
        self._kinematic_humanoid_rigid_body_states[0] = kin
        self._dof_vel.zero_()
        self._logical_box_order[ids] = torch.arange(16, device=self.device)
        self._agent_box_assignment[ids] = torch.arange(4, device=self.device)
        self._box_states[ids] = 0
        self._box_states[ids, :, :3] = boxes + self._env_origins[ids, None]
        self._box_states[ids, :, 3:7] = rotations
        self._tar_pos[ids] = self.cleanup_goals[0] + self._env_origins[ids, None]
        self._reset_env_tensors(ids)
        self._refresh_sim_tensors()
        self._reset_relation_history(ids)
        self._compute_observations(ids)
        self._init_amp_obs(ids)
        self._update_box_assignment_colors(ids)
        print('[cleanup] round=1 targets=0,1,2,3', flush=True)

    def post_physics_step(self):
        super().post_physics_step()
        if self.cleanup_finished or self.reset_buf.any():
            return
        start = 4 * self.cleanup_round
        boxes = self._box_states[0, start:start + 4]
        near = (boxes[:, :2] - self._tar_pos[0, :, :2]).norm(dim=-1) < .12
        extent = vertical_extent(boxes[:, 3:7], self._box_size[0, start:start + 4] / 2)
        grounded = (boxes[:, 2] - extent - self._env_origins[0, 2]).abs() < .025
        still = (boxes[:, 7:10].norm(dim=-1) < .15) & (boxes[:, 10:13].norm(dim=-1) < .5)
        settled = near & grounded & still
        self.cleanup_stable = torch.where(settled, self.cleanup_stable + 1, 0)
        ready = self.cleanup_stable >= 15
        newly = ready & ~self.cleanup_waiting
        for agent in newly.nonzero().flatten().tolist():
            self.cleanup_events.append(dict(step=int(self.progress_buf[0]), round=self.cleanup_round + 1,
                                           agent=agent, box=start + agent))
            print('[cleanup] round={} agent={} box={} delivered'.format(self.cleanup_round + 1, agent, start + agent), flush=True)
        self.cleanup_waiting |= ready
        self.cleanup_completed[start:start + 4] |= ready
        if not ready.all():
            return
        if self.cleanup_round == 1:
            self.cleanup_finished = True
            print('[cleanup] completed 2 rounds / 8 boxes', flush=True)
            return
        self.cleanup_round = 1
        self.cleanup_waiting.zero_()
        self.cleanup_stable.zero_()
        self._relation_graph_spec = cleanup_graph_spec(1)
        ids = torch.zeros(1, dtype=torch.long, device=self.device)
        self._sample_episode_graph(ids)
        self._agent_box_assignment[0] = torch.arange(4, 8, device=self.device)
        self._tar_pos[0] = self.cleanup_goals[1] + self._env_origins[0]
        self._reset_relation_history(ids)
        self._compute_observations()
        self._update_box_assignment_colors(ids)
        if self._enable_markers:
            self._marker_pos[0] = self._tar_pos[0]
            actor_ids = self._marker_actor_ids.flatten().contiguous()
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
        self.extras['policy_obs'] = self.obs_buf.clone()
        print('[cleanup] round=2 targets=4,5,6,7 (scene preserved)', flush=True)

    def _update_camera(self):
        if not hasattr(self, '_box_states'):
            return super()._update_camera()
        if self._camera_follow_flag:
            origin = self._env_origins[0].cpu().tolist()
            self.gym.viewer_camera_look_at(self.viewer, None,
                gymapi.Vec3(origin[0] + 4.2, origin[1] - 5.6, 6.8),
                gymapi.Vec3(origin[0], origin[1], .4))

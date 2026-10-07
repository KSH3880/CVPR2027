from isaacgym import gymapi
import torch

from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
from utils.motion_lib import MotionLib


class AtGoalDemo(HumanoidMACarry):
    def __init__(self, cfg, *args, **kwargs):
        if not cfg['args'].test or not cfg['args'].eval:
            raise ValueError('AT goal demo is inference-only')
        if (cfg['env']['numAgents'], cfg['env']['numObjects'], cfg['env']['numEnvs']) != (2, 3, 1):
            raise ValueError('AT goal demo requires 2 agents, 3 boxes, 1 environment')
        self.goal_height = 0.
        self.goal_xy_mode = 'above_box'
        probabilities = cfg['env']['relationGraph']['template_probabilities']
        for name in probabilities:
            probabilities[name] = float(name == 'HOLDING_AT')
        cfg['args'].task_graph = 'random_scenario'
        cfg['env']['debug']['reward'] = False
        cfg['env']['relationReward']['diagnostics']['enabled'] = False
        super().__init__(cfg, *args, **kwargs)

    def _load_motion(self, motion_file):
        self._skill_categories = ['loco']
        self._motion_lib = {'loco': MotionLib(motion_file=motion_file, skill='loco',
            dof_body_ids=self._dof_body_ids, dof_offsets=self._dof_offsets,
            key_body_ids=self._key_body_ids.cpu().numpy(), device=self.device)}

    def _reset_env_tensors(self, env_ids):
        env, agent = self._expand_slots(env_ids)
        boxes = self._scenario_physical_object(env, agent)
        goals = self._scenario_goal(env, agent)
        box_xy = self._box_states[env, boxes, :2]
        human_xy = self._humanoid_root_states[env, agent, :2]
        target_xy = box_xy.clone()
        if self.goal_xy_mode == 'midpoint':
            target_xy = (box_xy + human_xy) / 2
        elif self.goal_xy_mode == 'side':
            direction = box_xy - human_xy
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            target_xy += torch.stack((-direction[:, 1], direction[:, 0]), -1)
        self._tar_pos[env, goals, :2] = target_xy
        self._tar_pos[env, goals, 2] = self._env_origins[env, 2] + self.goal_height
        self.demo_boxes = boxes
        self.demo_goals = goals
        self.demo_camera_xy = torch.cat((human_xy, box_xy, target_xy)).mean(0).cpu().tolist()
        super()._reset_env_tensors(env_ids)

    def _build_marker(self, env_id, agent_id, env_ptr):
        super()._build_marker(env_id, agent_id, env_ptr)
        self.gym.set_actor_scale(env_ptr, self._marker_handles[-1], 1.5)

    def _update_camera(self):
        if not hasattr(self, 'demo_camera_xy'):
            return super()._update_camera()
        if self._camera_follow_flag:
            x, y = self.demo_camera_xy
            z = float(self._env_origins[0, 2])
            self.gym.viewer_camera_look_at(self.viewer, None,
                gymapi.Vec3(x + 4.5, y - 6., z + 4.5),
                gymapi.Vec3(x, y, z + 1.))

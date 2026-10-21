import isaacgym
import torch

from env.tasks.multi_agent.cliff_geometry import height_observation, sensor_grid, stage_height
from env.tasks.multi_agent.humanoid_ma import HumanoidMA
from env.tasks.multi_agent.humanoid_ma_cliff import HumanoidMACliff
from env.tasks.multi_agent.scene_features import build_gta_pose_records
from utils import torch_utils


def state(origins):
    n = len(origins)
    task = object.__new__(HumanoidMACliff)
    task._cliff = {'height': 1.5}
    task._clean_local_root_obs_policy = True
    task._clean_root_height_obs_policy = True
    task._num_obs = 223
    task.num_bodies = 15
    task.device = 'cpu'
    task._env_origins = origins
    root = torch.zeros(n, 2, 13)
    root[..., :3] = origins[:, None] + torch.tensor([[[.3, -.4, .94], [-.5, .6, .96]]])
    root[..., 6] = 1
    root[..., 7:10] = torch.tensor([.8, -.2, .1])
    root[..., 10:13] = torch.tensor([.1, .2, .3])
    body = root[:, :, None].expand(-1, -1, 15, -1).clone()
    offsets = torch.randn(1, 2, 15, 3) * .2
    offsets[:, :, 0] = 0
    body[..., :3] += offsets
    task._humanoid_root_states = root
    task._kinematic_humanoid_rigid_body_states = body
    return task


def test_rsi_placement_preserves_heading_local_human_features_and_relative_body_states():
    torch.manual_seed(11)
    origins = torch.tensor([[0., 0., 0.], [48., -32., 0.], [704., 512., 0.]])
    task = state(origins)
    old = task._kinematic_humanoid_rigid_body_states.clone()
    before = HumanoidMA._compute_clean_humanoid_nodes(task, old[..., :3], old[..., 3:7],
                                                     old[..., 7:10], old[..., 10:13], origins)
    xy = origins[:, None, :2] + torch.tensor([[[1.2, -.5], [-1.1, .7]]])
    yaw = torch.tensor([[0.7, -2.1]]).expand(3, -1)
    task._place_humans(torch.arange(3), xy, yaw)
    body = task._kinematic_humanoid_rigid_body_states
    after = task._compute_clean_humanoid_nodes(body[..., :3], body[..., 3:7],
                                              body[..., 7:10], body[..., 10:13], origins)
    torch.testing.assert_close(before, after, atol=2e-4, rtol=1e-5)
    torch.testing.assert_close(body[..., 0, :3], task._humanoid_root_states[..., :3])
    heading = torch_utils.calc_heading(body[..., 0, 3:7].reshape(-1, 4)).view_as(yaw)
    torch.testing.assert_close(heading, yaw)
    torch.testing.assert_close(task._humanoid_root_states[..., :2], xy)
    torch.testing.assert_close(task._humanoid_root_states[..., 2], old[..., 0, 2] + 1.5)


def test_sensor_queries_are_translation_invariant_for_far_apart_environments():
    origins = torch.tensor([[0., 0., 0.], [48., -32., 0.], [704., 512., 0.]])
    root = origins[:, None] + torch.tensor([[[1.37, -.21, 2.44], [-1.23, .17, 2.47]]])
    heading = torch.tensor([[.31, -1.14]]).expand(3, -1)
    half = torch.tensor([[2.25, 3.5]]).expand(3, -1)
    grid = sensor_grid(32, 2., 'cpu')
    sensors = height_observation(root, heading, grid, origins, half, 1.5, 3.)
    torch.testing.assert_close(sensors, sensors[:1].expand_as(sensors), atol=1e-6, rtol=0)
    ground = stage_height(root[..., :2], origins, half, 1.5)
    torch.testing.assert_close(ground, torch.full((3, 2), 1.5))


def test_gta_poses_keep_object_drop_and_remove_only_environment_origin():
    origins = torch.tensor([[0., 0., 0.], [48., -32., 0.]])
    humans = origins[:, None] + torch.tensor([[[1., 0., 2.44], [-1., 0., 2.44]]])
    boxes = origins[:, None] + torch.tensor([[[3., 0., .2], [-3., 0., .3], [0., 4., .25]]])
    goals = origins[:, None].expand(-1, 2, -1)
    hrot = torch.zeros(2, 2, 4); hrot[..., 3] = 1
    orot = torch.zeros(2, 3, 4); orot[..., 3] = 1
    poses = build_gta_pose_records(humans, hrot, boxes, orot, goals, origins)
    torch.testing.assert_close(poses[0], poses[1])
    torch.testing.assert_close(poses[:, 2, 2] - poses[:, 0, 2], torch.full((2,), -2.24))

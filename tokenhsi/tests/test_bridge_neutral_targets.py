import isaacgym
import pytest
import torch

from env.tasks.multi_agent.humanoid_ma_sensor_bridge_before import HumanoidMASensorBridgeBefore
from env.tasks.multi_agent.humanoid_ma_sensor_compact import HumanoidMASensorCompact
from env.tasks.multi_agent.edge_interaction_reward import evaluate_interaction_edges
from env.tasks.multi_agent.before_reward import predecessor_at_progress
from env.tasks.multi_agent.square_sensor_scene import yaw_quat
from utils.geometry import rotate
from utils import stage2_spec
from utils.evaluation_protocol import spatial_success
from utils.experiments import native_config


@pytest.mark.parametrize('yaw', [0., 1.3, -2.4])
@pytest.mark.parametrize('default', [True, False])
def test_default_follower_targets_follow_final_root_without_moving_active_goals(monkeypatch, yaw, default):
    task = object.__new__(HumanoidMASensorBridgeBefore)
    task._state_init = task.StateInit.Default if default else task.StateInit.Random
    task._roles = torch.tensor([[0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0]])
    task._before_same_platform = torch.tensor([True, False, False])
    origins = torch.tensor([[500., 100., 0.], [900., -200., 0.], [6000., 6000., 0.]])
    roots = torch.zeros(3, 4, 13)
    roots[..., 0] = torch.arange(4)*1.5
    roots[..., 2], roots[..., 6] = .94, 1.
    if not default:
        roots[..., :3] += origins[:, None]
    task._humanoid_root_states = roots
    task._tar_pos = origins[:, None].expand(-1, 4, -1).clone()
    follower = task._roles[:, 3]
    inactive = torch.zeros(3, 4, dtype=torch.bool)
    inactive[torch.arange(3), follower] = True
    task._tar_pos[~inactive] = roots[..., :3][~inactive]+torch.tensor([2., 1., -.6])
    original_goals, original_roots = task._tar_pos.clone(), roots.clone()
    transformed = task._tar_pos.clone()

    def placed(self, ids):
        q = yaw_quat(torch.tensor([yaw]))[0]
        for e in ids.tolist():
            offset = origins[e]+torch.tensor([-7. if self._before_same_platform[e] else 7., 0., 1.56])
            self._humanoid_root_states[e, :, :3] = rotate(q, roots[e, :, :3]-origins[e])+offset
            self._humanoid_root_states[e, :, 2] += .03
            self._tar_pos[e] = rotate(q, self._tar_pos[e]-origins[e])+offset
            transformed[e] = self._tar_pos[e]

    monkeypatch.setattr(HumanoidMASensorCompact, '_place_scene', placed)
    ids = torch.tensor([2, 0])
    task._place_scene(ids)
    if default:
        torch.testing.assert_close(task._tar_pos[ids, follower[ids]], roots[ids, follower[ids], :3], rtol=0, atol=0)
    else:
        torch.testing.assert_close(task._tar_pos, transformed, rtol=0, atol=0)
    torch.testing.assert_close(task._tar_pos[~inactive], transformed[~inactive], rtol=0, atol=0)
    torch.testing.assert_close(task._tar_pos[1], original_goals[1], rtol=0, atol=0)
    torch.testing.assert_close(roots[1], original_roots[1], rtol=0, atol=0)
    torch.testing.assert_close(roots[..., 3:], original_roots[..., 3:], rtol=0, atol=0)


@pytest.mark.parametrize('family', [1, 2, 4])
def test_inactive_follower_target_correction_preserves_rewards_and_success(family):
    torch.manual_seed(42)
    templates = torch.tensor([[5, 3, 5, family]])
    pre = torch.tensor([[-1, -1, -1, 1]])
    graph = stage2_spec.compose_graph(templates, torch.tensor([[0, 1, 0, 1]]), pre, 8, shuffle=True)
    roots, hands, feet = torch.rand(1, 4, 3), torch.rand(1, 4, 2, 3), torch.rand(1, 4, 2, 3)
    objects = torch.zeros(1, 8, 13)
    objects[..., :3], objects[..., 6] = torch.rand(1, 8, 3), 1.
    sizes = torch.rand(1, 8, 3)+.2
    goals = torch.rand(1, 4, 3)
    goals[pre >= 0] = 3000.
    corrected = torch.where((pre >= 0)[..., None], roots, goals)
    cfg = native_config('stage2_sensor_900_bridge_before_fixed_layout_cache')['env']['relationReward']
    a, b = [evaluate_interaction_edges(hands, feet, roots, objects, sizes, target, graph, cfg, .94)
            for target in (goals, corrected)]
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    for key in a[1]:
        torch.testing.assert_close(a[1][key], b[1][key], rtol=0, atol=0)
    before = [predecessor_at_progress(result[1]['progress'], roots, objects, sizes, target, graph, cfg)
              for result, target in ((a, goals), (b, corrected))]
    torch.testing.assert_close(*before, rtol=0, atol=0)
    packet = stage2_spec.task_graph_packet(graph, 1).reshape(1, 4, 5)
    torch.testing.assert_close(spatial_success(packet, roots, objects, sizes, goals),
                               spatial_success(packet, roots, objects, sizes, corrected), rtol=0, atol=0)

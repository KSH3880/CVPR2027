"""Mixed-task diagnostics must measure CONTACT directly, excluding other tasks."""
from types import SimpleNamespace

import pytest
import torch

from env.tasks.multi_agent.edge_context_task import EdgeContextTaskMixin
from env.tasks.multi_agent.jointpush_reward import Stage1PushRuntime
from env.tasks.multi_agent.relation_diagnostics import split_reward_tensorboard_tag
from utils.coupled_push_spec import compose_graph
from utils.experiments import native_config
from utils.stage1_jointpush import PUSH


@pytest.mark.parametrize("templates", [[5, 3, 1, 2], [5, 5, 5, 5], [3, 4, 1, 2]])
@pytest.mark.parametrize("premature_penalty", [False, True])
def test_contact_edge_means_and_reward_are_separate_in_mixed_tasks(templates, premature_penalty):
    graph = compose_graph(torch.tensor([templates]), torch.arange(4)[None], False,
                          region_goals=True)
    cfg = native_config("stage2_coupled_push_region_goals")["env"]["relationReward"]
    runtime = Stage1PushRuntime(1, graph, cfg, "cpu", contact_weights=[.2, .2, .2])
    phi = torch.linspace(.1, .8, graph.edge_valid.numel()).reshape_as(graph.edge_valid)
    zero = torch.zeros_like(phi)
    diag = {key: zero for key in ("progress", "z_error", "feet_height_error",
                                 "region_error", "distance", "distance_xy")}
    task = SimpleNamespace(
        num_envs=1, num_agents=4, relation_runtime=runtime,
        _humanoid_root_states=torch.zeros(1, 4, 13), _box_states=torch.zeros(1, 8, 13),
        _reward_box_values=lambda x: x[:, :4], _evaluate_relations=lambda: (phi, diag),
        _power_reward=False, _agent_collision_penalty=False, _box_vel_penalty=False,
        _relation_cfg=dict(cfg, diagnostics={"enabled": True, "sample_envs": 0}),
        _reward_term_sums=torch.zeros(7), _reward_term_count=0,
        rew_buf=torch.zeros(4), extras={}, _edge_steps=0,
        _edge_ever_goal=torch.zeros(1, 4, dtype=torch.bool),
        _edge_ever_scene=torch.zeros(1, dtype=torch.bool),
        _edge_goal_owners=torch.ones(1, 4, dtype=torch.bool),
        _edge_metric_sums={}, _edge_metric_denominators={}, _edge_metric_steps=0,
        _edge_episode_sums=torch.zeros(6), _record_ontop_diagnostics=lambda *x: None,
        _consume_sampling_diagnostics=lambda: {},
    )
    if premature_penalty:
        task._premature_motion_penalty = torch.tensor([[-.2, 0., -.1, 0.]])
    EdgeContextTaskMixin._compute_relation_reward(task, None)
    expected_reward = runtime.last_result["agent_task_reward"].flatten().clone()
    diagnostics = EdgeContextTaskMixin.consume_relation_diagnostics(task)
    if premature_penalty:
        expected_reward += task._premature_motion_penalty.flatten()
    torch.testing.assert_close(task.rew_buf, expected_reward)
    torch.testing.assert_close(task.extras["reward_terms"][:, -1], expected_reward)
    for relation, name in ((7, "at"), (PUSH, "contact")):
        mask = graph.edge_valid & (graph.edge_relation == relation)
        for field in ("state_component", "progress_component", "success_component", "total"):
            key = "edge/" + name + "/" + field
            if mask.any():
                expected = runtime.last_result[field][mask].mean()
                torch.testing.assert_close(diagnostics[key], expected)
                assert split_reward_tensorboard_tag(key).startswith("reward_terms/" + name + "_")
            else:
                assert key not in diagnostics
    # Mixed SIT/CLIMB/holding reward must never be assigned to CONTACT.
    if templates == [5, 3, 1, 2]:
        assert diagnostics["agent/task_total"] - diagnostics["edge/at/total"] != diagnostics["edge/contact/total"]
    assert split_reward_tensorboard_tag("edge/sit/total") is None
    assert split_reward_tensorboard_tag("edge/contact/own_success") is None

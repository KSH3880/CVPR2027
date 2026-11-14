"""CPA geometry, reward replacement and static/CPA checkpoint isolation."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
from env.tasks.multi_agent.collision_reward import (
    agent_collision_config,
    check_agent_collision_checkpoint,
    compute_agent_cpa_collision_penalty,
)
from env.tasks.multi_agent.edge_context_task import EdgeContextTaskMixin
from utils.edge_scenario_spec import sample_graph

ROOT = Path(__file__).resolve().parents[1]
CPA = yaml.safe_load((ROOT / "data/cfg/multi_agent/stage1.yaml").read_text())


def test_direct_approach_is_penalized_before_static_threshold_with_step_discount():
    pos = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 3.0]]])
    vel = torch.tensor([[[1.0, 0.0, 10.0], [-1.0, 0.0, -10.0]]])
    actual = compute_agent_cpa_collision_penalty(pos, vel, 0.7, 0.99, 1 / 30)
    torch.testing.assert_close(actual, torch.full((1, 2), 0.99**30))
    torch.testing.assert_close(
        compute_agent_cpa_collision_penalty(pos, vel, 0.7, 0.99, 1 / 60),
        torch.full((1, 2), 0.99**60),
    )


@pytest.mark.parametrize(
    "kind", ["stationary", "receding", "parallel", "miss", "coincident"]
)
def test_no_cpa_risk_for_non_closing_or_safe_passes_and_stationary_overlap(kind):
    pos = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]])
    vel = torch.zeros_like(pos)
    if kind == "receding":
        vel[0, 0, 0] = -1
    elif kind == "parallel":
        vel[..., 1] = 1
    elif kind == "miss":
        pos[0, 1] = torch.tensor([2.0, 1.0, 0.0])
        vel[0, 0, 0] = 1
    elif kind == "coincident":
        pos.zero_()
        vel[0, 0, 0] = 1
    result = compute_agent_cpa_collision_penalty(pos, vel, 0.7, 0.99, 1 / 30)
    assert torch.isfinite(result).all() and (not result.any())


def test_cpa_is_bounded_symmetric_and_invariant_to_translation_velocity_and_human_order():
    torch.manual_seed(7)
    pos = torch.randn(16, 4, 3)
    vel = torch.randn_like(pos)
    risk = compute_agent_cpa_collision_penalty(pos, vel, 0.7, 0.99, 1 / 30)
    assert torch.isfinite(risk).all() and ((risk >= 0) & (risk <= 1)).all()
    order = torch.tensor([2, 0, 3, 1])
    shuffled = compute_agent_cpa_collision_penalty(
        pos[:, order], vel[:, order], 0.7, 0.99, 1 / 30
    )
    torch.testing.assert_close(shuffled, risk[:, order])
    shifted = compute_agent_cpa_collision_penalty(
        pos + torch.tensor([8.0, -5.0, 20.0]),
        vel + torch.tensor([4.0, 3.0, -11.0]),
        0.7,
        0.99,
        1 / 30,
    )
    torch.testing.assert_close(shifted, risk, atol=2e-06, rtol=2e-05)
    for i in range(4):
        pairs = [
            compute_agent_cpa_collision_penalty(
                pos[:, [i, j]], vel[:, [i, j]], 0.7, 0.99, 1 / 30
            )[:, 0]
            for j in range(4)
            if j != i
        ]
        torch.testing.assert_close(risk[:, i], torch.stack(pairs, -1).max(-1).values)
    assert not compute_agent_cpa_collision_penalty(
        pos[:, :1], vel[:, :1], 0.7, 0.99, 1 / 30
    ).any()


@pytest.mark.parametrize(
    "key,value",
    [
        ("agentCollisionMode", "bad"),
        ("agentCollisionPenalty", 1),
        ("agentCollisionCoeff", -1),
        ("agentCollisionCoeff", float("nan")),
        ("agentCollisionDist", 0),
        ("agentCollisionTTCDiscount", 0),
        ("agentCollisionTTCDiscount", 1.01),
    ],
)
def test_invalid_collision_settings_fail_instead_of_silently_changing_rewards(
    key, value
):
    env = deepcopy(CPA["env"])
    env[key] = value
    with pytest.raises(ValueError):
        agent_collision_config(env)


def test_edge_reward_pays_selected_cpa_once_and_preserves_collision_log_key():
    task = SimpleNamespace(
        num_envs=1,
        num_agents=2,
        _edge_context=True,
        _edge_interaction=False,
        _edge_ontop=False,
        _power_reward=False,
        _agent_collision_penalty=True,
        _agent_collision_coeff=0.5,
        _agent_collision_dist=0.7,
        _box_vel_penalty=False,
        _edge_steps=0,
        _relation_cfg={"diagnostics": {"enabled": False}},
        extras={},
        rew_buf=torch.zeros(2),
        _reward_term_sums=torch.zeros(7),
        _reward_term_count=0,
        _edge_ever_goal=torch.zeros(1, 2, dtype=torch.bool),
        _edge_ever_scene=torch.zeros(1, dtype=torch.bool),
    )
    graph = sample_graph(1, CPA["env"]["relationGraph"], preset="holding_at")
    shape = graph.edge_valid.shape
    result = {
        k: torch.zeros(shape)
        for k in [
            "state_component",
            "progress_component",
            "success_component",
            "reward_saturated",
        ]
    }
    result["agent_task_reward"] = torch.ones(1, 2)
    task.relation_runtime = SimpleNamespace(
        graph=graph,
        done=torch.zeros(1, 2, dtype=torch.bool),
        own_success=torch.zeros(shape, dtype=torch.bool),
        step=lambda *args, **kwargs: result,
    )
    task._humanoid_root_states = torch.zeros(1, 2, 13)
    task._humanoid_root_states[0, 1, 0] = 0.35
    task._humanoid_root_states[0, 0, 7] = 0.5
    task._humanoid_root_states[0, 1, 7] = -0.5
    task._box_states = torch.zeros(1, 4, 13)
    task._reward_box_values = lambda values: values
    task._evaluate_relations = lambda: (
        torch.zeros(shape),
        {
            "progress": torch.zeros(shape),
            "z_error": torch.zeros(shape),
            "feet_height_error": torch.zeros(shape),
            "region_error": torch.zeros(shape),
        },
    )
    selected = lambda roots, distance: compute_agent_cpa_collision_penalty(
        roots, task._humanoid_root_states[..., 7:10], distance, 0.99, 1 / 30
    )
    EdgeContextTaskMixin._compute_relation_reward(task, selected)
    expected = -0.5 * 0.99**10.5
    torch.testing.assert_close(task.rew_buf, torch.full((2,), 1 + expected))
    torch.testing.assert_close(
        task.extras["reward_terms"][:, 4], torch.full((2,), expected)
    )
    task._humanoid_root_states[..., 7:10].zero_()
    EdgeContextTaskMixin._compute_relation_reward(task, selected)
    torch.testing.assert_close(task.rew_buf, torch.ones(2))


def test_checkpoint_collision_contract():
    expected = agent_collision_config(CPA["env"])
    check_agent_collision_checkpoint({"agent_collision_config": expected}, CPA["env"])
    check_agent_collision_checkpoint(
        {"relation_experiment_config": {"env": CPA["env"]}}, CPA["env"]
    )
    with pytest.raises(ValueError):
        check_agent_collision_checkpoint({}, CPA["env"])
    with pytest.raises(ValueError):
        check_agent_collision_checkpoint(
            {"agent_collision_config": {**expected, "distance": 0.8}}, CPA["env"]
        )

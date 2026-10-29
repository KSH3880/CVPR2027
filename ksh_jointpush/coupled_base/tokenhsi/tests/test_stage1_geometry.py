from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
import stage1_test_utils as reference
import test_stage1_policy as action
from utils.size_rsi import SIZE_RANGES, STAGE1_RANGES, sample_sizes, late_frames
from utils.stage1_geometry import max_bound_stack_height, pose_geometry_valid, sampled_phase
from utils.task_role_spec import task_size_probabilities, task_graph_packet
from utils.edge_scenario_spec import sample_size_conditioned_graph, classify_templates
from utils.edge_ontop_spec import permute_graph
from utils.relation_task_spec import (
    validate_relation_config,
    checkpoint_metadata,
    check_checkpoint_metadata,
)
from utils.unified_training import (
    validate_unified_env,
    validate_stage1_network,
    family_from_templates,
    append_family,
    sample_family_matched,
    amp_family_ids,
)
from utils.stage1_sampling import check_sampling_resume
from env.tasks.multi_agent.edge_interaction_reward import interaction_own_success
from env.tasks.multi_agent.edge_ontop_reward import inner_xy_region_error

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "data/cfg/multi_agent/stage1.yaml").read_text())["env"]
TRAIN = yaml.safe_load((ROOT / "data/cfg/train/rlg/stage1.yaml").read_text())




def test_sizes_priors_and_shuffled_task_amp_rows():
    original = deepcopy(SIZE_RANGES)
    torch.manual_seed(123)
    sizes = sample_sizes(16000, "cpu", ranges=STAGE1_RANGES)
    probabilities = task_size_probabilities(sizes[:, :2], STAGE1_RANGES)
    torch.testing.assert_close(
        probabilities.mean((0, 1)),
        torch.tensor([0.0, 0.1, 0.25, 0.325, 0.325]),
        atol=0.008,
        rtol=0,
    )
    assert ((sizes[:, 2:, :2] >= 0.4) & (sizes[:, 2:, :2] <= 0.8)).all()
    assert ((sizes[:, 2:, 2] >= 0.25) & (sizes[:, 2:, 2] <= 0.5)).all()
    high = sizes[:, :2, 2] > 0.60001
    assert high.any() and (probabilities[high][:, 2] == 1).all()
    assert SIZE_RANGES == original
    old = sample_sizes(128, "cpu", "climb")
    assert old[:, :2, 2].max() <= 0.5
    graph = sample_size_conditioned_graph(
        probabilities[:300], CFG["relationGraph"], "random_scenario"
    )
    shuffled = permute_graph(graph, torch.rand(300, 4).argsort(-1))
    assert torch.equal(task_graph_packet(graph, 300), task_graph_packet(shuffled, 300))
    env = torch.arange(300).repeat_interleave(2)
    agent = torch.arange(2).repeat(300)
    templates = classify_templates(shuffled, env, agent, True)
    assert (probabilities[env, agent, templates] > 0).all()
    families = family_from_templates(templates)
    pool = append_family(
        families[:, None, None].expand(-1, 10, 4).float(),
        families[:, None].expand(-1, 10),
    ).flatten(1)
    order = torch.randperm(len(pool))
    reference_rows = pool[order]
    sampled = sample_family_matched(pool, reference_rows, 10)
    assert torch.equal(amp_family_ids(sampled, 10), families[order])
    assert torch.equal(sampled[:, 0], families[order].float())


def test_only_bound_carry_stacks_contribute_to_height_guard():
    sizes = torch.tensor(
        [
            [[0.7, 0.7, 0.7], [0.6, 0.6, 0.7], [0.4, 0.8, 0.5], [0.8, 0.4, 0.5]],
            [[0.6, 0.6, 0.6], [0.4, 0.4, 0.6], [0.4, 0.8, 0.5], [0.8, 0.4, 0.5]],
        ]
    )
    torch.testing.assert_close(max_bound_stack_height(sizes), torch.tensor([0.7, 1.1]))


def test_at_distance_only_changes_at_and_preserves_one_mm_height():
    distance = torch.tensor([0.1999, 0.2001, 0.1, 0.1, 0.1999])
    phi = torch.exp(-10 * distance.square())[:, None]
    z = torch.tensor([0.0, 0.0, 0.0009, 0.0011, 0.0])[:, None]
    graph = SimpleNamespace(
        edge_relation=torch.tensor([[7], [7], [7], [7], [6]]),
        edge_valid=torch.ones(5, 1, dtype=torch.bool),
    )
    result = interaction_own_success(
        phi, z, torch.zeros_like(z), graph, CFG["relationReward"], torch.zeros_like(z)
    )
    assert result.flatten().tolist() == [True, False, True, False, False]


def test_shared_plane_inset_and_height_conditions():
    box = torch.tensor([[0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]]).expand(3, -1)
    size = torch.tensor([[0.5, 0.5, 0.5]]).expand(3, -1)
    region = inner_xy_region_error(
        torch.tensor([[0.199, 0.0], [0.201, 0.0], [0.0, 0.0]]), box, size, 0.1
    )[:, None]
    for relation in [9, 10, 8]:
        graph = SimpleNamespace(
            edge_relation=torch.full((3, 1), relation),
            edge_valid=torch.ones(3, 1, dtype=torch.bool),
        )
        result = interaction_own_success(
            torch.ones(3, 1),
            torch.zeros(3, 1),
            torch.zeros(3, 1),
            graph,
            CFG["relationReward"],
            region,
        )
        assert result.flatten().tolist() == [True, False, True]
    graph = SimpleNamespace(
        edge_relation=torch.tensor([[10], [10], [10]]),
        edge_valid=torch.ones(3, 1, dtype=torch.bool),
    )
    result = interaction_own_success(
        torch.ones(3, 1),
        torch.tensor([[0.201], [0.1], [0.1]]),
        torch.tensor([[0.0], [0.071], [0.069]]),
        graph,
        CFG["relationReward"],
        torch.zeros(3, 1),
    )
    assert result.flatten().tolist() == [False, False, True]


def test_rsi_geometry_and_phase_follow_actual_height_without_retargeting():
    body = torch.tensor(
        [[[0.0, 0.0, 0.4], [1.0, 0.0, 0.04]], [[1.0, 0.0, 0.4], [1.0, 0.0, 0.04]]]
    )
    feet = torch.tensor([[[1.0, 0.0, 0.04], [1.0, 0.0, 0.04]]]).expand(2, -1, -1)
    boxes = torch.zeros(2, 1, 13)
    boxes[:, :, 2] = 0.35
    boxes[:, :, 6] = 1
    sizes = torch.full((2, 1, 3), 0.7)
    assert pose_geometry_valid(body, feet, boxes, sizes).tolist() == [False, True]
    roots = torch.tensor([[0.0, 0.0, 0.55], [0.0, 0.0, 1.3]])
    ref_feet = torch.tensor(
        [[[0.0, 0.0, 0.04], [0.0, 0.0, 0.04]], [[0.0, 0.0, 0.4], [0.0, 0.0, 0.4]]]
    )
    sizes = torch.tensor([[0.6, 0.6, 0.4], [0.6, 0.6, 0.7]])
    assert sampled_phase("sit", roots, ref_feet, torch.zeros(2), sizes)[0].tolist() == [
        True,
        False,
    ]
    phase = sampled_phase("climb", roots, ref_feet, torch.zeros(2), sizes)
    assert phase[0].tolist() == [False, True] and (not phase[1].any())
    assert len(late_frames(torch.ones(2, dtype=torch.bool), phase[1])) == 0


@pytest.mark.parametrize("branch", ["actor", "critic"])
def test_stage1_action_task_and_token_shuffle_outputs_and_gradients(monkeypatch, branch):
    monkeypatch.setattr(reference, "CFG", CFG)
    monkeypatch.setattr(reference, "TRAIN", TRAIN)
    action.test_token_edge_task_shuffle_outputs_and_all_gradients(branch)

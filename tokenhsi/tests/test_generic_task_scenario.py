"""Variable-count inference uses valid, disjoint tasks and the trained packet schema."""

from pathlib import Path
import pytest
import torch
import yaml
from utils.edge_scenario_spec import (
    compile_graph,
    compose_canonical_graph,
    sample_size_conditioned_graph,
    validate_graph,
)
from utils.edge_stage1_spec import max_stage1_stack_height
from utils.edge_ontop_spec import copy_graph_rows, expand_graph
from utils.size_rsi import sample_sizes
from utils.task_role_spec import task_graph_packet, task_size_probabilities

ROOT = Path(__file__).resolve().parents[1]
SPEC = yaml.safe_load((ROOT / "data/cfg/multi_agent/stage1_v2_fixed.yaml").read_text())[
    "env"
]["relationGraph"]


@pytest.mark.parametrize(
    "m,o",
    [
        (1, 1),
        (1, 5),
        (2, 2),
        (2, 4),
        (3, 3),
        (3, 4),
        (4, 16),
        (9, 14),
        (12, 12),
        (23, 29),
        (32, 64),
    ],
)
def test_generic_counts_binding_and_size_eligibility(m, o):
    prototype = compile_graph(SPEC, m, o)
    assert (prototype.num_agents, prototype.num_objects, len(prototype.ids)) == (
        m,
        o,
        2 * m,
    )
    validate_graph(prototype)
    sizes = sample_sizes(16, "cpu", num_agents=m, num_objects=o)
    assert sizes.shape == (16, o, 3)
    probabilities = task_size_probabilities(sizes[:, :m])
    graph = sample_size_conditioned_graph(probabilities, SPEC, "random_scenario", o)
    validate_graph(graph)
    packet = task_graph_packet(graph, 16).reshape(16, m, 5).long()
    assert (packet[..., 0] == 1).all()
    owners = torch.arange(m).expand(16, -1)
    assert torch.equal(packet[..., 2], owners)
    task, payload, target = (packet[..., 1], packet[..., 3], packet[..., 4])
    assert ((task >= 0) & (task < 4)).all()
    assert (probabilities.gather(-1, (task + 1)[..., None]) > 0).all()
    primary = torch.where(task >= 2, payload, target)
    assert torch.equal(primary, m + owners)
    assert (target[task == 2] == (m + o + owners)[task == 2]).all()
    assert ((target[task == 3] >= 2 * m) & (target[task == 3] < m + o)).all()
    assert ((task == 3).sum(-1) <= o - m).all()
    for row in range(16):
        supports = target[row][task[row] == 3]
        assert len(supports.unique()) == len(supports)


@pytest.mark.parametrize("supports", [0, 1, 5])
def test_support_shortage_redraws_without_standalone_holding(supports):
    m = 9
    probabilities = torch.tensor([0.0, 0.1, 0.1, 0.1, 0.7]).expand(256, m, -1)
    graph = sample_size_conditioned_graph(
        probabilities, SPEC, "random_scenario", m + supports
    )
    packet = task_graph_packet(graph, 256).reshape(256, m, 5)
    assert ((packet[..., 1] == 3).sum(-1) <= supports).all()
    assert (packet[..., 0] == 1).all()
    assert (packet[..., 1] < 3).any()


def test_full_supports_preserve_canonical_owner_binding():
    graph = compose_canonical_graph(torch.full((1, 7), 4), num_objects=15)
    packet = task_graph_packet(graph, 1).reshape(1, 7, 5)
    assert torch.equal(packet[0, :, 4], torch.arange(14, 21).float())


def test_partial_reset_keeps_other_scenes_and_variable_packet_width():
    graph = expand_graph(compile_graph(SPEC, 9, 14), 5)
    before = task_graph_packet(graph, 5).clone()
    replacement = sample_size_conditioned_graph(
        torch.ones(2, 9, 5), SPEC, "sit_climb", 14
    )
    copy_graph_rows(graph, torch.tensor([1, 4]), replacement)
    after = task_graph_packet(graph, 5)
    assert after.shape == (5, 45)
    assert torch.equal(after[[0, 2, 3]], before[[0, 2, 3]])
    assert torch.equal(after[[1, 4]], task_graph_packet(replacement, 2))


def test_generic_sit_climb_and_single_object_height():
    probabilities = torch.ones(4, 5, 5)
    graph = sample_size_conditioned_graph(probabilities, SPEC, "sit_climb", 5)
    packet = task_graph_packet(graph, 4).reshape(4, 5, 5)
    assert torch.equal(packet[..., 1], (torch.arange(5) % 2).float().expand(4, -1))
    assert max_stage1_stack_height(torch.tensor([[[0.4, 0.4, 0.5]]])).item() == 0.5


@pytest.mark.parametrize("m,o", [(0, 0), (3, 2)])
def test_invalid_counts_fail_before_simulation(m, o):
    with pytest.raises(ValueError):
        compile_graph(SPEC, m, o)
    with pytest.raises(ValueError):
        sample_sizes(1, "cpu", num_agents=m, num_objects=o)

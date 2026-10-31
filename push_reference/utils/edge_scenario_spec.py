"""Canonical independent Stage1 scenarios and variable-count evaluation."""

import math
import torch
from push_reference.utils.edge_context_spec import EdgeContextGraph, HOLDING, AT
from push_reference.utils.edge_ontop_spec import (
    ON_TOP,
    permute_graph,
    select_graph,
    validate_graph as validate_base_graph,
)
from push_reference.utils.edge_interaction_spec import SIT, CLIMB

UNIFIED_SAMPLER = "two_agent_four_object_independent_unified"
CLIMB_TEMPLATES = ("HOLDING", "SIT", "CLIMB", "HOLDING_AT", "HOLDING_ON_TOP")
PRESETS = (
    "random_scenario",
    "holding",
    "sit",
    "climb",
    "holding_at",
    "holding_ontop",
    "sit_climb",
)


def scenario_templates(spec):
    if spec.get('sampler') in ('stage1_jointpush', 'stage2_jointpush'):
        return CLIMB_TEMPLATES + ('PUSH_AT',)
    return CLIMB_TEMPLATES


def validate_sampler(spec, m=2, o=4):
    expected = dict(
        mode="edge_composition",
        sampler=UNIFIED_SAMPLER,
        semantic_only=True,
        policy_task_roles=True,
        edge_capacity=4,
        max_edges_per_agent=2,
        random_binding={"objects": "canonical", "goals": "canonical"},
        shuffle_token_order=True,
    )
    if set(spec) != set(expected) | {"template_probabilities", "shuffle_edge_order"}:
        raise ValueError("Unsupported Stage1 scenario fields")
    if (
        any((spec[key] != val for key, val in expected.items()))
        or type(spec["shuffle_edge_order"]) is not bool
    ):
        raise ValueError("Unsupported Stage1 sampler contract")
    if m < 1 or o < m:
        raise ValueError("Stage1 requires M>=1 and O>=M")
    probs = spec["template_probabilities"]
    if (
        tuple(probs) != CLIMB_TEMPLATES
        or probs["HOLDING"] != 0
        or any(
            (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or (not math.isfinite(v))
                or (v < 0)
                for v in probs.values()
            )
        )
        or (not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-08))
    ):
        raise ValueError("Stage1 template probabilities must follow the task contract")


def compose_canonical_graph(template_ids, shuffle=False, num_objects=None):
    n, m = template_ids.shape
    o = 2 * m if num_objects is None else num_objects
    if m < 1 or o < m:
        raise ValueError("Independent tasks require at least one object per human")
    device = template_ids.device
    owners = torch.arange(m, device=device)[None].expand(n, -1)
    ontop = template_ids == 4
    if (ontop.sum(-1) > o - m).any():
        raise ValueError(
            "ON_TOP tasks require one distinct spare support object per human"
        )
    support = owners + m if o >= 2 * m else m + ontop.long().cumsum(-1) - 1
    placement = template_ids >= 3
    src = torch.stack((owners, owners + m), -1).flatten(1)
    dst = torch.stack(
        (owners + m, torch.where(template_ids == 3, owners + m + o, support + m)), -1
    ).flatten(1)
    base_relations = torch.tensor(
        [HOLDING, SIT, CLIMB, HOLDING, HOLDING], device=device
    )
    relation = torch.stack(
        (base_relations[template_ids], torch.where(template_ids == 3, AT, ON_TOP)), -1
    ).flatten(1)
    owner = owners[:, :, None].expand(-1, -1, 2).flatten(1)
    valid = torch.stack((torch.ones_like(placement), placement), -1).flatten(1)
    required = torch.stack((~placement, placement), -1).flatten(1)
    order = valid.long().argsort(dim=-1, descending=True, stable=True)
    values = [
        value.masked_fill(~valid, 0).gather(1, order)
        for value in (src, dst, relation, owner, valid, required)
    ]
    e = 2 * m
    graph = EdgeContextGraph(
        tuple(("slot{}".format(i) for i in range(e))),
        m,
        o,
        *values,
        torch.zeros(n, e, e, dtype=torch.bool, device=device),
        torch.full((n, e), -1, dtype=torch.long, device=device)
    )
    if shuffle:
        graph = permute_graph(graph, torch.rand(n, e, device=device).argsort(-1))
    return graph


def sample_size_conditioned_graph(probabilities, spec, preset, num_objects=None):
    from push_reference.utils.task_role_spec import task_preset

    preset = task_preset(preset)
    n, m = probabilities.shape[:2]
    o = 2 * m if num_objects is None else num_objects
    if m < 1 or o < m:
        raise ValueError("Independent tasks require at least one object per human")
    if preset == "random_scenario":
        template_ids = torch.multinomial(probabilities.flatten(0, 1), 1).reshape(n, m)
        if o < 2 * m:
            ontop = template_ids == 4
            priority = torch.rand(n, m, device=probabilities.device).masked_fill(
                ~ontop, 2.0
            )
            rank = priority.argsort(-1).argsort(-1)
            redraw = ontop & (rank >= o - m)
            if redraw.any():
                alternatives = probabilities[redraw].clone()
                alternatives[:, 4] = 0
                if (alternatives.sum(-1) <= 0).any():
                    raise ValueError(
                        "Not enough supports for the eligible ON_TOP-only tasks"
                    )
                template_ids[redraw] = torch.multinomial(alternatives, 1).flatten()
    elif preset == "sit_climb":
        template_ids = (1 + torch.arange(m, device=probabilities.device) % 2).expand(
            n, -1
        )
    else:
        template_ids = torch.full(
            (n, m),
            PRESETS.index(preset) - 1,
            dtype=torch.long,
            device=probabilities.device,
        )
    return compose_canonical_graph(
        template_ids,
        shuffle=spec["shuffle_edge_order"] and preset == "random_scenario",
        num_objects=o,
    )


def validate_graph(graph):
    validate_base_graph(graph)
    if graph.edge_src.ndim != 1:
        for i in range(graph.edge_src.shape[0]):
            validate_graph(select_graph(graph, i))
        return
    if graph.prereq_mask.any() or (graph.term_index >= 0).any():
        raise ValueError("Independent scenarios cannot contain temporal context")
    for agent in range(graph.num_agents):
        ids = (graph.edge_valid & (graph.edge_owner == agent)).nonzero().flatten()
        if not 1 <= len(ids) <= 2:
            raise ValueError("Each scenario owner requires one or two edges")
        rel = graph.edge_relation[ids]
        hold = ids[rel == HOLDING]
        placement = ids[(rel == AT) | (rel == ON_TOP)]
        interaction = ids[(rel == SIT) | (rel == CLIMB)]
        if len(placement):
            if len(hold) != 1 or len(placement) != 1 or len(interaction):
                raise ValueError(
                    "Placement template must contain HOLDING plus one placement"
                )
            if (
                graph.edge_dst[hold[0]] != graph.edge_src[placement[0]]
                or graph.required_goal[hold[0]]
                or (not graph.required_goal[placement[0]])
            ):
                raise ValueError("Placement pair owner/source/required-goal mismatch")
        elif len(ids) != 1 or not graph.required_goal[ids[0]]:
            raise ValueError("Standalone template must be its required current goal")


def classify_templates(graph, slot_env, slot_agent, with_climb=True):
    owned = graph.edge_valid[slot_env] & (
        graph.edge_owner[slot_env] == slot_agent[:, None]
    )
    rel = graph.edge_relation[slot_env]
    has = lambda value: ((rel == value) & owned).any(-1)
    result = torch.where(
        has(SIT),
        1,
        torch.where(
            has(CLIMB),
            2,
            torch.where(
                has(AT), 3, torch.where(has(ON_TOP), 4, torch.zeros_like(slot_agent))
            ),
        ),
    )
    return torch.where(has(12), 5, result)


def agent_object_indices(graph, slot_env, slot_agent):
    owned = graph.edge_valid[slot_env] & (
        graph.edge_owner[slot_env] == slot_agent[:, None]
    )
    rel = graph.edge_relation[slot_env]
    target = torch.where(
        (rel == HOLDING) | (rel == SIT) | (rel == CLIMB) | (rel == 11) | (rel == 12), graph.edge_dst[slot_env], 0
    )
    return target.masked_fill(~owned, 0).amax(-1) - graph.num_agents


def agent_goal_indices(graph, slot_env, slot_agent):
    owned = graph.edge_valid[slot_env] & (
        graph.edge_owner[slot_env] == slot_agent[:, None]
    )
    target = torch.where(
        (graph.edge_relation[slot_env] == AT) & owned,
        graph.edge_dst[slot_env],
        graph.num_agents + graph.num_objects,
    )
    return target.amax(-1) - graph.num_agents - graph.num_objects


def paired_placement_success(success, graph):
    valid = graph.edge_valid
    unbatched = valid.ndim == 1
    if unbatched:
        valid = valid.unsqueeze(0)
        relation = graph.edge_relation.unsqueeze(0)
        owner = graph.edge_owner.unsqueeze(0)
        src = graph.edge_src.unsqueeze(0)
        dst = graph.edge_dst.unsqueeze(0)
        success = success.unsqueeze(0) if success.ndim == 1 else success
    else:
        relation, owner, src, dst = (
            graph.edge_relation,
            graph.edge_owner,
            graph.edge_src,
            graph.edge_dst,
        )
    holding = valid & ((relation == HOLDING) | (relation == 12))
    placement = valid & ((relation == AT) | (relation == ON_TOP))
    paired = (
        holding[:, :, None]
        & placement[:, None, :]
        & (owner[:, :, None] == owner[:, None, :])
        & (dst[:, :, None] == src[:, None, :])
    )
    result = (paired & success[:, None, :]).any(-1)
    return result[0] if unbatched else result


def sample_graph(
    n, spec, device="cpu", preset="random_scenario", role_swap=False, generator=None
):
    validate_sampler(spec)
    from push_reference.utils.task_role_spec import task_preset

    preset = task_preset(preset)
    if preset not in PRESETS:
        raise ValueError("Unknown Stage1 TASK_GRAPH preset: " + preset)
    probs = torch.tensor(list(spec["template_probabilities"].values()), device=device)
    if preset == "random_scenario":
        template_ids = torch.multinomial(
            probs, 2 * n, replacement=True, generator=generator
        ).reshape(n, 2)
    elif preset == "sit_climb":
        template_ids = torch.tensor([1, 2], device=device).expand(n, -1)
    else:
        template_ids = torch.full(
            (n, 2), PRESETS.index(preset) - 1, dtype=torch.long, device=device
        )
    return compose_canonical_graph(
        template_ids, shuffle=spec["shuffle_edge_order"] and preset == "random_scenario"
    )


def compile_graph(spec, m, o, device=None):
    validate_sampler(spec, m, o)
    return select_graph(
        compose_canonical_graph(
            torch.full((1, m), 3, dtype=torch.long, device=device or "cpu"),
            num_objects=o,
        ),
        0,
    )

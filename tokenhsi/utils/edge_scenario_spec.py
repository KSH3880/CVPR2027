import torch


from utils.edge_context_spec import HOLDING, AT


from utils.edge_ontop_spec import ON_TOP


from utils.edge_interaction_spec import SIT, CLIMB


CLIMB_TEMPLATES = ("HOLDING", "SIT", "CLIMB", "HOLDING_AT", "HOLDING_ON_TOP")


def scenario_templates(spec):
    return CLIMB_TEMPLATES


def classify_templates(graph, slot_env, slot_agent, with_climb=False):
    owned = graph.edge_valid[slot_env] & (
        graph.edge_owner[slot_env] == slot_agent[:, None]
    )
    rel = graph.edge_relation[slot_env]
    has = lambda value: ((rel == value) & owned).any(-1)
    if not with_climb and has(CLIMB).any():
        raise ValueError("CLIMB graph rejected by no-CLIMB RSI")
    if with_climb:
        return torch.where(
            has(SIT),
            1,
            torch.where(
                has(CLIMB),
                2,
                torch.where(
                    has(AT),
                    3,
                    torch.where(has(ON_TOP), 4, torch.zeros_like(slot_agent)),
                ),
            ),
        )
    return torch.where(
        has(SIT),
        1,
        torch.where(
            has(AT), 2, torch.where(has(ON_TOP), 3, torch.zeros_like(slot_agent))
        ),
    )


def agent_object_indices(graph, slot_env, slot_agent):
    owned = graph.edge_valid[slot_env] & (
        graph.edge_owner[slot_env] == slot_agent[:, None]
    )
    rel = graph.edge_relation[slot_env]
    target = torch.where(
        (rel == HOLDING) | (rel == SIT) | (rel == CLIMB), graph.edge_dst[slot_env], 0
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
    holding = valid & (relation == HOLDING)
    placement = valid & ((relation == AT) | (relation == ON_TOP))
    paired = (
        holding[:, :, None]
        & placement[:, None, :]
        & (owner[:, :, None] == owner[:, None, :])
        & (dst[:, :, None] == src[:, None, :])
    )
    result = (paired & success[:, None, :]).any(-1)
    return result[0] if unbatched else result

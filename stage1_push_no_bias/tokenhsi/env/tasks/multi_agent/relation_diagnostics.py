"""TensorBoard naming for Stage1 graph, sampling and penalty metrics."""

from utils.edge_context_spec import HOLDING, AT
from utils.edge_interaction_spec import SIT, CLIMB
from utils.edge_ontop_spec import ON_TOP, batched
from utils.stage1_jointpush import PUSH


def edge_metric_masks(graph, num_envs, include_push=False):
    """Split PUSH contact and its owner's AT from carrying AT for logging only."""
    relation = batched(graph.edge_relation, num_envs)
    valid = batched(graph.edge_valid, num_envs)
    masks = {
        name: valid & (relation == kind)
        for kind, name in ((HOLDING, "holding"), (AT, "at"),
                           (ON_TOP, "ontop"), (SIT, "sit"), (CLIMB, "climb"))
    }
    if include_push:
        contact = valid & (relation == PUSH)
        owner = batched(graph.edge_owner, num_envs)
        push_owners = owner.new_zeros((num_envs, graph.num_agents))
        push_owners.scatter_add_(1, owner, contact.long())
        push_edges = push_owners.gather(1, owner) > 0
        masks["contact"] = contact
        masks["push_at"] = masks["at"] & push_edges
        masks["at"] = masks["at"] & ~push_edges
    return masks


def relation_tensorboard_tag(key):
    return "relation/" + key

"""Owner and required-goal reductions shared by Stage1 rewards and evaluation."""
from utils.edge_ontop_spec import batched

def owner_sum(values, graph):
    return values.new_zeros(values.shape[0], graph.num_agents).scatter_add(1, batched(graph.edge_owner, values.shape[0]), values)

def goal_success(success, graph):
    mask = batched(graph.required_goal & graph.edge_valid, success.shape[0])
    failed = owner_sum((~success & mask).float(), graph)
    count = owner_sum(mask.expand_as(success).float(), graph)
    return (failed == 0) & (count > 0)

def scene_success(success, graph):
    mask = batched(graph.required_goal & graph.edge_valid, success.shape[0])
    return (success | ~mask).all(-1) & mask.any(-1)

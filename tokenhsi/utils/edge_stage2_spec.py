import torch
from utils.edge_context_spec import EdgeContextGraph

STAGE2_CONTEXT_MODE = "state_relation_edge_stage2_v1"


def _empty_graph(n, m, o, capacity, device):
    longs = [
        torch.zeros(n, capacity, dtype=torch.long, device=device) for _ in range(4)
    ]
    valid = torch.zeros(n, capacity, dtype=torch.bool, device=device)
    required = valid.clone()
    pre = torch.zeros(n, capacity, capacity, dtype=torch.bool, device=device)
    term = torch.full((n, capacity), -1, dtype=torch.long, device=device)
    return EdgeContextGraph(
        tuple("slot{}".format(i) for i in range(capacity)),
        m,
        o,
        *longs,
        valid,
        required,
        pre,
        term,
    )

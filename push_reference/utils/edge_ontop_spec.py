"""Batch selection, permutation and topology validation for task graphs."""

from dataclasses import fields, replace
import torch
from push_reference.utils.edge_context_spec import HOLDING, AT

ON_TOP = 8


def batched(value, n):
    return value.unsqueeze(0).expand(n, *value.shape) if value.ndim == 1 else value


def select_graph(g, ids):
    if g.edge_src.ndim == 1:
        return g
    return replace(
        g,
        **{
            f.name: getattr(g, f.name)[ids]
            for f in fields(g)
            if isinstance(getattr(g, f.name), torch.Tensor)
        }
    )


def expand_graph(g, n):
    return replace(
        g,
        **{
            f.name: getattr(g, f.name)
            .unsqueeze(0)
            .expand(n, *getattr(g, f.name).shape)
            .clone()
            for f in fields(g)
            if isinstance(getattr(g, f.name), torch.Tensor)
        }
    )


def copy_graph_rows(dst, ids, src):
    for f in fields(dst):
        v = getattr(dst, f.name)
        if isinstance(v, torch.Tensor):
            v[ids] = getattr(src, f.name)


def permute_graph(g, permutation):
    """Per-scene edge permutation, including both PRE axes and TERM references."""
    n, e = permutation.shape
    inv = permutation.argsort(-1)
    data = {}
    for f in fields(g):
        v = getattr(g, f.name)
        if not isinstance(v, torch.Tensor):
            continue
        if f.name == "prereq_mask":
            data[f.name] = v.gather(1, permutation[..., None].expand(n, e, e)).gather(
                2, permutation[:, None].expand(n, e, e)
            )
        elif f.name == "term_index":
            old = v.gather(1, permutation)
            data[f.name] = torch.where(old >= 0, inv.gather(1, old.clamp_min(0)), -1)
        else:
            data[f.name] = v.gather(1, permutation)
    return replace(g, **data)


def validate_graph(g):
    """Validate explicit topology; PRE and support DAGs separately (TERM may point back)."""
    if g.edge_src.ndim != 1:
        for i in range(g.edge_src.shape[0]):
            validate_graph(select_graph(g, i))
        return
    active = g.edge_valid.nonzero(as_tuple=False).flatten().tolist()

    def acyclic(adjacency):
        visiting = set()
        done = set()

        def visit(node):
            if node in visiting:
                raise ValueError("Cyclic support or PRE graph")
            if node in done:
                return
            visiting.add(node)
            for child in adjacency.get(node, []):
                visit(child)
            visiting.remove(node)
            done.add(node)

        for node in adjacency:
            visit(node)

    acyclic(
        {i: g.prereq_mask[i].nonzero(as_tuple=False).flatten().tolist() for i in active}
    )
    support = {}
    children = set()
    placements = {
        int(g.edge_src[i]): i for i in active if int(g.edge_relation[i]) in (AT, ON_TOP)
    }
    held = {int(g.edge_dst[i]) for i in active if int(g.edge_relation[i]) == HOLDING}
    for i in active:
        if int(g.edge_relation[i]) != ON_TOP:
            continue
        s, t = (int(g.edge_src[i]), int(g.edge_dst[i]))
        if s == t or t in children:
            raise ValueError("Self support or multiple children on one support")
        if t in held and t not in placements:
            raise ValueError("HOLDING-only object is not a free support")
        children.add(t)
        support.setdefault(s, []).append(t)
    acyclic(support)

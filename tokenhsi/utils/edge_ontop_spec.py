from dataclasses import fields, replace


import torch


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
        },
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
        },
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

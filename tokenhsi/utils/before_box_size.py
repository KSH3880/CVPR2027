import torch

from utils.before_push_spec import sample_graph


def sample_intersection(sizes, ranges, probabilities, preset, num_objects,
                        pairing_probability, mode):
    graph, predecessor, templates = sample_graph(sizes, ranges, probabilities,
        preset, num_objects, pairing_probability=pairing_probability)
    at = ranges['HOLDING_AT']
    bounds = {kind: [[max(a[0], b[0]), min(a[1], b[1])] for a, b in zip(at, ranges[name])]
              for kind, name in ((1, 'SIT'), (2, 'CLIMB'), (4, 'SUPPORT'))}
    if mode == 'common':
        common = [[max(bounds[k][axis][0] for k in bounds),
                   min(bounds[k][axis][1] for k in bounds)] for axis in range(3)]
        bounds = {kind: common for kind in bounds}
    selected = sizes.clone()
    uniform = torch.rand(sizes.shape, device=sizes.device)
    rows, followers = (predecessor >= 0).nonzero(as_tuple=True)
    leaders = predecessor[rows, followers]
    for kind, interval in bounds.items():
        chosen = templates[rows, followers] == kind
        envs, owners = rows[chosen], leaders[chosen]
        ticks = (sizes.new_tensor(interval) * 20).round().long()
        samples = ticks[:, 0] + (uniform[envs, owners] * (ticks[:, 1] - ticks[:, 0] + 1)).floor().long()
        selected[envs, owners] = samples.float() / 20
    return selected, graph, predecessor, templates

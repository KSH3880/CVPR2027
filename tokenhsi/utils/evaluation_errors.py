import statistics

import torch

from utils.evaluation_protocol import TASK_NAMES, _spatial_distances

ERROR_PROTOCOL = 'region_success_min_distance_v1'


def spatial_errors(packet, roots, boxes, sizes, goals):
    point, xy, z, point_task, valid = _spatial_distances(packet, roots, boxes, sizes, goals)
    region = torch.where(point_task, point, torch.stack((xy, z), -1).norm(dim=-1))
    return torch.stack((point, region), -1).masked_fill(~valid[..., None], float('inf'))


def update_error_minima(minimum, packet, roots, boxes, sizes, goals, alive,
                        center_minimum=None, centers=None, center_goals=None, groups=None):
    distances = spatial_errors(packet, roots, boxes, sizes, goals)
    minimum.copy_(torch.minimum(minimum, distances.masked_fill(~alive[..., None], float('inf'))))
    if center_minimum is not None:
        group_alive = ((groups[:, :, None] == groups[:, None, :]) & alive[:, None, :]).any(-1)
        distance = (centers.double() - center_goals.double()).norm(dim=-1)
        center_minimum.copy_(torch.minimum(center_minimum, distance.masked_fill(~group_alive, float('inf'))))


def error_scores(packet, parent, groups, ever, simultaneous, minimum,
                 center_ever=None, center_minimum=None):
    valid = packet[..., 0].bool()
    followers = parent >= 0
    rows, people = followers.nonzero(as_tuple=True)
    enabling = followers.clone()
    enabling[rows, parent[rows, people]] = True
    members = (groups[:, :, None] == groups[:, None, :]).sum(-1)
    coupled = (members > 1) & ~enabling
    leader = groups == torch.arange(groups.shape[1], device=groups.device)[None]
    success = ever[..., 1].clone() & valid
    pair_success = simultaneous[rows, people, 1]
    success[rows, people] = pair_success
    success[rows, parent[rows, people]] = pair_success

    def summarize(values):
        count = len(values)
        means = (values.mean(0) * 100).tolist() if count else [None, None]
        return dict(successes=count, point_error_cm=means[0], region_error_cm=means[1])

    per_task = {}
    for kind, name in enumerate(TASK_NAMES):
        selected = success & ~coupled & (packet[..., 1] == kind)
        values = minimum[selected]
        if name == 'PUSH' and center_minimum is not None:
            selected = valid & coupled & leader & center_ever
            values = torch.cat((values, center_minimum[selected][:, None].expand(-1, 2)))
        per_task[name] = summarize(values)
    by_size = {}
    if center_minimum is not None:
        for size in range(2, 5):
            selected = valid & coupled & leader & center_ever & (members == size)
            by_size[str(size)] = summarize(center_minimum[selected][:, None].expand(-1, 2))
    return dict(protocol=ERROR_PROTOCOL, unit='cm', success_criterion='region',
                per_task=per_task, coupled_by_size=by_size)


def summarize_errors(runs):
    versions = {r.get('errors', {}).get('protocol') for r in runs}
    if versions != {ERROR_PROTOCOL}:
        raise ValueError('Cannot summarize missing or different error protocols together')
    seeds = sorted({r['seed'] for r in runs})
    result = dict(protocol=ERROR_PROTOCOL, unit='cm', success_criterion='region', seeds=seeds)
    for section in ('per_task', 'coupled_by_size'):
        result[section] = {}
        names = sorted({name for run in runs for name in run['errors'][section]})
        for name in names:
            rows = [r['errors'][section].get(name) for r in runs]
            item = dict(successes=sum(row['successes'] for row in rows if row))
            for metric in ('point_error_cm', 'region_error_cm'):
                values = []
                for seed in seeds:
                    selected = [r['errors'][section][name] for r in runs
                                if r['seed'] == seed and name in r['errors'][section]]
                    count = sum(row['successes'] for row in selected)
                    values.append(sum(row[metric] * row['successes'] for row in selected
                                      if row['successes']) / count if count else None)
                complete = None not in values
                item[metric] = dict(mean=statistics.mean(values) if complete else None,
                    std=statistics.stdev(values) if complete and len(values) > 1 else None, seeds=values)
            result[section][name] = item
    return result

import itertools

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from utils.coupled_push_spec import inverse_rotate


MODES = ('random', 'ordered', 'distance', 'optimal')


def validate_mode(mode, task, evaluation, protocol='', resume=0):
    if mode not in MODES:
        raise ValueError('Unknown coupled assignment: '+mode)
    if mode != 'random' and (task != 'HumanoidMACoupledPush' or not evaluation or protocol or resume):
        raise ValueError('--coupled_assignment requires coupled viewer/test inference without eval_protocol or resume')


def crossing_counts(roots, targets, assignments):
    first, second = np.triu_indices(len(roots), 1)
    a, c = roots[first], roots[second]
    b, d = targets[assignments[:, first]], targets[assignments[:, second]]

    def cross(x, y):
        return x[..., 0]*y[..., 1]-x[..., 1]*y[..., 0]

    ab, cd = b-a, d-c
    o1, o2 = cross(ab, c-a), cross(ab, d-a)
    o3, o4 = cross(cd, a-c), cross(cd, b-c)
    tolerance = 1e-10
    opposite1 = ((o1 <= tolerance) & (o2 >= -tolerance)) | ((o2 <= tolerance) & (o1 >= -tolerance))
    opposite2 = ((o3 <= tolerance) & (o4 >= -tolerance)) | ((o4 <= tolerance) & (o3 >= -tolerance))
    overlap = (np.maximum(np.minimum(a, b), np.minimum(c, d)) <=
               np.minimum(np.maximum(a, b), np.maximum(c, d))+tolerance).all(-1)
    return (opposite1 & opposite2 & overlap).sum(-1)


def scores(roots, targets, distances, assignments):
    travel = distances[np.arange(len(roots)), assignments]
    return np.column_stack((crossing_counts(roots, targets, assignments),
                            travel.max(-1), travel.sum(-1)))


def best_assignment(roots, targets, mode):
    count = len(roots)
    ordered = np.argsort(roots[:, 0], kind='stable').argsort(kind='stable')
    if mode == 'ordered':
        return ordered
    distances = np.linalg.norm(roots[:, None]-targets[None], axis=-1)
    rows, columns = linear_sum_assignment(distances)
    nearest = np.empty(count, dtype=np.int64)
    nearest[rows] = columns
    if mode == 'distance':
        return nearest

    def best(candidates):
        values = scores(roots, targets, distances, candidates)
        index = np.lexsort((values[:, 2], values[:, 1], values[:, 0]))[0]
        return candidates[index].copy(), tuple(values[index])

    if count <= 8:
        permutations = itertools.permutations(range(count))
        winner, value = None, (np.inf, np.inf, np.inf)
        while True:
            batch = list(itertools.islice(permutations, 2048))
            if not batch:
                return winner
            candidate, score = best(np.asarray(batch))
            if score < value:
                winner, value = candidate, score

    winner, value = best(np.stack((ordered, nearest)))
    # Larger groups use deterministic best-improvement pair swaps.
    for _ in range(2*count):
        candidate, score = winner, value
        pairs = iter(itertools.combinations(range(count), 2))
        while True:
            batch = list(itertools.islice(pairs, 256))
            if not batch:
                break
            alternatives = np.broadcast_to(winner, (len(batch), count)).copy()
            for row, (i, j) in enumerate(batch):
                alternatives[row, i], alternatives[row, j] = winner[j], winner[i]
            swapped, swapped_score = best(alternatives)
            if swapped_score < score:
                candidate, score = swapped, swapped_score
        if score >= value:
            break
        winner, value = candidate, score
    return winner


def assign_regions(groups, roots, boxes, sizes, regions, mode, clearance):
    if mode == 'random':
        return regions
    local = inverse_rotate(boxes[..., 3:7], roots[..., :3]-boxes[..., :3])
    local, dimensions = local[..., :2].double().cpu().numpy(), sizes[..., :2].double().cpu().numpy()
    labels = groups.cpu().numpy()
    result = np.zeros(labels.shape, dtype=np.int64)
    for row, scene in enumerate(labels):
        for label in np.unique(scene):
            people = np.flatnonzero(scene == label)
            if len(people) == 1:
                continue
            width, depth = dimensions[row, people[0]]
            targets = np.column_stack(((np.arange(len(people))+.5)*width/len(people)-width/2,
                                       np.full(len(people), -depth/2-clearance)))
            result[row, people] = best_assignment(local[row, people], targets, mode)
    return torch.as_tensor(result, device=groups.device, dtype=regions.dtype)

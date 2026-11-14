import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

import reference_metrics


def verify(metrics):
    directory = Path(__file__).parent
    source = (directory/'teamhoi_reference.py').read_text()
    expected = json.loads((directory/'source.json').read_text())['functions']
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in expected:
            actual = hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
            if actual != expected[node.name]['function_sha256']:
                raise ValueError('Preserved TeamHOI function changed: '+node.name)
    cases = json.loads((Path(__file__).parent/'cases.json').read_text())
    count = 0
    box = [0, 0, 0, 0, 0, 0, 1]
    for case in cases['region']:
        actual = metrics.region_distances(case['point'], case['target'], box, case['size'])
        if actual['success'] != case['success']:
            raise ValueError(case['name'])
        for name in ('point_error', 'region_error'):
            np.testing.assert_allclose(actual[name], case[name], atol=1e-12, rtol=1e-12)
        count += 1
    for case in cases['enabling']:
        if metrics.first_enabling(case['success'], case['valid']) != case['first']:
            raise ValueError(case['name'])
        count += 1
    for case in cases['counter']:
        centers = np.zeros((len(case['x']), 3))
        centers[:, 0] = case['x']
        actual = metrics.start_counters(centers, [case['initial_x'], 0, 0], case['first_success'])
        if list(actual) != case['expected']:
            raise ValueError(case['name'])
        count += 1
    for case in cases['ratio']:
        actual = metrics.cooperative_ratio(case['count'], case['a'], case['b'])
        if case['expected'] == 'NaN':
            if not np.isnan(actual):
                raise ValueError(case['name'])
        else:
            np.testing.assert_equal(actual, case['expected'])
        count += 1
    np.testing.assert_allclose(metrics.support_goal([0, 0, .5, 0, 0, 0, 1], [1, 1, 1], .1), [0, 0, 1.1])
    q = [np.sin(np.pi/4), 0, 0, np.cos(np.pi/4)]
    np.testing.assert_allclose(metrics.support_goal([0, 0, 0, *q], [1, 2, 4], .1), [0, 0, 1.1])
    rotated = metrics.region_distances([0, 1.5, 0], [0, 0, 0],
                                       [0, 0, 0, 0, 0, np.sin(np.pi/4), np.cos(np.pi/4)], [4, 1, 1])
    if not rotated['success']:
        raise ValueError('Yaw-local full XY region')
    points = [[.5, 0, 0], [0, 0, .6], [0, 0, 0], [.6, .6, 0]]
    np.testing.assert_allclose(metrics.surface_distance(points, box, [1, 1, 1]),
                               [0, .1, .5, np.sqrt(.02)], atol=5e-8)
    if metrics.first_coupled([[0, 0, 0], [0, 0, .2]], [0, 0, 0], [[False, False], [True, False]]) != 2:
        raise ValueError('3D20cm and any original participant alive')
    if metrics.first_coupled([[0, 0, .201]], [0, 0, 0], [[True, True]]) != -1:
        raise ValueError('Coupled height must enter 3D distance')
    if metrics.minimum_error([.01, .5, .001], [True, True, False], True) != 1.:
        raise ValueError('Minimum valid error, including before success')
    if metrics.minimum_error([.01], [True], False) is not None:
        raise ValueError('Failure trial excluded from Error')
    np.testing.assert_allclose(metrics.interaction_rates([1, 0], [1], [0, 0, 0]), [50, 100, 0, 50])
    if metrics.interaction_rates([1], [], [1]) != [100, None, 100, None]:
        raise ValueError('Empty relation denominator must be null')
    pose = np.zeros((8, 7), np.float32)
    pose[:, 6] = 1
    rim = np.asarray(metrics.world_rim(pose, [2, 1, .6]))
    np.testing.assert_array_equal(rim[0, 0], [-1, -.5, 0])
    if not np.all((np.abs(rim[0, :, 0]) == 1) | (np.abs(rim[0, :, 1]) == .5)):
        raise ValueError('Rim points must be on the mid-height XY perimeter')
    np.testing.assert_array_equal(rim[..., 2], 0)
    cube = np.zeros((8, 64, 3), np.float32)
    cube[:] = ((np.arange(8)*.0333)**3)[:, None, None]
    np.testing.assert_allclose(metrics.mean_absolute_jerk(cube, 0, 8), 6, atol=3e-4)
    np.testing.assert_equal(metrics.mean_absolute_jerk(cube, 5, 6), 0)
    np.testing.assert_equal(metrics.mean_absolute_jerk(cube, 8, 8), 0)
    tail = cube.copy()
    tail[5:] = 10000
    np.testing.assert_equal(metrics.mean_absolute_jerk(cube, 1, 4), metrics.mean_absolute_jerk(tail, 1, 4))
    return count+17


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--implementation', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    metrics = reference_metrics
    if args.implementation:
        path = args.implementation.resolve()
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location('candidate_metrics', path)
        metrics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics)
    print(json.dumps(dict(protocol_id='cvpr2027-evaluation-v1.0.0', passed=verify(metrics),
                          implementation=str(args.implementation or 'bundled reference'),
                          scope='calculation examples only; scene sampling and policy execution require separate checks')))

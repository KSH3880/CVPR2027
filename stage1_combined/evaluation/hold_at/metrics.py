"""Native Chair AT location-ever metrics; independent of training reward."""
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from .carry_metrics import score as carry_score, PROTOCOL as CARRY_PROTOCOL
except ImportError:  # Direct script invocation / driver imports neighboring modules.
    from carry_metrics import score as carry_score, PROTOCOL as CARRY_PROTOCOL

PROTOCOL = 'native_chair_hold_at_30s_v1'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def classify(objects, goals, dtype=np.float64):
    objects = np.asarray(objects, dtype=dtype)
    goals = np.asarray(goals, dtype=dtype)
    delta = objects[..., :3] - goals[..., :3]
    return np.linalg.norm(delta, axis=-1) <= .20

def first_episode(point, done):
    point, done = np.asarray(point, bool), np.asarray(done, bool)
    if point.shape != done.shape or point.ndim != 2:
        raise ValueError('Expected matching [frames, environments] arrays')
    active = np.ones(point.shape[1], bool)
    masks, ever, lengths = [], np.zeros_like(active), np.zeros(len(active), int)
    for p, d in zip(point, done):
        masks.append(active.copy())
        ever |= p & active  # Include the terminal frame, exclude all later frames.
        lengths += active
        active &= ~d
    if active.any():
        raise ValueError('Incomplete first episodes')
    return np.stack(masks), ever, lengths

def aggregate(records, expected_envs=512):
    if sorted(r['seed'] for r in records) != [0, 1, 2]:
        raise ValueError('Need one independent process for each seed 0/1/2')
    if any(r['num_envs'] != expected_envs or r['protocol'] != PROTOCOL for r in records):
        raise ValueError('Incompatible evaluation records')
    values = [100 * r['point_ever_count'] / expected_envs for r in sorted(records, key=lambda x: x['seed'])]
    if any(r['region_ever_count'] != r['point_ever_count'] for r in records):
        raise ValueError('AT point and region must be identical')
    return dict(seed_percent=values, mean_percent=float(np.mean(values)),
                sample_std_percentage_points=float(np.std(values, ddof=1)))


def aggregate_tokenhsi(records, expected_envs=512):
    proxy = []
    for record in records:
        secondary = record['tokenhsi_criteria']
        if secondary['protocol'] != CARRY_PROTOCOL:
            raise ValueError('Incompatible secondary carry criteria')
        proxy.append(dict(record, point_ever_count=secondary['success_count'],
                          region_ever_count=secondary['success_count']))
    stats = aggregate(proxy, expected_envs)
    stats['protocol'] = CARRY_PROTOCOL
    total = sum(r['tokenhsi_criteria']['success_count'] for r in records)
    stats['pooled_success_precision_cm'] = (sum(r['tokenhsi_criteria']['success_precision_cm'] *
        r['tokenhsi_criteria']['success_count'] for r in records) / total) if total else None
    return stats

def validate(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    result = json.loads((folder / 'results.json').read_text())
    assert manifest['protocol'] == result['protocol'] == PROTOCOL
    with np.load(folder / 'trace_0.npz') as t:
        data = {k: t[k] for k in t.files}
    with np.load(folder / 'initial_0.npz') as t:
        initial = {k: t[k] for k in t.files}
    assert np.isfinite(data['objects']).all() and np.isfinite(data['goals']).all()
    p = classify(data['objects'], data['goals'])
    np.testing.assert_array_equal(p, data['point'])
    np.testing.assert_array_equal(data['point'], data['region'])
    active, ever, lengths = first_episode(p, data['done'])
    np.testing.assert_array_equal(active, data['active'])
    _, ever32, _ = first_episode(classify(data['objects'], data['goals'], np.float32), data['done'])
    np.testing.assert_array_equal(ever, ever32)
    np.testing.assert_array_equal(lengths, result['episode_steps'])
    assert int(ever.sum()) == result['point_ever_count'] == result['region_ever_count']
    assert len(ever) == manifest['num_envs'] == result['num_envs']
    np.testing.assert_allclose(data['target'], data['goals'][..., :3], atol=0, rtol=0)
    terminal = data['terminate'][lengths-1, np.arange(len(ever))]
    assert (lengths[~terminal] == 899).all()
    distance = np.linalg.norm(initial['objects'][:, :2] - initial['root'][:, :2], axis=-1)
    assert ((distance >= 1-1e-5) & (distance <= 5+1e-5)).all()
    assert (np.abs(initial['goals'][:, :2] - initial['origins'][:, :2]) <= 4.5+1e-5).all()
    for target in (initial['root'][:, :2], initial['objects'][:, :2]):
        assert (((initial['goals'][:, :2]-target)**2).sum(-1) >= 2-1e-5).all()
    np.testing.assert_allclose(initial['goals'][:, 2], initial['ground_z'], rtol=0, atol=1e-6)
    # Data were recorded only after actions: initial geometry never enters ever.
    assert manifest['initial_frame_scored'] is False
    assert manifest['reset_after_step'] is False and manifest['deterministic'] is True
    if 'tokenhsi_criteria' in result:
        with np.load(folder / 'object_metadata.npz') as metadata:
            np.testing.assert_array_equal(metadata['model_ids'], initial['model_ids'])
            up = metadata['up_facing']
        secondary = carry_score(data, up)
        assert secondary == result['tokenhsi_criteria']
        secondary32 = carry_score(data, up, np.float32)
        np.testing.assert_array_equal(secondary['success_by_env'], secondary32['success_by_env'])
        assert np.all(~np.asarray(secondary['success_by_env']) | ever)
    for name, digest in json.loads((folder / 'completed.json').read_text())['sha256'].items():
        assert sha256(folder / name) == digest, name
    assert sha256(manifest['checkpoint']) == manifest['checkpoint_sha256']
    report = dict(ok=True, point_region_identical=True, first_episode=True,
                  terminal_frame_included=True, float32_float64_ever_equal=True,
                  time_limit_899=True, native_sampling_valid=True, artifact_hashes_valid=True)
    if 'tokenhsi_criteria' in result:
        report.update(tokenhsi_criteria_rescored=True, tokenhsi_599_action_limit=True,
                      tokenhsi_float32_float64_ever_equal=True)
    (folder / 'rescore_validation.json').write_text(json.dumps(report, indent=2))
    return report

if __name__ == '__main__':
    import sys
    for path in sys.argv[1:]:
        print(path, validate(path))

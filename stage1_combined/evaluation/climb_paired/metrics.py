"""Independent saved-trace scoring and paired, unfiltered seed statistics."""
import numpy as np
from scipy.spatial.transform import Rotation


def score_trace(path,action='CLIMB'):
    assert action in ['SIT','CLIMB']
    with np.load(path) as trace:
        root = trace['root'].astype(np.float64)
        objects = trace['objects'].astype(np.float64)
        alive = trace['alive'].astype(bool)
        done = trace['done'].astype(bool)
        surface, extent = trace['surface'], trace['extent']
        assert np.isfinite(root).all() and np.isfinite(objects).all()
        frames, count = root.shape[:2]
        rotation = Rotation.from_quat(objects[..., 3:7].reshape(-1, 4))
        target = objects[..., :3] + rotation.apply(np.broadcast_to(surface, (frames, count, 3)).reshape(-1, 3)).reshape(frames, count, 3)
        target[..., 2] += .1 if action=='SIT' else .94
        delta = root-target
        local = rotation.inv().apply(delta.reshape(-1, 3)).reshape(frames, count, 3)
        point = np.linalg.norm(delta, axis=-1) <= .2
        region = (np.abs(local[..., :2]) <= extent*.4).all(-1) & (np.abs(delta[..., 2]) <= .2)
        # Terminal frame belongs to the first episode. Later frames never do.
        expected_alive = np.concatenate([np.ones((1, count), bool), ~np.maximum.accumulate(done[:-1], axis=0)], axis=0)
        np.testing.assert_array_equal(alive, expected_alive)
        assert (done & alive).sum(0).min() == 1, 'Incomplete first episode'
        assert (done & alive).sum(0).max() == 1
        expected_end = np.argmax(done, axis=0)
        np.testing.assert_array_equal(trace['end_frame'], expected_end)
        return dict(point_count=int((point & alive).any(0).sum()),
                    region_count=int((region & alive).any(0).sum()),
                    num_envs=count, max_end_frame=int(expected_end.max()),
                    frames=frames)


def summarize(rows):
    """Retain every evaluated seed; winners are a diagnostic subset only."""
    assert rows and len({row['seed'] for row in rows}) == len(rows)
    summary = dict(evaluated_seeds=[row['seed'] for row in rows], metrics={})
    for metric in ['point', 'region']:
        public = np.array([row['tokenhsi'][metric+'_ever'] for row in rows])*100
        ours = np.array([row['ours'][metric+'_ever'] for row in rows])*100
        gap = ours-public
        def stats(values):
            return dict(mean_percent=float(values.mean()), sample_std_percentage_points=float(values.std(ddof=1)) if len(values)>1 else None)
        summary['metrics'][metric] = dict(tokenhsi=stats(public), ours=stats(ours),
            paired_delta_mean_percentage_points=float(gap.mean()),
            paired_delta_sample_std_percentage_points=float(gap.std(ddof=1)) if len(gap)>1 else None,
            winning_seeds=[row['seed'] for row, value in zip(rows, gap) if value>0],
            tied_seeds=[row['seed'] for row, value in zip(rows, gap) if value==0],
            losing_seeds=[row['seed'] for row, value in zip(rows, gap) if value<0])
    return summary

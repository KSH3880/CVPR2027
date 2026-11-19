"""Score follower behavior from completed Enabling rollouts."""
import argparse
import csv
from pathlib import Path
import sys
from xml.etree import ElementTree

import numpy as np
import torch

from cohort import ROOT, OUT, jobs, key, read

sys.path[:0] = [str(ROOT/'eval'), str(ROOT/'tokenhsi')]
from table2_metrics import surface_distance


def hand_ids():
    xml = ROOT/'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml'
    bodies = [body.get('name') for body in ElementTree.parse(xml).find('worldbody').iter('body')]
    return [bodies.index(name) for name in ('right_hand', 'left_hand')]


def score(source, analysis, setting):
    initial = torch.load(source/'initial.pt', map_location='cpu', weights_only=False)
    parent = initial['predecessor'].numpy()
    if not ((parent >= 0).sum(1) == 1).all():
        raise ValueError(f'Expected one follower per group: {source}')
    follower = (parent >= 0).argmax(1)
    env = np.arange(len(follower))
    dt = read(source/'table2.json')['control_dt']
    if not np.isclose(dt, 1/30):
        raise ValueError(f'Unexpected timestep: {dt}')

    with np.load(source/'trace.npz') as trace:
        roots = trace['roots'][:, env, follower, :2]
        alive = trace['valid'][:, env, follower]
        boxes = trace['boxes'] if setting == 'enable_ontop' else None
    with np.load(analysis) as scored:
        first_at, first_group = scored['first_at'], scored['first_group']
    frame = np.arange(1, len(roots)+1)[:, None]
    if len(roots) != 900 or (first_at.shape != (len(env),)) or first_group.shape != first_at.shape:
        raise ValueError(f'Expected a completed formal Enabling trace: {source}')
    alive &= (first_group < 0) | (frame <= first_group)

    positions = np.concatenate((initial['roots'].numpy()[env, follower, :2][None], roots))
    # Estimate planar speed over a trailing 0.2 s window.
    start = np.maximum(frame[:, 0]-6, 0)
    speed = np.linalg.norm(positions[1:]-positions[start], axis=-1) / ((frame-start[:, None])*dt)
    moving = speed >= .15
    pickup = np.zeros_like(alive)
    if boxes is not None:
        payload = initial['packet'].numpy()[env, follower, 3]-parent.shape[1]
        if not ((payload >= 0) & (payload < boxes.shape[2])).all():
            raise ValueError(f'Invalid follower payload indices: {source}')
        pose = boxes[:, env, payload]
        size = initial['sizes'].numpy()[env, payload]
        initial_z = initial['boxes'].numpy()[env, payload, 2]
        ids = hand_ids()
        with np.load(source/'body_trace.npz') as body:
            hands = body['positions'][:, env, follower][:, :, ids, :]
        pose_tensor = torch.from_numpy(pose.reshape(-1, 7))
        hand_tensor = torch.from_numpy(hands.reshape(-1, 1, 2, 3))
        sizes = torch.from_numpy(size).repeat(len(roots), 1)
        near = surface_distance(hand_tensor, pose_tensor, sizes).reshape(len(roots), len(env), 2).numpy()
        candidate = (near <= .12).all(-1) & (pose[..., 2]-initial_z >= .05) & alive
        for offset in range(5):
            end = len(roots)-4+offset
            pickup[4:] = candidate[offset:end] if offset == 0 else pickup[4:] & candidate[offset:end]

    states = (alive & ~pickup & ~moving, alive & ~pickup & moving, alive & pickup)
    if not np.array_equal(sum(state.astype(np.int8) for state in states), alive):
        raise ValueError(f'Behavior states do not partition valid frames: {source}')
    return frame, first_at, alive, states


def rows_for_job(suite, job):
    name = key(job)
    frame, first_at, alive, states = score(
        suite/'rollouts'/name, suite/'analysis'/f'{name}.npz', job['setting'])
    clocks = (
        ('episode_start', frame-1, alive),
        ('first_at', frame-first_at, alive & (first_at > 0)),
    )
    for clock, relative_frame, valid in clocks:
        for start in range(-30 if clock == 'first_at' else 0, 30):
            mask = valid & (relative_frame >= 30*start) & (relative_frame < 30*(start+1))
            frames = int(mask.sum())
            if frames:
                counts = [int((mask & state).sum()) for state in states]
                yield dict(job, clock=clock, start_s=start, end_s=start+1, frames=frames,
                           episodes=int(mask.any(0).sum()), waiting_frames=counts[0],
                           moving_frames=counts[1], picking_up_frames=counts[2],
                           waiting_pct=100*counts[0]/frames, moving_pct=100*counts[1]/frames,
                           picking_up_pct=100*counts[2]/frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=OUT)
    parser.add_argument('--output', type=Path, default=OUT.parent/'follower_timeline.csv')
    args = parser.parse_args()
    expected = [job for job in jobs() if job['setting'].startswith('enable')]
    torch.set_num_threads(1)
    rows = [row for job in expected for row in rows_for_job(args.suite, job)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(expected)} rollouts, {len(rows)} time bins: {args.output}')


if __name__ == '__main__':
    main()

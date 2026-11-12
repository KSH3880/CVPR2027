"""Select a common physical scene by explicit observed rollout criteria."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import isaacgym
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('COMPARE_OUTPUT_ROOT', str(ROOT/'output_etc/check/push_comparison_capture_20261112'))).resolve()
SOURCE = Path(os.environ.get('COMPARE_SOURCE_ROOT', str(ROOT/'output_etc/check/push_wait_v4_seed0_5_20261112'))).resolve()


def first_run(mask, length=3):
    hits = np.flatnonzero(np.convolve(mask.astype(int), np.ones(length, int), 'valid') == length)
    return int(hits[0]) if len(hits) else -1


def analyze(trace, index, initial, env):
    x = {k: v[:, index].numpy() for k, v in trace.items()}
    position = x['boxes'][:, :3]
    start = position[0].copy()
    goal = initial['center_goals'][env, 0].numpy()
    direction = goal[:2]-start[:2]
    direction /= np.linalg.norm(direction)
    delta = position[:, :2]-start[:2]
    lateral = np.abs(delta[:, 0]*direction[1]-delta[:, 1]*direction[0])
    moved = np.linalg.norm(delta, axis=-1)
    error = np.linalg.norm(position-goal, axis=-1)
    valid = x['alive'].all(-1)
    face = x['on_face'] & x['alive']
    arrivals = [first_run(face[:, p]) for p in range(2)]
    present = [p for p in range(2) if arrivals[p] >= 0]
    early = min(present, key=lambda p: arrivals[p]) if present else -1
    first = arrivals[early] if early >= 0 else -1
    both = first_run(face.all(-1) & valid)
    stable_success = first_run((error <= .2) & valid, 15)
    success = np.flatnonzero((error <= .2) & valid)
    freeze = np.flatnonzero(x['frozen'].any(-1) & valid)
    release = np.flatnonzero(x['released'].all(-1) & valid)
    release_step = int(release[0]) if len(release) else -1
    contact_window = valid & (np.arange(len(valid)) >= max(release_step, both, 0))
    if stable_success >= 0:
        contact_window &= np.arange(len(valid)) <= stable_success
    contact_rate = float(face[contact_window].all(-1).mean()) if contact_window.any() else 0.
    second = arrivals[1-early] if early >= 0 else -1
    # TokenHSI phase 2 is explicitly before B has arrived.
    before_second = valid & (np.arange(len(valid)) >= max(first, 0))
    if second >= 0:
        before_second &= np.arange(len(valid)) < second
    early_move = float(moved[before_second].max()) if before_second.any() else 0.
    eligible2 = np.flatnonzero(before_second & (moved >= .2))
    phase2 = int(eligible2[0]) if len(eligible2) else max(both, first, 0)
    valid_after = np.flatnonzero(valid & (np.arange(len(valid)) > max(phase2, release_step, 0)+15))
    phase3 = int(valid_after[np.argmax(lateral[valid_after])]) if len(valid_after) else int(np.flatnonzero(valid)[-1])
    goal_miss = not len(success)
    summary = dict(first_person=early, arrivals=arrivals, first_step=first, all_contact_step=both,
        first_frozen_step=int(freeze[0]) if len(freeze) else -1, release_step=release_step,
        success_step=stable_success, ever_goal20cm=bool(len(success)),
        minimum_goal_error_m=float(error[valid].min()), final_goal_error_m=float(error[np.flatnonzero(valid)[-1]]),
        max_lateral_deviation_m=float(lateral[valid].max()), premature_box_motion_m=early_move,
        joint_contact_fraction=contact_rate, alive_at_end=bool(valid[-1]),
        phase_steps=[max(first, 0)+2, phase2, phase3],
        token_pattern=first >= 0 and early_move >= .2 and goal_miss and lateral[valid].max() >= .3,
        fsm_pattern=len(freeze) > 0 and release_step >= 0 and goal_miss and contact_rate < .85 and lateral[valid].max() >= .3)
    if len(freeze) and release_step >= 0:
        summary['phase_steps'] = [int(freeze[0])+3, release_step+3, phase3]
        a = int(freeze[0])
        summary['wait_box_motion_m'] = float(np.linalg.norm(position[a:release_step+1, :2]-position[a, :2], axis=-1).max())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2])
    args = parser.parse_args()
    cases = []
    for seed in args.seeds:
        folder = OUT/f'seed{seed}'
        if not all((folder/m/'rollout.pt').exists() for m in ('ours', 'tokenhsi', 'fsm')):
            continue
        initial = torch.load(SOURCE/f'seed{seed}/initial.pt', map_location='cpu', weights_only=False)
        runs = {m: torch.load(folder/m/'rollout.pt', map_location='cpu', weights_only=False) for m in ('ours', 'tokenhsi', 'fsm')}
        common = sorted(set.intersection(*(set(run['env_ids']) for run in runs.values())))
        original = {c['env_id']: c for c in json.loads((SOURCE/f'seed{seed}/candidates.json').read_text())['candidates']}
        for env in common:
            index = runs['ours']['env_ids'].index(env)
            methods = {m: analyze(run['trajectory'], run['env_ids'].index(env), initial, env) for m, run in runs.items()}
            ours = methods['ours']
            expected = original[env]
            a, b = sorted(ours['arrivals'])
            position = runs['ours']['trajectory']['boxes'][:, index, :3].numpy()
            waiting_motion = float(np.linalg.norm(position[a:b+1, :2]-position[a, :2], axis=-1).max()) if a >= 0 else 100.
            ours['wait_box_motion_m'] = waiting_motion
            ours['wait_seconds'] = (b-a)*initial['dt']
            ours['phase_steps'] = [max(a, 0)+3, max(ours['all_contact_step'], 0)+3, max(ours['success_step'], 0)+10]
            ours['ours_pattern'] = a >= 0 and b-a >= 15 and waiting_motion <= .15 and ours['success_step'] >= b
            matches = bool(ours['ours_pattern'] and methods['tokenhsi']['token_pattern'] and methods['fsm']['fsm_pattern'])
            same_first = all(s['first_person'] == ours['first_person'] for s in methods.values())
            cases.append(dict(seed=seed, env_id=env, index=index, matches=matches, same_first_person=same_first,
                methods=methods, original_ours=expected,
                rank=[int(matches), int(same_first), int(methods['tokenhsi']['token_pattern'])+int(methods['fsm']['fsm_pattern']),
                      ours['wait_seconds'], methods['tokenhsi']['premature_box_motion_m']]))
    cases.sort(key=lambda c: c['rank'], reverse=True)
    report = dict(criteria='Same initial snapshot. Token: >=20cm motion before partner arrival, never20cm goal, lateral>=30cm. FSM: froze/released, never20cm goal, contact fraction<85%, lateral>=30cm. Ours: waits>=15frames, box<=15cm, stable goal20cm.',
                  seeds=args.seeds, scenes=len(cases), exact_pattern_matches=sum(c['matches'] for c in cases), cases=cases)
    (OUT/'comparison_ranking.json').write_text(json.dumps(report, indent=2, default=lambda x: x.item())+'\n')
    print(json.dumps(dict(scenes=len(cases), matches=report['exact_pattern_matches'], top=cases[:2]), indent=2, default=lambda x: x.item()))


if __name__ == '__main__':
    main()

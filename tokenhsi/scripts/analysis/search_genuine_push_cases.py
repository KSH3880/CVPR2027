"""Require genuine solo early pushing, later normal B joining, and large failure."""
import argparse
import json
import os
from pathlib import Path
import isaacgym
import numpy as np
import torch
from push_comparison_select import analyze, first_run

ROOT = Path(__file__).resolve().parents[3]
MIN_PARTNER_GAP = float(os.environ.get('GENUINE_MIN_PARTNER_GAP', '.7'))
MIN_FINAL_ERROR = float(os.environ.get('GENUINE_MIN_FINAL_ERROR', '1.5'))


def genuine_metrics(trace, index, initial, env):
    info = analyze(trace, index, initial, env)
    p = trace['boxes'][:, index, :3].numpy()
    roots = trace['roots'][:, index, :, :3].numpy()
    valid = trace['alive'][:, index].all(-1).numpy()
    face = trace['on_face'][:, index].numpy()
    force = trace['force_contact'][:, index].numpy()
    early = info['first_person']
    result = dict(genuine_early_push=False, rejection=[], early_capture_step=-1)
    if early >= 0:
        a = info['arrivals'][early]
        gap_at_first = float(trace['surface_distance'][a, index, 1-early].min())
        result['partner_hand_gap_at_first_contact_m'] = gap_at_first
        if gap_at_first < MIN_PARTNER_GAP:
            result['rejection'].append('B is already beside the box when A first contacts; not a clear approach stage')
    if early < 0 or min(info['arrivals']) < 0 or info['all_contact_step'] < 0:
        result['rejection'].append('B never normally joins both assigned hands with A')
        return dict(info, **result)
    a, b, joint = info['arrivals'][early], info['arrivals'][1-early], info['all_contact_step']
    direction = initial['center_goals'][env, 0, :2].numpy()-p[0, :2]
    direction /= np.linalg.norm(direction)
    projection = (p[:, :2]-p[a, :2])@direction
    displacement = np.linalg.norm(p[:, :2]-p[a, :2], axis=-1)
    velocity_forward = np.concatenate(([0.], np.diff(p[:, :2], axis=0)@direction))
    other = 1-early
    gap = trace['surface_distance'][:, index, other].numpy().min(-1)
    b_distance = np.linalg.norm(roots[:, other, :2]-p[a, :2], axis=-1)
    b_approach = b_distance[a]-b_distance
    b_step = np.zeros_like(roots[:, other, :2])
    b_step[3:] = roots[3:, other, :2]-roots[:-3, other, :2]
    b_target = p[:, :2]-roots[:, other, :2]
    b_target /= np.linalg.norm(b_target, axis=-1, keepdims=True).clip(1e-6)
    b_forward = (b_step*b_target).sum(-1)
    active = face[:, early] & force[:, early] & valid
    first_force = first_run(active & (np.arange(len(active)) < b), 3)
    startup_end = min(a+15, b)
    startup_move = float(projection[a:startup_end+1].max())
    candidates = np.flatnonzero(active & (gap >= .5) & (projection >= .5)
        & (displacement >= .6) & (b_approach >= .15) & (b_forward >= .015) & (velocity_forward >= .003)
        & (np.arange(len(p)) >= a+6) & (np.arange(len(p)) <= b-6))
    if first_force < 0 or first_force > a+15 or startup_move < .10:
        result['rejection'].append('No prompt force-driven normal forward push after first contact')
    elif gap[first_force] < MIN_PARTNER_GAP:
        result['rejection'].append('B is already near the box when force-driven A pushing starts')
    joint_force = int((force[joint:] & valid[joint:, None]).all(-1).sum())
    joint_run = first_run(face.all(-1) & valid, 6)
    if joint_force < 3 or joint_run < 0:
        result['rejection'].append('No sustained normal joint push with force after B arrives')
    if not len(candidates):
        result['rejection'].append('No >=60cm solo motion / >=50cm goalward advance while B visibly approaches')
    else:
        # Keep B visibly separated and A actively pushing; never use a later jam.
        separated = candidates[gap[candidates] >= .75]
        usable = separated if len(separated) else candidates
        frame = int(usable[np.argmax(displacement[usable])])
        solo_force = int(active[a:frame+1].sum())
        if solo_force < 6 or solo_force/(frame-a+1) < .25:
            result['rejection'].append('Insufficient sustained solo pushing force')
        result.update(early_capture_step=frame,
            clear_partner_early_motion_m=float(displacement[frame]),
            goalward_advance_before_partner_m=float(projection[frame]),
            early_partner_min_hand_gap_m=float(gap[frame]),
            partner_approach_progress_m=float(b_approach[frame]),
            partner_approach_speed_mps=float(b_forward[frame]/(3*initial['dt'])),
            solo_force_contact_frames=solo_force,
            first_force_push_step=first_force,
            first_contact_startup_advance_m=startup_move,
            later_b_join_step=b, later_joint_step=joint,
            later_joint_force_frames=joint_force)
    result['genuine_early_push'] = not result['rejection']
    return dict(info, **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'output_etc/check/push_wait_v4_seed0_5_20261112')
    parser.add_argument('--rollouts', type=Path, default=ROOT/'output_etc/check/push_comparison_capture_20261112')
    args = parser.parse_args()
    rows = []
    for path in sorted(args.rollouts.glob('seed*/tokenhsi/rollout.pt')):
        seed = int(path.parent.parent.name[4:])
        run = torch.load(path, map_location='cpu', weights_only=False)
        initial = torch.load(args.source/f'seed{seed}/initial.pt', map_location='cpu', weights_only=False)
        ours = {c['env_id']: c for c in json.loads((args.source/f'seed{seed}/candidates.json').read_text())['candidates']}
        fsm_path = args.rollouts/f'seed{seed}/fsm/rollout.pt'
        fsm = torch.load(fsm_path, map_location='cpu', weights_only=False) if fsm_path.exists() else None
        for index, env in enumerate(run['env_ids']):
            if env not in ours:
                continue
            token = genuine_metrics(run['trajectory'], index, initial, env)
            same = token['first_person'] == ours[env]['first_person']
            large_failure = (not token['ever_goal20cm'] and token['minimum_goal_error_m'] >= .5
                and token['final_goal_error_m'] >= MIN_FINAL_ERROR and token['max_lateral_deviation_m'] >= .6)
            good_ours = ours[env]['qualifies'] and ours[env]['joint_contact_fraction_until_success'] >= .95
            strong = bool(token['genuine_early_push'] and large_failure and same and good_ours)
            fsm_info = analyze(fsm['trajectory'], fsm['env_ids'].index(env), initial, env) if fsm and env in fsm['env_ids'] else None
            complete = bool(strong and fsm_info and fsm_info['fsm_pattern'] and fsm_info['first_person']==token['first_person'])
            rows.append(dict(seed=seed, env_id=env, strong_token=strong, complete=complete,
                genuine_early_push=token['genuine_early_push'], same_first_person=same,
                tokenhsi=token, ours=ours[env], fsm=fsm_info,
                rank=[int(complete),int(strong),int(token['genuine_early_push']),int(same),
                    min(token.get('goalward_advance_before_partner_m',0)/.5,token['final_goal_error_m']/MIN_FINAL_ERROR),
                    token['max_lateral_deviation_m']]))
    rows.sort(key=lambda row:row['rank'], reverse=True)
    report = dict(minimum_partner_gap_m=MIN_PARTNER_GAP,
        minimum_final_goal_error_m=MIN_FINAL_ERROR,
        criteria=f'B hand gaps>={MIN_PARTNER_GAP}m at A first contact/force; prompt A force-driven forward push; >=60cm solo displacement and >=50cm goalward advance; B visibly approaches with hand gaps>=50cm (prefer75cm), then normally joins; >=3 joint force frames; never20cm goal, minimum goal error>=50cm, final goal error>={MIN_FINAL_ERROR}m, lateral>=60cm; Ours wait/contact success; same-scene FSM failure.',
        scenes=len(rows), genuine_cases=sum(r['genuine_early_push'] for r in rows),
        strong_cases=sum(r['strong_token'] for r in rows), complete_cases=sum(r['complete'] for r in rows), cases=rows)
    target=args.rollouts/'genuine_push_ranking.json'
    target.write_text(json.dumps(report,indent=2,default=lambda x:x.item())+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
    for row in rows[:10]:
        t=row['tokenhsi']
        print(row['seed'],row['env_id'],'genuine',row['genuine_early_push'],'strong',row['strong_token'],'complete',row['complete'],
            'early',round(t.get('clear_partner_early_motion_m',0),2),'final',round(t['final_goal_error_m'],2),'rejections',t['rejection'])


if __name__ == '__main__':
    main()

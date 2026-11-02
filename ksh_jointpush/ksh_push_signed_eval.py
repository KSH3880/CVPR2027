"""Signed terminal PUSH errors, using the recorded reset-to-goal XY direction."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from ksh_jointpush.ksh_push_terminal_eval import push_terminal_records, write_csv, checkpoint_label


def signed_geometry(start, point, goal):
    start, point, goal = [np.asarray(x, dtype=np.float64) for x in (start, point, goal)]
    direction = goal[:2]-start[:2]
    length = np.linalg.norm(direction)
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError('The recorded initial PUSH-to-goal XY direction must be nonzero')
    axis = direction/length
    delta = point-goal
    along = float(np.dot(delta[:2], axis))
    distance = float(np.linalg.norm(delta))
    return dict(signed_goal_distance_m=distance if along > 0 else -distance,
                signed_along_m=along,
                lateral_error_m=float(abs(delta[0]*axis[1]-delta[1]*axis[0])),
                terminal_past_goal=along > 0, terminal_past_goal_by_20cm=along > .2,
                initial_goal_xy_distance_m=float(length), axis_x=float(axis[0]), axis_y=float(axis[1]),
                start_x=float(start[0]), start_y=float(start[1]), start_z=float(start[2]),
                goal_x=float(goal[0]), goal_y=float(goal[1]), goal_z=float(goal[2]),
                terminal_x=float(point[0]), terminal_y=float(point[1]), terminal_z=float(point[2]))


def signed_terminal_records(trace, initial, seed):
    agents, physical = push_terminal_records(trace, seed)
    packet = np.asarray(trace['packet'])
    boxes = np.asarray(trace['boxes'])
    goals = np.asarray(trace['goals'])
    groups = np.asarray(trace['groups'])
    initial_boxes = np.asarray(initial['boxes'])
    assert np.array_equal(np.asarray(initial['packet']), packet)
    assert np.array_equal(np.asarray(initial['goals']), goals)
    assert np.array_equal(np.asarray(initial['groups']), groups)
    m, objects = packet.shape[1], boxes.shape[2]
    for row in agents:
        env, agent, step = row['env'], row['agent'], row['terminal_step']
        task_index = int(np.flatnonzero(packet[env, :, 2] == agent)[0])
        payload = int(packet[env, task_index, 3]-m)
        target = int(packet[env, task_index, 4]-m-objects)
        row.update(signed_geometry(initial_boxes[env, payload, :3], boxes[step-1, env, payload, :3], goals[env, target]))
    for row in physical:
        env, step, group = row['env'], row['terminal_step'], row['group']
        selected = ((packet[env, :, 0] == 1) & (packet[env, :, 1] == 4)
                    & (groups[env, packet[env, :, 2]] == group))
        payload = packet[env, selected, 3]-m
        target = packet[env, selected, 4]-m-objects
        start = initial_boxes[env, payload, :3].astype(np.float64).mean(0)
        point = boxes[step-1, env, payload, :3].astype(np.float64).mean(0)
        goal = goals[env, target].astype(np.float64).mean(0)
        row.update(signed_geometry(start, point, goal))
    for row in agents+physical:
        assert abs(abs(row['signed_goal_distance_m'])-row['distance_3d_m']) < 1e-10
        assert abs(row['signed_along_m']) <= row['distance_xy_m']+1e-10
        reconstructed = row['signed_along_m']**2+row['lateral_error_m']**2+(row['terminal_z']-row['goal_z'])**2
        assert abs(reconstructed-row['distance_3d_m']**2) < 1e-7
    return agents, physical


def signed_stats(records):
    if not records:
        return None
    result = dict(trials=len(records),
                  terminal_past_goal_rate=float(np.mean([r['terminal_past_goal'] for r in records])),
                  terminal_past_goal_by_20cm_rate=float(np.mean([r['terminal_past_goal_by_20cm'] for r in records])))
    for name in ('signed_goal_distance_m', 'signed_along_m', 'lateral_error_m'):
        values = np.asarray([r[name] for r in records])
        means = {str(seed):float(np.mean([r[name] for r in records if r['seed']==seed]))
                 for seed in sorted({r['seed'] for r in records})}
        result[name] = dict(mean=float(values.mean()), median=float(np.median(values)),
                           std=float(values.std(ddof=1)) if len(values)>1 else None,
                           min=float(values.min()), max=float(values.max()), seed_means=means,
                           mean_of_seed_means=statistics.mean(means.values()),
                           std_of_seed_means=statistics.stdev(means.values()) if len(means)>1 else None)
    return result


def summarize_signed(directory):
    import torch
    agents, physical, sources = [], [], []
    for path in sorted(directory.glob('general_seed*/trace.npz')):
        run=json.loads((path.parent/'result.json').read_text())
        initial_path=path.parent/'initial.pt'
        initial=torch.load(initial_path,map_location='cpu')
        with np.load(path) as archive:
            trace={name:archive[name] for name in ('packet','boxes','goals','groups','fall_step','ever')}
        a,b=signed_terminal_records(trace,initial,run['seed'])
        assert len(a)==run['scores']['point']['PUSH']['trials']
        assert sum(r['ever_success'] for r in a)==run['scores']['point']['PUSH']['success']
        sources.append(dict(seed=run['seed'],trace=str(path),initial=str(initial_path),
                            initial_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest()))
        agents.extend(a);physical.extend(b)
    assert len(sources)==3 and sorted(s['seed'] for s in sources)==[0,1,2]
    report=dict(units='metres',sources=sources,
                axis_definition='u=(goal_xy-initial_box_or_region_xy)/norm(goal_xy-initial_box_or_region_xy), from initial.pt immediately after reset, not frame 1 or current box orientation.',
                signed_along_formula='dot(terminal_xy-goal_xy,u); positive beyond goal, negative short of goal; lateral offset excluded.',
                signed_goal_distance_formula='3D terminal-to-goal distance with positive sign if signed_along>0, negative otherwise. Exactly on the goal plane with lateral error is assigned negative; exact goal is zero.',
                terminal_definition='Same episode-end snapshots as the original terminal report: horizon or first all-agent fall; all successes/failures/falls included.',
                averaging='Pool trials across three evaluation seeds. Per-seed means and their sample standard deviation are also stored. Positive and negative errors cancel in the signed mean.',
                agent_to_region_goal=signed_stats(agents),box_to_common_goal=signed_stats(physical),
                agent_by_members={str(k):signed_stats([r for r in agents if r['members']==k]) for k in range(1,5)},
                box_by_members={str(k):signed_stats([r for r in physical if r['members']==k]) for k in range(1,5)})
    write_csv(directory/'push_signed_terminal_agents.csv',agents)
    write_csv(directory/'push_signed_terminal_boxes.csv',physical)
    (directory/'push_signed_terminal_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# PUSH 목표 통과 방향을 반영한 종료 오차','',
           f'{checkpoint_label(directory)} 평가의 초기 reset snapshot(initial.pt)과 종료 trace를 재집계했다. +는 초기 상자→목표 XY 방향으로 목표선을 넘은 쪽, −는 넘지 않은 쪽이다. 현재 상자 회전 방향으로 부호를 바꾸지 않는다. 모든 성공·실패·낙상을 포함한다.','',
           '- ± 목표 거리: 기존3D 목표 거리의 크기를 유지하고 통과 여부로 부호를 붙인 값.',
           '- 방향상 오차: (종료XY−목표XY)를 초기 진행방향에 투영한 값. 횡방향 오차는 포함하지 않는다.',
           '- 20cm 초과 통과율: 방향상 오차가+0.20m를 넘은 비율. 종료 시점 기준이다.','',
           '| 대상 | 표본 수 | ± 목표 거리 평균 (m) | 방향상 오차 평균 (m) | 목표선 통과 (%) | 20cm 초과 통과 (%) |',
           '|---|---:|---:|---:|---:|---:|']
    def add(label,stats):
        if stats:
            lines.append(f"| {label} | {stats['trials']} | {stats['signed_goal_distance_m']['mean']:+.3f} | {stats['signed_along_m']['mean']:+.3f} | {stats['terminal_past_goal_rate']*100:.2f} | {stats['terminal_past_goal_by_20cm_rate']*100:.2f} |")
    add('전체 구역',report['agent_to_region_goal'])
    for k in range(1,5):add(f'{k}명 PUSH 구역',report['agent_by_members'][str(k)])
    add('전체 물리 상자',report['box_to_common_goal'])
    lines+=['','평균은 +/−가 상쇄되므로 통과율을 함께 본다. 구역은 개인 목표 기준으로 사람당 한 번, 물리 상자는 공통 목표 기준으로 그룹당 한 번 집계한다. 부호가 있는 목표 거리에는 횡방향·높이 오차도 포함된다.','',
            '[전체 JSON](push_signed_terminal_summary.json) · [구역별 CSV](push_signed_terminal_agents.csv) · [상자별 CSV](push_signed_terminal_boxes.csv) · [원래 절대거리 결과](REPORT.md)','']
    (directory/'PUSH_SIGNED_DISTANCE.md').write_text('\n'.join(lines))
    main_report = directory/'REPORT.md'
    link = '\n[목표 통과 방향을 반영한 부호 거리](PUSH_SIGNED_DISTANCE.md)\n'
    if main_report.exists() and link not in main_report.read_text():
        main_report.write_text(main_report.read_text()+link)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    print(json.dumps(summarize_signed(args.directory),indent=2),flush=True)


if __name__=='__main__':main()

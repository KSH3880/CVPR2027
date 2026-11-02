"""PUSH terminal distances from complete, reset-free formal evaluation traces."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import numpy as np


def push_terminal_records(trace, seed):
    packet = np.asarray(trace['packet'])
    boxes = np.asarray(trace['boxes'])
    goals = np.asarray(trace['goals'], dtype=np.float64)
    groups = np.asarray(trace['groups'])
    falls = np.asarray(trace['fall_step'])
    valid, kind, actor, payload, target = np.moveaxis(packet, -1, 0)
    valid = valid.astype(bool)
    frames, n, objects = boxes.shape[:3]
    m = packet.shape[1]
    assert np.array_equal(np.sort(actor, axis=1), np.broadcast_to(np.arange(m), (n, m)))
    assert np.isfinite(boxes).all() and np.isfinite(goals).all()
    assert valid.any(1).all() and ((falls == -1) | ((falls >= 1) & (falls <= frames))).all()
    rows = np.arange(n)[:, None]
    task_falls = falls[rows, actor]
    ended_by_fall = ((task_falls >= 1) | ~valid).all(1)
    end_step = np.where(ended_by_fall, np.where(valid, task_falls, 0).max(1), frames)
    end_boxes = boxes[end_step-1, np.arange(n)]
    push = valid & (kind == 4)
    payload_id, goal_id = payload-m, target-m-objects
    assert ((payload_id[push] >= 0) & (payload_id[push] < objects)).all()
    assert ((goal_id[push] >= 0) & (goal_id[push] < m)).all()
    payload_id, goal_id = payload_id.clip(0, objects-1), goal_id.clip(0, m-1)
    point = end_boxes[rows, payload_id, :3].astype(np.float64)
    goal = goals[rows, goal_id]
    own_step = np.where(task_falls >= 1, task_falls, end_step[:, None])
    own_point = boxes[own_step-1, rows, payload_id, :3].astype(np.float64)
    task_group = groups[rows, actor]
    ever = np.asarray(trace['ever'])[..., 0]
    agents, physical = [], []
    for env in range(n):
        _, counts = np.unique(groups[env], return_counts=True)
        layout = '+'.join(map(str, sorted(counts.tolist())))
        for task_index in np.flatnonzero(push[env]):
            group = int(task_group[env, task_index])
            members = int((groups[env] == group).sum())
            delta = point[env, task_index]-goal[env, task_index]
            agents.append(dict(seed=int(seed), env=env, agent=int(actor[env, task_index]), group=group,
                members=members, layout=layout, terminal_step=int(end_step[env]),
                terminal_reason='all_agents_fallen' if ended_by_fall[env] else 'horizon',
                distance_3d_m=float(np.linalg.norm(delta)), distance_xy_m=float(np.linalg.norm(delta[:2])),
                height_error_m=float(abs(delta[2])), geometric_at_goal=bool(np.linalg.norm(delta) <= .2),
                ever_success=bool(ever[env, task_index]), fallen=bool(task_falls[env, task_index] >= 1),
                agent_last_valid_step=int(own_step[env, task_index]),
                agent_last_valid_distance_3d_m=float(np.linalg.norm(own_point[env, task_index]-goal[env, task_index]))))
        for group in np.unique(task_group[env, push[env]]):
            selected = valid[env] & (task_group[env] == group)
            assert push[env, selected].all()
            assert int(selected.sum()) == int((groups[env] == group).sum())
            # Equal-width region offsets sum to zero under any box rotation.
            # Their mean is the full physical box center; fixed goal offsets
            # similarly average to the common center target.
            center = point[env, selected].mean(0)
            center_goal = goal[env, selected].mean(0)
            delta = center-center_goal
            physical.append(dict(seed=int(seed), env=env, group=int(group), members=int(selected.sum()),
                layout=layout, terminal_step=int(end_step[env]),
                terminal_reason='all_agents_fallen' if ended_by_fall[env] else 'horizon',
                distance_3d_m=float(np.linalg.norm(delta)), distance_xy_m=float(np.linalg.norm(delta[:2])),
                height_error_m=float(abs(delta[2])), geometric_at_goal=bool(np.linalg.norm(delta) <= .2),
                all_members_ever_success=bool(ever[env, selected].all())))
    return agents, physical


def distance_stats(records):
    if not records:
        return None
    result = dict(trials=len(records), terminal_at_goal=sum(r['geometric_at_goal'] for r in records),
                  terminal_at_goal_rate=float(np.mean([r['geometric_at_goal'] for r in records])))
    for key in ('distance_3d_m', 'distance_xy_m'):
        values = np.array([r[key] for r in records])
        seed_means = {str(seed): float(np.mean([r[key] for r in records if r['seed'] == seed]))
                      for seed in sorted({r['seed'] for r in records})}
        result[key] = dict(mean=float(values.mean()), std=float(values.std(ddof=1)) if len(values)>1 else None,
                           median=float(np.median(values)), p90=float(np.quantile(values, .9)),
                           min=float(values.min()), max=float(values.max()), seed_means=seed_means,
                           mean_of_seed_means=statistics.mean(seed_means.values()),
                           std_of_seed_means=statistics.stdev(seed_means.values()) if len(seed_means)>1 else None)
    return result


def write_csv(path, records):
    if records:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)


def summarize_terminal(directory):
    agents, physical, sources = [], [], []
    for path in sorted(directory.glob('general_seed*/trace.npz')):
        run = json.loads((path.parent/'result.json').read_text())
        assert run['protocol'] == 'general'
        with np.load(path) as trace:
            assert len(trace['boxes']) == run['simulated_steps']
            a, b = push_terminal_records(trace, run['seed'])
        assert len(a) == run['scores']['point']['PUSH']['trials']
        assert sum(r['ever_success'] for r in a) == run['scores']['point']['PUSH']['success']
        write_csv(path.parent/'push_terminal_agents.csv', a)
        write_csv(path.parent/'push_terminal_boxes.csv', b)
        sources.append(dict(seed=run['seed'], frames=run['simulated_steps'], source=str(path)))
        agents.extend(a); physical.extend(b)
    assert len(sources) == 3 and sorted(s['seed'] for s in sources) == [0, 1, 2]
    report = dict(units='metres', sources=sources,
        terminal_definition='Each environment ends at 1200 steps, or the first step when all its valid agents have fallen. Falls and failed attempts are included. Fall frames are included; no reset states are used.',
        agent_distance='Each PUSH virtual region center to its fixed assigned goal; solo uses its physical box center.',
        physical_box_distance='One observation per physical PUSH group: mean of equal-width region centers to mean of fixed goals. These means recover the full box center and common goal. Groups are not weighted by member count.',
        stats_note='mean/median/p90 pool trials. mean_of_seed_means and std_of_seed_means describe the three evaluation seeds, not three trained models.',
        agent_to_region_goal=distance_stats(agents), box_to_common_goal=distance_stats(physical),
        agent_by_members={str(k): distance_stats([r for r in agents if r['members']==k]) for k in range(1,5)},
        box_by_members={str(k): distance_stats([r for r in physical if r['members']==k]) for k in range(1,5)},
        agent_trials_ended_by_all_falls=sum(r['terminal_reason']=='all_agents_fallen' for r in agents))
    write_csv(directory/'push_terminal_agents.csv', agents)
    write_csv(directory/'push_terminal_boxes.csv', physical)
    (directory/'push_terminal_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def fmt(value, scale=1):
    return '—' if value is None else f'{value*scale:.2f}'


def checkpoint_label(directory):
    """Use the evaluated checkpoint identity instead of a previous run's title."""
    source_path = directory/'source_manifest.json'
    source = json.loads(source_path.read_text()) if source_path.exists() else {}
    epoch = source.get('checkpoint_epoch')
    if epoch is None:
        manifest = json.loads((directory/'manifest.json').read_text())
        checkpoint = Path(manifest['resolved_checkpoint'])
        match = re.search(r'_(\d+)$', checkpoint.stem)
        return f'{int(match.group(1))} iter' if match else checkpoint.stem
    return f'{int(epoch)} iter'


def write_report(directory, terminal):
    summary = json.loads((directory/'summary.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    provenance_path = directory/'source_manifest.json'
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    conditions = '공식 성공 채점은 보상·접촉 gate와 독립된 기하 판정이다.'
    if 'training_gate_aligned' in provenance and 'evaluation_gate_aligned' in provenance:
        label = lambda value: '포함' if value else '제거'
        conditions = (f"체크포인트 학습 당시: aligned {label(provenance['training_gate_aligned'])}. "
                      f"평가 실행 코드: aligned {label(provenance['evaluation_gate_aligned'])}. "
                      '저장된 학습 가중치를 그대로 평가했으며, '+conditions)
    rows = [f'# Stage2 {checkpoint_label(directory)} task별 평가', '',
            '512환경 × seed 0·1·2, 4명, 1200step, deterministic, 학습의 random_scenario 분포. 성공률은 평가 seed 평균 ± 표본 표준편차다.', '',
            '| Task | 시도 수 | 점 기준 ever 성공률 (%) | 영역 기준 ever 성공률 (%) |',
            '|---|---:|---:|---:|']
    for task in ('AT', 'ON_TOP', 'SIT', 'CLIMB', 'PUSH'):
        point, region = [summary['scores'][criterion][task] for criterion in ('point', 'region')]
        rows.append(f"| {task} | {point['trials']} | {fmt(point['mean'],100)} ± {fmt(point['std'],100)} | {fmt(region['mean'],100)} ± {fmt(region['std'],100)} |")
    rows += ['', '## PUSH 종료 거리', '',
             '각 환경의1200번째 step 또는 전원 낙상한 step을 종료 시점으로 사용한다. 성공·실패·낙상 사례를 모두 포함한다. 주 지표는 각 구역 중심→개인 고정 목표의3D 거리다. 전체 상자 중심 거리는 그룹당 한 번 집계한다.', '',
             '| 대상 | 표본 수 | 평균 (m) | 중앙값 (m) | 90백분위 (m) | 종료 시 20cm 이내 (%) |',
             '|---|---:|---:|---:|---:|---:|']
    for label, values in [('전체 구역',terminal['agent_to_region_goal']),
                          *[(f'{k}명 PUSH 구역',terminal['agent_by_members'][str(k)]) for k in range(1,5)],
                          ('전체 물리 상자',terminal['box_to_common_goal'])]:
        if values:
            d=values['distance_3d_m']
            rows.append(f"| {label} | {values['trials']} | {fmt(d['mean'])} | {fmt(d['median'])} | {fmt(d['p90'])} | {fmt(values['terminal_at_goal_rate'],100)} |")
    rows += ['', '종료 시20cm 비율은 종료 위치의 기하 판정이며, 낙상 전에 한 번이라도 달성한 공식 ever 성공률과 별도다. XY거리·평가 seed별 거리 평균·개인 마지막 유효 시점 거리도 JSON/CSV에 보존한다.', '',
             f"Checkpoint: `{manifest['resolved_checkpoint']}`", '',
             conditions, '',
             '[공식 task 집계](summary.json) · [PUSH 종료 거리 집계](push_terminal_summary.json) · [구역별 원시 CSV](push_terminal_agents.csv) · [상자별 원시 CSV](push_terminal_boxes.csv) · [거리 분포 PNG](push_terminal_distance.png)', '']
    (directory/'REPORT.md').write_text('\n'.join(rows))



def plot_terminal(directory):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(directory/'mpl_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (directory/'push_terminal_agents.csv').open() as f:
        agents=list(csv.DictReader(f))
    with (directory/'push_terminal_boxes.csv').open() as f:
        boxes=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),constrained_layout=True)
    def ecdf(ax,rows,label):
        values=np.sort([float(r['distance_3d_m']) for r in rows])
        if len(values):ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=f'{label} (n={len(values)})')
    for k in range(1,5):
        ecdf(axes[0],[r for r in agents if int(r['members'])==k],f'{k}-agent PUSH')
    ecdf(axes[1],agents,'Region to assigned goal; per agent')
    ecdf(axes[1],boxes,'Full box to common goal; per group')
    maximum=max(float(r['distance_3d_m']) for r in agents+boxes)
    for ax in axes:
        ax.axvline(.2,color='#444',ls='--',lw=1,label='20cm')
        ax.set_xscale('symlog',linthresh=.2)
        ax.set_xlim(0,maximum*1.05);ax.set_ylim(0,1)
        ax.set_xlabel('Terminal 3D distance (m; linear to 0.2m, log thereafter)')
        ax.set_ylabel('Cumulative fraction');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
    axes[0].set_title('Assigned-goal distance by PUSH group size')
    axes[1].set_title('Region error and full-box center error')
    fig.suptitle(f'Stage2 {checkpoint_label(directory)} | 512 environments x 3 evaluation seeds | all PUSH attempts included')
    fig.savefig(directory/'push_terminal_distance.png',dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    terminal=summarize_terminal(args.directory)
    write_report(args.directory,terminal)
    plot_terminal(args.directory)
    print(json.dumps(terminal,indent=2),flush=True)


if __name__=='__main__':
    main()

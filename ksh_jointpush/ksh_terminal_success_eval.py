"""Endpoint and uninterrupted success from complete formal evaluation traces (CPU)."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'coupled_base' / 'tokenhsi'))
from utils.evaluation_protocol import TASK_NAMES, spatial_success

CRITERIA = ('point', 'region')
METRICS = ('ever', 'terminal', 'continuous', 'geometric_terminal')
TASK_ORDER = ('AT', 'ON_TOP', 'SIT', 'CLIMB', 'PUSH')


def success_history(current, end_step, alive_at_end):
    """Fall-frame successes count for ever; terminal/continuous require survival."""
    frames, n, m, criteria = current.shape
    assert end_step.shape == (n,) and alive_at_end.shape == (n, m)
    assert ((end_step >= 1) & (end_step <= frames)).all()
    in_episode = np.arange(frames)[:, None, None, None] < end_step[None, :, None, None]
    current = current & in_episode
    seen = np.maximum.accumulate(current, axis=0)
    ever = seen[-1]
    broken = (seen & ~current & in_episode).any(axis=0)
    terminal = current[end_step - 1, np.arange(n)] & alive_at_end[..., None]
    continuous = ever & ~broken & alive_at_end[..., None]
    first_step = np.where(ever, current.argmax(axis=0) + 1, 0)
    assert not (terminal & ~ever).any()
    assert not (continuous & ~terminal).any()
    return dict(ever=ever, terminal=terminal, continuous=continuous, first_step=first_step)


@torch.no_grad()
def terminal_success_records(trace, seed):
    packet = np.asarray(trace['packet'])
    roots, boxes = np.asarray(trace['roots']), np.asarray(trace['boxes'])
    n, m = packet.shape[:2]
    frames = len(roots)
    valid, kind, actor = (packet[..., j] for j in range(3))
    valid = valid.astype(bool)
    rows = np.arange(n)[:, None]
    assert np.array_equal(np.sort(actor, axis=1), np.broadcast_to(np.arange(m), (n, m)))
    assert valid.any(axis=1).all()
    assert np.isfinite(roots).all() and np.isfinite(boxes).all()
    if 'predecessor' in trace:
        assert (np.asarray(trace['predecessor']) < 0).all(), 'Independent coupled tasks only'
    falls = np.asarray(trace['fall_step'])
    assert falls.shape == (n, m)
    assert ((falls == -1) | ((falls >= 1) & (falls <= frames))).all()
    task_falls = falls[rows, actor]
    all_fallen = ((task_falls >= 1) | ~valid).all(axis=1)
    end_step = np.where(all_fallen, np.where(valid, task_falls, 0).max(axis=1), frames)
    alive_at_end = valid & ((task_falls < 0) | (task_falls > end_step[:, None]))
    # The saved valid mask includes each person's fall frame, then stays false.
    physical_valid = np.zeros((n, m), dtype=bool)
    physical_valid[rows, actor] = valid
    expected_valid = physical_valid[None] & ((falls[None] < 0)
                      | (np.arange(1, frames + 1)[:, None, None] <= falls[None]))
    assert np.array_equal(trace['valid'], expected_valid), 'Incomplete/reactivated trace'
    packet_t, sizes_t, goals_t = [torch.from_numpy(np.asarray(trace[key]))
                                 for key in ('packet', 'sizes', 'goals')]
    current = np.empty((frames, n, m, 2), dtype=bool)
    geometric_terminal = np.zeros((n, m, 2), dtype=bool)
    for frame in range(frames):
        geometric = spatial_success(packet_t, torch.from_numpy(roots[frame]),
                                    torch.from_numpy(boxes[frame]), sizes_t, goals_t).numpy()
        current[frame] = geometric & expected_valid[frame, rows, actor, None]
        ending = end_step == frame + 1
        geometric_terminal[ending] = geometric[ending]
    result = success_history(current, end_step, alive_at_end)
    assert np.array_equal(result['ever'], trace['ever']), 'Original ever scoring mismatch'
    result['geometric_terminal'] = geometric_terminal
    groups = np.asarray(trace['groups'])
    task_group = groups[rows, actor]
    agents, push_groups = [], []
    for env in range(n):
        _, counts = np.unique(groups[env], return_counts=True)
        layout = '+'.join(map(str, sorted(counts.tolist())))
        common = dict(seed=int(seed), env=env, layout=layout, terminal_step=int(end_step[env]),
                      terminal_reason='all_agents_fallen' if all_fallen[env] else 'horizon')
        for index in np.flatnonzero(valid[env]):
            group = int(task_group[env, index])
            record = dict(common, task_index=int(index), agent=int(actor[env, index]),
                          task=TASK_NAMES[kind[env, index]], group=group,
                          members=int((groups[env] == group).sum()),
                          fall_step=int(task_falls[env, index]), alive_at_terminal=bool(alive_at_end[env, index]))
            for criterion_index, criterion in enumerate(CRITERIA):
                record['first_success_step_' + criterion] = int(result['first_step'][env, index, criterion_index])
                for metric in METRICS:
                    record[metric + '_' + criterion] = bool(result[metric][env, index, criterion_index])
            agents.append(record)
        for group in np.unique(task_group[env, valid[env] & (kind[env] == 4)]):
            selected = valid[env] & (task_group[env] == group)
            assert (kind[env, selected] == 4).all()
            assert selected.sum() == (groups[env] == group).sum()
            record = dict(common, task='PUSH', group=int(group), members=int(selected.sum()))
            for criterion_index, criterion in enumerate(CRITERIA):
                for metric in METRICS:
                    record[metric + '_' + criterion] = bool(result[metric][env, selected, criterion_index].all())
            push_groups.append(record)
    return agents, push_groups


def success_stats(records, criterion):
    if not records:
        return None
    result = dict(trials=len(records))
    for metric in METRICS:
        key = metric + '_' + criterion
        per_seed = {}
        for seed in sorted({r['seed'] for r in records}):
            selected = [r for r in records if r['seed'] == seed]
            count = sum(r[key] for r in selected)
            per_seed[str(seed)] = dict(trials=len(selected), successes=count, rate=count / len(selected))
        count = sum(r[key] for r in records)
        rates = [s['rate'] for s in per_seed.values()]
        result[metric] = dict(successes=count, pooled_rate=count / len(records), per_seed=per_seed,
                              mean=statistics.mean(rates), std=statistics.stdev(rates) if len(rates) > 1 else None)
    ever = result['ever']['successes']
    for metric in ('terminal', 'continuous'):
        result[metric]['pooled_given_ever'] = result[metric]['successes'] / ever if ever else None
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(directory, summary):
    from ksh_push_terminal_eval import checkpoint_label
    source_path = directory/'source_manifest.json'
    source = json.loads(source_path.read_text()) if source_path.exists() else {}
    conditions = '보상·gate와 독립된 공식 기하 조건으로 채점했다.'
    if 'training_gate_aligned' in source and 'evaluation_gate_aligned' in source:
        label = lambda enabled: '포함' if enabled else '제거'
        conditions = (f"체크포인트 학습 당시 aligned {label(source['training_gate_aligned'])}, "
                      f"평가 rollout 코드에서 aligned {label(source['evaluation_gate_aligned'])}.")
    lines = [f'# {checkpoint_label(directory)}: 종료 성공과 첫 성공 이후 연속 유지', '',
             '기존 512환경 × 평가 seed 0·1·2의 완전한 첫 episode trace를 CPU로 재채점했다. 학습·시뮬레이션 재실행은 없다.', '',
             '- **Ever**: 개인 낙상 프레임까지 한 번이라도 공식 기하 성공 조건 충족.',
             '- **종료 성공**: 환경 종료 프레임에 조건 충족 + 해당 개인이 낙상하지 않음. 중간 이탈 후 재도달도 포함.',
             '- **연속 유지**: 첫 성공 프레임부터 환경 종료까지 매 프레임 성공 + 낙상 없음. 한 프레임 이탈도 실패. 별도 최소 유지 시간 조건은 없음.',
             '- 환경 종료는1200step 또는 유효 인원 전원 낙상 최초step. 실패·낙상 사례도 전체 분모에 포함한다.',
             '- 판정은 기존 float64 점/영역 기하 판정을 그대로 사용한다. 보상·contact gate와 독립이다.', '',
             conditions, '',
             '아래 task 수치는 평가 seed별 성공률의 평균 ± 표본 표준편차(%)다. 서로 다른 학습 seed의 산포가 아니다.']
    for criterion, title in [('point', '점'), ('region', '영역')]:
        lines += ['', f'## {title} 기준', '',
                  '| Task | 전체 시도 | Ever (%) | 종료 성공 (%) | 연속 유지 (%) |',
                  '|---|---:|---:|---:|---:|']
        for task in TASK_ORDER:
            data = summary['tasks'][criterion][task]
            if data:
                values = []
                for metric in ('ever', 'terminal', 'continuous'):
                    v = data[metric]
                    spread = '—' if v['std'] is None else f"{v['std'] * 100:.2f}"
                    values.append(f"{v['mean'] * 100:.2f} ± {spread}")
                lines.append(f"| {task} | {data['trials']} | " + ' | '.join(values) + ' |')
    for scope, title in [('push_agents_by_members', 'PUSH 개인 구역'),
                         ('push_groups_by_members', 'PUSH 그룹 전원')]:
        lines += ['', f'## {title}', '',
                  '세 평가 seed를 합친 성공 수/전체 시도와 비율이다. 단독에는1+3·2+1+1 등 혼합 장면의1명 PUSH도 포함한다.', '',
                  '| 인원 | 전체 시도 | Ever | 종료 성공 | 연속 유지 | 종료/Ever |',
                  '|---|---:|---:|---:|---:|---:|']
        for members in range(1, 5):
            data = summary[scope][str(members)]
            if not data:
                continue
            values = [f"{data[key]['successes']}/{data['trials']} ({100 * data[key]['pooled_rate']:.2f}%)"
                      for key in ('ever', 'terminal', 'continuous')]
            ratio = data['terminal']['pooled_given_ever']
            keep = '—' if ratio is None else f'{100 * ratio:.2f}%'
            lines.append(f"| {members} | {data['trials']} | " + ' | '.join(values) + f' | {keep} |')
    lines += ['', '그룹 Ever는 구성원 각각이 한 번 이상 성공한 경우이며 같은 시점일 필요는 없다. 그룹 종료 성공은 전원이 동일한 환경 종료 프레임에 각자 구역 목표를 만족하고 생존한 경우다. 그룹 연속 유지는 각 구성원이 자신의 첫 성공 이후 매 프레임 성공한 경우다.', '',
              '기존 종료 거리의 최종20cm 비율은 낙상 이후 위치도 포함한 순수 기하 지표다. 이번 종료 성공은 개인 생존도 요구한다. 차이를 확인할 수 있도록 `geometric_terminal` 수치도 JSON/CSV에 별도 저장한다.', '',
              '[집계 JSON](terminal_success_summary.json) · [개인 원시 CSV](terminal_success_agents.csv) · [PUSH 그룹 CSV](terminal_success_push_groups.csv) · [검증](terminal_success_verification.json)', '']
    (directory / 'TERMINAL_SUCCESS.md').write_text('\n'.join(lines))


def summarize(directory):
    from ksh_push_terminal_eval import write_csv
    torch.set_num_threads(1)
    paths = sorted(directory.glob('general_seed*/trace.npz'))
    protected = [p for p in directory.glob('*.json') if not p.name.startswith('terminal_success')]
    protected.extend(p for p in directory.glob('*.csv') if not p.name.startswith('terminal_success'))
    for path in paths:
        protected.extend([path, path.parent / 'result.json', path.parent / 'initial.pt'])
    before = {str(p.relative_to(directory)): sha256(p) for p in protected}
    agents, groups, sources = [], [], []
    for path in paths:
        run = json.loads((path.parent / 'result.json').read_text())
        assert run['protocol'] == 'general'
        with np.load(path) as trace:
            assert len(trace['roots']) == run['simulated_steps']
            assert run['simulated_steps'] == run['horizon'] or (trace['fall_step'] >= 1).all()
            a, g = terminal_success_records(trace, run['seed'])
        for criterion in CRITERIA:
            for task in TASK_NAMES:
                data = success_stats([r for r in a if r['task'] == task], criterion)
                original = run['scores'][criterion][task]
                assert (data['trials'] if data else 0) == original['trials']
                assert (data['ever']['successes'] if data else 0) == original['success']
        agents.extend(a)
        groups.extend(g)
        sources.append(dict(seed=run['seed'], trace=str(path), frames=run['simulated_steps']))
        print(f"seed {run['seed']}: {len(a)} tasks, {len(g)} PUSH groups; ever exact", flush=True)
    assert sorted(s['seed'] for s in sources) == [0, 1, 2]
    summary = dict(sources=sources, task_trials=len(agents), push_group_trials=len(groups),
                   definitions=dict(terminal='Geometry at scene end AND no personal fall by that frame',
                                    continuous='Every frame from first success to scene end succeeds AND no personal fall',
                                    ever='At least one geometric success at/before personal fall, including fall frame',
                                    denominator='All initial task trials, including failures and falls'),
                   tasks={c: {task: success_stats([r for r in agents if r['task'] == task], c)
                              for task in TASK_ORDER} for c in CRITERIA},
                   push_agents_by_members={str(k): success_stats([r for r in agents if r['task'] == 'PUSH' and r['members'] == k], 'point') for k in range(1, 5)},
                   push_groups_by_members={str(k): success_stats([r for r in groups if r['members'] == k], 'point') for k in range(1, 5)})
    write_csv(directory / 'terminal_success_agents.csv', agents)
    write_csv(directory / 'terminal_success_push_groups.csv', groups)
    (directory / 'terminal_success_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    write_report(directory, summary)
    after = {name: sha256(directory / name) for name in before}
    assert before == after, 'Existing raw data or results changed'
    verification = dict(raw_and_existing_json_unchanged=True, original_ever_exact=True,
                        continuous_subset_terminal_subset_ever=True, task_trials=len(agents),
                        push_group_trials=len(groups), inputs_sha256=before,
                        postprocessor_sha256=sha256(Path(__file__)), simulation_rerun=False)
    (directory / 'terminal_success_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    report_path = directory / 'REPORT.md'
    if report_path.exists():
        report = report_path.read_text()
        if '(TERMINAL_SUCCESS.md)' not in report:
            report_path.write_text(report + '\n[종료 성공·연속 유지 성공률](TERMINAL_SUCCESS.md)\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    summarize(args.directory.resolve())

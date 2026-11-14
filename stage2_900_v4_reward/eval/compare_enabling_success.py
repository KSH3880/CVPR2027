import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tokenhsi'))
from utils.evaluation_protocol import INTERACTION_METRICS, TASK_NAMES, summarize_interaction_groups


def compare(source, output):
    original, simultaneous, details = [], [], []
    for path in sorted(source.glob('general_seed*/result.json')):
        data = json.loads(path.read_text())
        previous_check = json.loads((path.parent / 'rescore.json').read_text())
        if not previous_check['cpu_gpu_exact'] or not previous_check['json_counts_exact']:
            raise ValueError('Source trace has not passed CPU verification: ' + str(path))
        before = data['interaction_group_scores']
        after = copy.deepcopy(before)
        with np.load(path.parent / 'trace.npz') as trace:
            packet, parent = trace['packet'], trace['predecessor']
            both, same = trace['both_ever'], trace['simultaneous']
        followers = (parent >= 0) & packet[..., 0].astype(bool)
        if np.any(same & ~both):
            raise ValueError('Simultaneous successes must be a subset of both-ever successes')
        row = dict(seed=data['seed'], run=str(path.parent), criteria={})
        for criterion, index in (('point', 0), ('region', 1)):
            tasks = {}
            for kind, name in enumerate(TASK_NAMES):
                mask = followers & (packet[..., 1] == kind)
                counts = dict(trials=int(mask.sum()),
                              both_ever_success=int(both[..., index][mask].sum()),
                              simultaneous_success=int(same[..., index][mask].sum()))
                recorded = data['scores'][criterion][name]
                if any(recorded[key] != value for key, value in counts.items()
                       if key != 'trials') or recorded['pair_trials'] != counts['trials']:
                    raise ValueError('Trace and recorded pair counts differ')
                if counts['trials']:
                    tasks[name] = counts
            count = int(followers.sum())
            old_success = int(both[..., index][followers].sum())
            new_success = int(same[..., index][followers].sum())
            recorded_success = {'both_ever': old_success, 'simultaneous': new_success}[
                data.get('enabling_success', 'both_ever')]
            expected = dict(groups=count, success=recorded_success, rate=recorded_success / count)
            if before[criterion][INTERACTION_METRICS[1]] != expected:
                raise ValueError('Original enabling aggregate differs')
            before[criterion][INTERACTION_METRICS[1]] = dict(
                groups=count, success=old_success, rate=old_success / count)
            before[criterion][INTERACTION_METRICS[3]]['rate'] = sum(
                before[criterion][name]['rate'] for name in INTERACTION_METRICS[:3]) / 3
            after[criterion][INTERACTION_METRICS[1]] = dict(
                groups=count, success=new_success, rate=new_success / count)
            after[criterion][INTERACTION_METRICS[3]]['rate'] = sum(
                after[criterion][name]['rate'] for name in INTERACTION_METRICS[:3]) / 3
            dropped = followers & both[..., index] & ~same[..., index]
            row['criteria'][criterion] = dict(
                tasks=tasks, dropped_pairs=np.argwhere(dropped).tolist())
        original.append(before)
        simultaneous.append(after)
        details.append(row)
        print('seed {} verified: saved flags match recorded pair counts'.format(data['seed']), flush=True)
    if not original:
        raise ValueError('No completed evaluations found')
    summaries = dict(both_ever=summarize_interaction_groups(original),
                     simultaneous=summarize_interaction_groups(simultaneous))
    output.mkdir(parents=True, exist_ok=True)
    result = dict(source=str(source), seeds=[row['seed'] for row in details],
                  summaries=summaries, original=original, simultaneous=simultaneous,
                  details=details, verification='Saved CPU-verified flags match all recorded pair counts; simultaneous subset of both-ever')
    (output / 'comparison.json').write_text(json.dumps(result, indent=2))
    lines = ['# Enabling 성공 기준 비교', '',
             '같은 저장 궤적의 Region 성공률(%). 평균±표본 표준편차(ddof=1). GPU·시뮬레이션 재실행 없음.', '',
             '| 지표 | 각각-ever | 같은 스텝 동시 성공 | 변화(pp) |', '| --- | ---: | ---: | ---: |']
    for name in INTERACTION_METRICS:
        a, b = [summaries[mode]['region'][name] for mode in ('both_ever', 'simultaneous')]
        lines.append('| {} | {:.2f}±{:.2f} | {:.2f}±{:.2f} | {:+.2f} |'.format(
            name, a['mean'] * 100, a['std'] * 100, b['mean'] * 100, b['std'] * 100,
            (b['mean'] - a['mean']) * 100))
    lines += ['', '| Seed | Enable 각각-ever | Enable 동시 성공 | Mixed 각각-ever | Mixed 동시 성공 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row, a, b in zip(details, original, simultaneous):
        ea, eb = [v['region'][INTERACTION_METRICS[1]] for v in (a, b)]
        lines.append('| {} | {}/{} ({:.2f}%) | {}/{} ({:.2f}%) | {:.2f}% | {:.2f}% |'.format(
            row['seed'], ea['success'], ea['groups'], ea['rate'] * 100,
            eb['success'], eb['groups'], eb['rate'] * 100,
            a['region'][INTERACTION_METRICS[3]]['rate'] * 100,
            b['region'][INTERACTION_METRICS[3]]['rate'] * 100))
    lines += ['', '동시 성공은 유효한 같은 frame에서 AT와 후행 조건을 모두 만족한 적이 있는 경우다. 시간 순서·유지시간 조건은 추가하지 않는다.',
              '원본 평가 결과와 메인 eval·프로토콜은 변경하지 않았다. Point 기준·행동별 건수·탈락한 쌍의 인덱스는 comparison.json에 저장한다.']
    report = '\n'.join(lines) + '\n'
    (output / 'report.md').write_text(report)
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    compare(args.source.resolve(), args.output.resolve())

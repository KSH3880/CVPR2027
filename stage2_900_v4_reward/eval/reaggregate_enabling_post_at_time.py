from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tokenhsi'))
from utils import evaluation_protocol
from stage2_xy100 import install

install()
torch.set_num_threads(1)
SEEDS = (1, 6, 7)
SETTINGS = ('enable_sit', 'enable_climb', 'enable_ontop')
LABELS = {'tokenhsi': 'TokenHSI (+ PUSH)', 'stage1': 'SCT (26k)',
          'fsm': 'SCT + FSM (existing recording)', 'coopft': 'SCT + Coop. FT (4k)', 'ours': 'Ours (V3 2k)'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def first_step(mask):
    return np.where(mask.any(0), mask.argmax(0)+1, -1)


def completion_steps(success, valid, leaders):
    current = success & valid
    rows = np.arange(current.shape[1])
    return dict(first_at=first_step(current[:, rows, leaders]),
                first_follower=first_step(current[:, rows, 1-leaders]),
                first_group=first_step(current.all(-1)))


def calculate(job, output):
    source = Path(job['source'])
    files = [source/name for name in ('trace.npz', 'table2.json', 'initial.pt')]
    hashes = {str(p): sha(p) for p in files}
    recorded = json.loads((source/'table2.json').read_text())
    with np.load(source/'trace.npz') as data:
        packet, parent = data['packet'], data['predecessor']
        sizes, goals, valid = data['sizes'], data['goals'], data['valid']
        roots, boxes = data['roots'], data['boxes']
    steps, count, people = valid.shape
    if (steps, count, people) != (900, 512, 2):
        raise ValueError('Unexpected Enabling cohort')
    followers = (parent >= 0).argmax(-1)
    leaders = parent[np.arange(count), followers]
    expected = {'enable_sit': 0, 'enable_climb': 1, 'enable_ontop': 3}[job['setting']]
    if not (((parent >= 0).sum(-1) == 1).all()
            and (packet[np.arange(count), leaders, 1] == 2).all()
            and (packet[np.arange(count), followers, 1] == expected).all()):
        raise ValueError('Enabling roles differ from setting')
    current = np.empty_like(valid)
    for start in range(0, steps, 30):
        end = min(start+30, steps)
        width = end-start
        result = evaluation_protocol.spatial_success(
            torch.from_numpy(packet).repeat(width, 1, 1),
            torch.from_numpy(roots[start:end]).reshape(width*count, people, 3),
            torch.from_numpy(boxes[start:end]).reshape(width*count, boxes.shape[2], 7),
            torch.from_numpy(sizes).repeat(width, 1, 1), torch.from_numpy(goals).repeat(width, 1, 1))
        current[start:end] = result[..., 1].reshape(width, count, people).numpy()
    arrays = completion_steps(current, valid, leaders)
    success = arrays['first_group'] >= 1
    if success.sum() != recorded['successes'] or recorded['trials'] != count:
        raise ValueError('Group SR mismatch')
    if (source/'table2_trials.npz').exists():
        cached = source/'table2_trials.npz'
        hashes[str(cached)] = sha(cached)
        with np.load(cached) as data:
            np.testing.assert_array_equal(arrays['first_group'], data['first_success'])
    else:
        cached = source/'first_success.npy'
        hashes[str(cached)] = sha(cached)
        np.testing.assert_array_equal(arrays['first_group'], np.load(cached))
    if np.any(success & ((arrays['first_at'] < 1) | (arrays['first_group'] < arrays['first_at']))):
        raise ValueError('Group success must follow first AT success')
    arrays['post_at_time_s'] = np.where(success, (arrays['first_group']-arrays['first_at'])/30, np.nan)
    means = {key: float(arrays[field][success].mean()/30) if success.any() else None
             for key, field in [('completion_time_s', 'first_group'), ('at_time_s_successful_groups', 'first_at')]}
    if success.any() and not np.isclose(means['completion_time_s'], recorded['completion_time_s'], atol=2e-6, rtol=0):
        raise ValueError('Completion time mismatch')
    row = dict(**job, trials=count, successes=int(success.sum()), group_success_rate_percent=float(success.mean()*100),
               **means, post_at_time_s=float(arrays['post_at_time_s'][success].mean()) if success.any() else None,
               zero_post_at_trials=int((arrays['post_at_time_s'] == 0).sum()),
               at_success_trials=int((arrays['first_at'] >= 1).sum()),
               source_sha256=hashes, first_group_exact=True)
    if success.any() and not np.isclose(row['post_at_time_s'], means['completion_time_s']-means['at_time_s_successful_groups'], atol=1e-12):
        raise ValueError('Conditional means do not sum')
    if any(sha(p) != h for p, h in hashes.items()):
        raise ValueError('Original record changed')
    destination = Path(output)/job['name']
    destination.mkdir(exist_ok=True)
    np.savez_compressed(destination/'trials.npz', **arrays)
    (destination/'result.json').write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
    return row


def stats(values):
    complete = None not in values
    return dict(mean=statistics.mean(values) if complete else None,
                std=statistics.stdev(values) if complete else None, seeds=values)


def summarize(rows, output):
    metrics = ('post_at_time_s', 'completion_time_s', 'at_time_s_successful_groups', 'group_success_rate_percent')
    summary = {}
    for model in LABELS:
        by_setting = {}
        for setting in SETTINGS:
            selected = [next(r for r in rows if (r['model'], r['setting'], r['seed']) == (model, setting, seed)) for seed in SEEDS]
            by_setting[setting] = {metric: stats([r[metric] for r in selected]) for metric in metrics}
            by_setting[setting]['successes'] = [r['successes'] for r in selected]
        avg = {metric: stats([statistics.mean(by_setting[setting][metric]['seeds'][i] for setting in SETTINGS)
                             if all(by_setting[s][metric]['seeds'][i] is not None for s in SETTINGS) else None
                             for i in range(3)]) for metric in metrics}
        summary[model] = dict(by_setting=by_setting, average=avg)
    (output/'summary.json').write_text(json.dumps(dict(seeds=SEEDS, models=summary), indent=2, allow_nan=False)+'\n')
    def fmt(item):
        return 'N/A' if item['mean'] is None else f"{item['mean']:.2f}±{item['std']:.2f}"
    lines = ['# Enabling Post-AT Time 재집계', '',
             '512환경 × seed1/6/7 × 900step·XY100%. 다섯 모델의 기존45개 Enabling 기록을 CPU로 재집계했다.', '',
             '`T_post = (최초 유효 그룹 동시 성공 step − 최초 유효 AT 성공 step)/30`. 그룹 동시 성공 trial만 포함하고 실패는 제외한다. AT 완료 때 후행도 성공하면0초다.', '',
             '행동별 성공 trial 평균 → 해당 seed의 세 행동 동일 가중 평균 → seed 평균±표본표준편차(ddof=1). AT Time은 아래 Group SR에 해당하는 성공 trial에서만 계산하므로 전체 AT 성공 trial의 평균과 다르다.', '',
             '| 모델 | Enabling Interaction Group Success Rate (%) ↑ | Completion Time (s) ↓ | AT Time, successful groups (s) | Post-AT Time (s) ↓ |',
             '|---|---:|---:|---:|---:|']
    for model in LABELS:
        row = summary[model]['average']
        lines.append('| '+LABELS[model]+' | '+' | '.join(fmt(row[k]) for k in ('group_success_rate_percent','completion_time_s','at_time_s_successful_groups','post_at_time_s'))+' |')
    lines += ['', '## 행동별 Post-AT Time (s)', '', '| 모델 | AT→SIT | AT→CLIMB | AT→ON_TOP | Avg. |', '|---|---:|---:|---:|---:|']
    for model in LABELS:
        row = summary[model]
        lines.append('| '+LABELS[model]+' | '+' | '.join(fmt(row['by_setting'][s]['post_at_time_s']) for s in SETTINGS)+' | '+fmt(row['average']['post_at_time_s'])+' |')
    lines += ['', '**해석:** 성공 trial만의 지연이다. 성공률·성공 표본 구성이 다른 모델끼리는 SR·전체 Completion Time과 함께 해석한다. 짧은 값만으로 사전 준비 전략의 인과적 우수성을 증명하지 않는다.', '',
              '**FSM:** 첫 정책 입력의 관측 불일치가 있는 기존 기록의 참고값이며 이 후처리로 수정되지 않는다. TokenHSI는 원래397D 입력을 유지한다.', '',
              '## Seed별 수치', '', '| 모델 | 후행 | seed | 성공/시도 | Post-AT Time (s) | Completion Time (s) |', '|---|---|---:|---:|---:|---:|']
    for row in rows:
        post = 'N/A' if row['post_at_time_s'] is None else f"{row['post_at_time_s']:.6f}"
        time = 'N/A' if row['completion_time_s'] is None else f"{row['completion_time_s']:.6f}"
        lines.append(f"| {LABELS[row['model']]} | {row['setting']} | {row['seed']} | {row['successes']}/{row['trials']} | {post} | {time} |")
    lines += ['', '[원시 통계](summary.json) · [검증](verification.json)']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output_etc/check/enabling_post_at_time_seeds167_20261111')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only: set CUDA_VISIBLE_DEVICES=')
    output = Path(args.output).resolve()
    output.relative_to(ROOT/'output_etc/check')
    output.mkdir(parents=True, exist_ok=True)
    suite = ROOT/'output_etc/check/v3_2000_tables12_seeds167_20261110/table2'
    sources = {m: suite/m for m in ('stage1', 'fsm', 'ours')}
    sources['coopft'] = ROOT.parent/'sct_coopft/output_etc/check/sct_coopft_4000_tables12_seeds167_20261111/table2'
    sources['tokenhsi'] = ROOT.parent/'tokenhsi_push/output_etc/check/tokenhsi_tables_512_seeds167_20261111_154749'
    jobs = []
    for model in LABELS:
        for setting in SETTINGS:
            for seed in SEEDS:
                path = sources[model]/setting/f'general_seed{seed}' if model != 'tokenhsi' else sources[model]/f'seed_{seed}'/setting/f'general_seed{seed}'
                jobs.append(dict(model=model, setting=setting, seed=seed, source=str(path), name=f'{model}_{setting}_seed{seed}'))
    (output/'manifest.json').write_text(json.dumps(dict(jobs=jobs, envs=512, horizon=900, seeds=SEEDS,
        definition='first simultaneous group success minus first AT success', time_step_s=1/30), indent=2)+'\n')
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(calculate, job, str(output)) for job in jobs]):
            row = future.result()
            rows.append(row)
            print(f"{len(rows)}/{len(jobs)} {row['name']}", flush=True)
    rows.sort(key=lambda row: (list(LABELS).index(row['model']), SETTINGS.index(row['setting']), row['seed']))
    summarize(rows, output)
    if any(sha(p) != h for row in rows for p, h in row['source_sha256'].items()):
        raise ValueError('Source changed after reaggregation')
    proof = dict(completed=True, runs=len(rows), trials_per_run=512, first_group_step_exact=True,
                 group_success_rate_unchanged=True, post_at_nonnegative=True, conditional_mean_identity=True,
                 original_record_hashes_unchanged=True, gpu_executions=0)
    (output/'verification.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(output/'report.md', flush=True)


if __name__ == '__main__':
    main()

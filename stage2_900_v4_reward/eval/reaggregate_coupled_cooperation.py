from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tokenhsi'))
from coupled_cooperation_metrics import calculate, sha

LABELS = {'stage1': 'SCT (26k)', 'fsm': 'SCT + FSM (existing recording)',
          'coopft': 'SCT + Coop. FT (4k)', 'ours': 'Ours (V3 2k)'}


def statistics(values, seeds):
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    return dict(pooled_mean=float(valid.mean()), valid=len(valid),
                seed_mean=float(np.mean(seeds)), seed_std=float(np.std(seeds, ddof=1)), seeds=seeds)


def summarize(output, jobs):
    models = {}
    runs = [json.loads((output/j['name']/'result.json').read_text()) for j in jobs]
    for model in LABELS:
        model_runs = [r for r in runs if r['model'] == model]
        variants = {}
        for key in ('20', '50'):
            by_size = {}
            for people in (2, 3, 4):
                selected = [r for r in model_runs if r['people'] == people]
                values = {'tcoop_percent': [], 'jerk': [], 'group_success_rate_percent': []}
                means = {k: [] for k in values}
                for row in selected:
                    with np.load(output/row['name']/'trials.npz') as archive:
                        j = archive[key+'__jerk'].astype(np.float64)
                        c = archive[key+'__tcoop'].astype(np.float64)*100
                        s = (archive['first_success'] >= 1).astype(np.float64)*100
                    for metric, array in [('jerk', j[j != 0]), ('tcoop_percent', c),
                                          ('group_success_rate_percent', s)]:
                        values[metric].append(array)
                        means[metric].append(float(np.nanmean(array)))
                by_size[str(people)] = {m: statistics(np.concatenate(values[m]), means[m]) for m in values}
            average = {}
            for metric in values:
                parts = [by_size[str(p)][metric] for p in (2, 3, 4)]
                seeds = np.mean([p['seeds'] for p in parts], axis=0).tolist()
                average[metric] = dict(pooled_mean=float(np.mean([p['pooled_mean'] for p in parts])),
                                       seed_mean=float(np.mean(seeds)), seed_std=float(np.std(seeds, ddof=1)),
                                       seeds=seeds, valid=sum(p['valid'] for p in parts))
            counts = {k: sum(r['variants'][key][k] for r in model_runs)
                      for k in ('no_start', 'nonpositive_duration', 'upper_clipped', 'lower_clipped',
                                'tcoop_nan', 'tcoop_valid', 'jerk_nonzero', 'jerk_zero')}
            variants[key] = dict(by_size=by_size, average=average, counts=counts)
        models[model] = variants
    (output/'summary.json').write_text(json.dumps(models, indent=2, allow_nan=False)+'\n')
    lines = ['# Coupled cooperation: TeamHOI 방식 재집계', '',
             '512환경 × seed1/6/7 × 900step. 4모델 × 2/3/4명 × 3seed의 기존36개 기록만 CPU로 재집계했다.', '',
             '전원 양손–실제 전체 상자 표면 거리<12cm, 힘/추가 생존 필터 없음. t_coop 분자는 성공 후를 포함한 전체900frame이며 분모는 원본 counter 차이, float32 나눗셈·clip·NaN 제외를 따른다. Jerk는 중간 높이 둘레64점·원본 함수·float32·dt0.0333·0 trial 제외다.', '',
             '**FSM은 첫 정책 입력의 reset 관측 불일치가 확인된 기존 기록의 참고값이다. 이 재집계는 그 rollout 문제를 수정하지 않는다.**', '',
             '첫 성공은 실제 공유 중심3D20cm 및 원래 참여자 중 한 명 이상 유효 조건으로 재검증했다. 성공/실패 모두 지표에 포함하며 Group Success Rate는 기존 결과와 같다.', '']
    for key in ('20', '50'):
        lines += [f'## 출발 변위 {key}cm', '',
                  '주 표: 3seed의 trial을 합친 인원별 평균 → 2/3/4명 동일 가중 평균.', '',
                  '| 모델 | Coupled Interaction Group Success Rate (%) ↑ | t_coop (%) ↑ | Mean Absolute Jerk (m/s³) ↓ |',
                  '|---|---:|---:|---:|']
        for model in LABELS:
            avg = models[model][key]['average']
            numbers = [avg[m]['pooled_mean'] for m in ('group_success_rate_percent', 'tcoop_percent', 'jerk')]
            lines.append('| '+LABELS[model]+' | '+' | '.join(f'{v:.2f}' for v in numbers)+' |')
        lines += ['', '### Seed 평균±표본표준편차', '',
                  '| 모델 | 인원 | Group Success Rate (%) | t_coop (%) | Mean Absolute Jerk (m/s³) |',
                  '|---|---:|---:|---:|---:|']
        for model in LABELS:
            for people in ('2', '3', '4', 'average'):
                part = models[model][key]['average'] if people == 'average' else models[model][key]['by_size'][people]
                formatted = [f"{part[m]['seed_mean']:.2f}±{part[m]['seed_std']:.2f}"
                             for m in ('group_success_rate_percent', 'tcoop_percent', 'jerk')]
                lines.append('| '+LABELS[model]+' | '+people+' | '+' | '.join(formatted)+' |')
        lines += ['', '### 유효 표본·clip', '',
                  '| 모델 | 전체 trial | 미출발 | 분모≤0 | t_coop NaN | 상한 clip (%) | Jerk 유효 | Jerk=0 제외 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for model in LABELS:
            c = models[model][key]['counts']
            lines.append(f"| {LABELS[model]} | 4608 | {c['no_start']} | {c['nonpositive_duration']} | {c['tcoop_nan']} | {100*c['upper_clipped']/4608:.2f} | {c['jerk_nonzero']} | {c['jerk_zero']} |")
        lines += ['']
    lines += ['## Seed별 결과', '', '| 모델 | 인원 | seed | 시작(cm) | Group Success Rate (%) | t_coop (%) | Mean Absolute Jerk (m/s³) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in runs:
        for key in ('20', '50'):
            v = row['variants'][key]
            lines.append(f"| {LABELS[row['model']]} | {row['people']} | {row['seed']} | {key} | {row['group_success_rate_percent']:.2f} | {v['tcoop_percent']:.2f} | {v['jerk']:.2f} |")
    lines += ['', 'Trial 합산 평균과 seed 평균±표준편차는 유효 trial 수 차이로 다를 수 있다. 20cm/50cm 결과를 섞지 않는다. t_coop은 구간 내 순수 접촉 비율이나 힘 동기화가 아니며 낮은 Jerk는 SR과 함께 해석한다.', '',
              '[정확한 수치](summary.json) · [완료 검증](verification.json) · [적용 프로토콜](reference/coupled_push_tcoop_jerk_protocol.md)']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output_etc/check/coupled_cooperation_four_models_20261111')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Set CUDA_VISIBLE_DEVICES= for CPU reaggregation')
    output = Path(args.output).resolve()
    output.relative_to(ROOT/'output_etc/check')
    output.mkdir(parents=True, exist_ok=True)
    reference = output/'reference'
    reference.mkdir(exist_ok=True)
    parent = ROOT.parent/'stage1_2_base_jointpush'
    original = parent/'output_etc/check/teamhoi_noforce_reaggregate_20261111/teamhoi_reference'
    references = {}
    for name in ('task_util.py', 'torch_utils.py', 'log_process.py'):
        shutil.copyfile(original/name, reference/name)
        references[str(original/name)] = sha(original/name)
    protocol = parent/'coupled_push_tcoop_jerk_protocol.md'
    shutil.copyfile(protocol, reference/protocol.name)
    references[str(protocol)] = sha(protocol)
    suite = ROOT/'output_etc/check/v3_2000_tables12_seeds167_20261110/table2'
    sources = {m: suite/m for m in ('stage1', 'fsm', 'ours')}
    sources['coopft'] = ROOT.parent/'sct_coopft/output_etc/check/sct_coopft_4000_tables12_seeds167_20261111/table2'
    jobs = [dict(model=m, people=p, seed=s, name=f'{m}_{p}h_seed{s}',
                 source=str(sources[m]/f'coupled_{p}/general_seed{s}'))
            for m in LABELS for p in (2, 3, 4) for s in (1, 6, 7)]
    for job in jobs:
        for filename in ('initial.pt', 'cooperation.npz', 'trace.npz', 'table2_trials.npz', 'table2.json'):
            if not (Path(job['source'])/filename).is_file():
                raise FileNotFoundError(str(job))
    (output/'manifest.json').write_text(json.dumps(dict(jobs=jobs, references_sha256=references), indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(calculate, job, str(output), str(reference)) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            print(f"{index}/{len(jobs)} {row['name']}", flush=True)
    runs = summarize(output, jobs)
    if any(sha(path) != digest for path, digest in references.items()):
        raise ValueError('Reference source changed')
    inputs = {p: h for row in runs for p, h in row['source_sha256'].items()}
    if any(sha(path) != digest for path, digest in inputs.items()):
        raise ValueError('Original input changed after reaggregation')
    proof = dict(completed=True, runs=len(runs), variants_per_run=2, trials_per_run=512,
                 first_success_and_group_sr_exact=all(r['first_success_exact'] for r in runs),
                 original_jerk_function_and_independent_mask_check=True,
                 recording_source_hashes_unchanged=True, reference_hashes_unchanged=True,
                 gpu_rollouts=0, fsm_existing_observation_issue_not_corrected=True)
    (output/'verification.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(output/'report.md', flush=True)


if __name__ == '__main__':
    main()

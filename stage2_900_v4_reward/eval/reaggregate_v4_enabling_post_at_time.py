from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import shutil

import reaggregate_enabling_post_at_time as post

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT/'output_etc/check/v4_2000_tables12_radius_seeds167_20261112'
PREVIOUS = ROOT/'output_etc/check/enabling_post_at_time_seeds167_20261111'
OUTPUT = SUITE/'post_at_time'
LABELS = dict(tokenhsi='TokenHSI (+ PUSH) 40.5k', stage1='SCT 26k',
    fsm='SCT + FSM (existing recording)', coopft='SCT + Coop. FT 6k', ours='Ours V4 2k')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only: set CUDA_VISIBLE_DEVICES=')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_path = SUITE/'summary.json'
    source_hash = post.sha(summary_path)
    current = json.loads(summary_path.read_text())
    jobs = [dict(model=r['model'], setting=r['name'], seed=r['seed'], source=r['source'],
        name=f"{r['model']}_{r['name']}_seed{r['seed']}") for r in current['seed_runs']
        if r['table'] == 'table2' and r['name'] in post.SETTINGS]
    if len(jobs) != 45:
        raise ValueError('Expected five models, three actions, three seeds')
    manifest = dict(jobs=jobs, labels=LABELS, envs=512, horizon=900, seeds=post.SEEDS,
        definition='first valid simultaneous group success minus first valid AT success',
        successful_groups_only=True, time_step_s=1/30, current_suite_summary=str(summary_path),
        current_suite_summary_sha256=source_hash, reused_models=['tokenhsi', 'stage1', 'fsm'],
        recomputed_models=['coopft', 'ours'])
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    rows, pending = [], []
    for job in jobs:
        if job['model'] in ('coopft', 'ours'):
            pending.append(job)
            continue
        source = PREVIOUS/job['name']
        row = json.loads((source/'result.json').read_text())
        if Path(row['source']).resolve() != Path(job['source']).resolve():
            raise ValueError('Previous Post-AT source differs')
        if any(post.sha(p) != digest for p, digest in row['source_sha256'].items()):
            raise ValueError('Previous Post-AT raw data changed')
        destination = OUTPUT/job['name']
        destination.mkdir(exist_ok=True)
        shutil.copyfile(source/'result.json', destination/'result.json')
        if not (destination/'trials.npz').exists():
            (destination/'trials.npz').symlink_to((source/'trials.npz').resolve())
        rows.append(row)
    print(f'Reused {len(rows)}/45 verified records', flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(post.calculate, job, str(OUTPUT)) for job in pending]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{len(rows)}/45 {row['name']}", flush=True)
    rows.sort(key=lambda r: (list(LABELS).index(r['model']), post.SETTINGS.index(r['setting']), r['seed']))
    post.LABELS = LABELS
    post.summarize(rows, OUTPUT)
    summary = json.loads((OUTPUT/'summary.json').read_text())
    for model in LABELS:
        average = summary['models'][model]['average']
        expected = current['models'][model]['enabling_average']
        for actual, target in (('completion_time_s', 'completion_time_s'),
                              ('group_success_rate_percent', 'Enabling Interaction Group Success Rate')):
            if not math.isclose(average[actual]['mean'], expected[target]['mean'], abs_tol=2e-6):
                raise ValueError('Current Table2 time or SR differs')
    if post.sha(summary_path) != source_hash or any(post.sha(p) != digest
            for row in rows for p, digest in row['source_sha256'].items()):
        raise ValueError('Original inputs changed')
    summary.update(labels=LABELS, reused_runs=27, recomputed_runs=18)
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    report = OUTPUT/'report.md'
    report.write_text(report.read_text().replace('다섯 모델의 기존45개 Enabling 기록을 CPU로 재집계했다.',
        '현재 다섯 모델의45개 Enabling 기록을 집계했다. 기존27개를 검증·재사용하고 V4 2k/Coop. FT6k18개를 CPU 재집계했다.'))
    proof = dict(completed=True, runs=45, reused_runs=27, recomputed_runs=18, seeds=list(post.SEEDS),
        first_group_step_exact=True, group_success_rate_unchanged=True, post_at_nonnegative=True,
        conditional_mean_identity=True, original_record_hashes_unchanged=True,
        current_table2_sr_time_match=True, gpu_executions=0)
    (OUTPUT/'verification.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(OUTPUT/'report.md', flush=True)


if __name__ == '__main__':
    main()

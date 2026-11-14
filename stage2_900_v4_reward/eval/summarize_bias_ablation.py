import argparse
import json
from pathlib import Path
import statistics

from run_bias_ablation import PREVIOUS


def stats(values):
    return dict(mean=statistics.mean(values), std=statistics.stdev(values), seeds=values)


def summarize(out):
    manifest = json.loads((out/'manifest.json').read_text())
    seeds = manifest['seeds']
    tables, per_seed = {}, []
    for mode, scene, sizes, model, metric in (
        ('task', 'independent', (1, 2, 4, 8, 16), 'stage1', 'Independent Interaction Success Rate'),
        ('dependency', 'mixed', (2, 4, 8, 16), 'ours', 'Mixed Interaction Group Success Rate')):
        tables[mode] = {}
        for people in sizes:
            name = scene+'_'+str(people)+'h'
            folder = out/mode/name
            proofs = [folder/f'verification_seed{seed}.json' for seed in seeds]
            baseline_paths = [(out/'none'/name if people == 1 else PREVIOUS/'table1'/model/name)/f'general_seed{seed}' for seed in seeds]
            if not all(path.exists() for path in proofs):
                continue
            if people == 1 and not all((out/'none'/name/f'verification_seed{seed}.json').exists() for seed in seeds):
                continue
            baseline, ablated = [], []
            for seed, source in zip(seeds, baseline_paths):
                record = json.loads((folder/f'general_seed{seed}/result.json').read_text())
                previous = json.loads((source/'result.json').read_text())
                for key in ('seed', 'num_agents', 'num_objects', 'num_envs', 'horizon', 'enabling_success', 'xy_extent_percent'):
                    if record[key] != previous[key]:
                        raise ValueError('Baseline configuration differs: '+key)
                for key, values in record['interaction_group_scores']['region'].items():
                    if 'groups' in values and values['groups'] != previous['interaction_group_scores']['region'][key]['groups']:
                        raise ValueError('Group sample counts differ')
                current = record['interaction_group_scores']['region'][metric]['rate']*100
                original = previous['interaction_group_scores']['region'][metric]['rate']*100
                ablated.append(current)
                baseline.append(original)
                per_seed.append(dict(mode=mode, agents=people, seed=seed,
                    baseline_percent=original, ablated_percent=current, delta_pp=current-original,
                    interaction_group_scores=record['interaction_group_scores'], errors=record['errors'],
                    proof=record['bias_ablation'], baseline_source=str(source.resolve())))
            tables[mode][str(people)] = dict(baseline=stats(baseline), ablated=stats(ablated),
                paired_delta_pp=stats([b-a for a,b in zip(baseline, ablated)]))
    status = json.loads((out/'status.json').read_text())
    completed = status['completed'] and len(tables['task']) == 5 and len(tables['dependency']) == 4
    data = dict(completed=completed, seeds=seeds, envs=512, horizon=900, xy_extent_percent=100,
        inference_only=True, tables=tables, per_seed=per_seed,
        source_hashes_unchanged=status.get('source_hashes_unchanged', False))
    (out/'summary.json').write_text(json.dumps(data, indent=2))

    def fmt(item):
        return f"{item['mean']:.2f} ± {item['std']:.2f}"

    lines = ['# Table 3 · 추론 시 관계 bias 제거', '',
        '512환경 × seed '+', '.join(map(str,seeds))+' ·900step·XY100%·M=2N. 평균 ± 표본표준편차.',
        '원본 pth·관계 할당·공유 임베딩·feature fusion을 유지하고 attention bias 출력만0으로 만든다. 재학습 없음.', '',
        '## (a) Stage1 · Independent Interaction Success Rate (%)', '',
        '| 방법 | 1H | 2H | 4H | 8H | 16H |', '|---|---:|---:|---:|---:|---:|']
    for label, key in [('SCT (Stage1 26k)', 'baseline'), ('w/o Task-Relation Bias (inference)', 'ablated')]:
        values = [fmt(tables['task'][str(n)][key]) if str(n) in tables['task'] else '대기' for n in (1,2,4,8,16)]
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines += ['', '## (b) Stage2 · Mixed Interaction Group Success Rate (%)', '',
              '| 방법 | 2H | 4H | 8H | 16H |', '|---|---:|---:|---:|---:|']
    for label, key in [('Ours (V3 2k)', 'baseline'), ('w/o Dependency-Relation Bias (inference)', 'ablated')]:
        values = [fmt(tables['dependency'][str(n)][key]) if str(n) in tables['dependency'] else '대기' for n in (2,4,8,16)]
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines += ['', '## Seed별 결과', '', '| Ablation | 인원 | Seed | 원본 SR (%) | Bias 제거 SR (%) | 변화(pp) |', '|---|---:|---:|---:|---:|---:|']
    for row in per_seed:
        lines.append(f"| {row['mode']} | {row['agents']} | {row['seed']} | {row['baseline_percent']:.2f} | {row['ablated_percent']:.2f} | {row['delta_pp']:+.2f} |")
    lines += ['', f"전체 완료: {completed}. 검증 {len(status['verified'])}/30회, 실패 {len(status['failed'])}회.",
              '1H SCT baseline3회와 bias 제거27회가 신규 실행이다. 2/4/8/16H baseline은 기존 검증 결과를 재사용한다.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    manifest.update(completed=completed, verified_runs=len(status['verified']), failed=status['failed'])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"Bias ablation: {len(status['verified'])}/30 verified; completed={completed}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    summarize(parser.parse_args().output.resolve())

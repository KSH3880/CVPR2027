import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tokenhsi'))
from utils.evaluation_protocol import INTERACTION_METRICS, summarize_interaction_groups


def report(output, seeds, cache):
    metadata = json.loads((output/'manifest.json').read_text()) if (output/'manifest.json').exists() else {}
    saved = json.loads(cache.read_text())
    reused = [saved['results']['100'][saved['seeds'].index(seed)] for seed in seeds]
    summaries, counts = {}, []
    for people in (2, 4, 8, 16):
        summaries[str(people)] = {}
        for mode in ('independent', 'mixed'):
            folder = output/(mode+'_'+str(people)+'h')
            if mode == 'mixed' and people == 4:
                scores = reused
                source = 'existing XY100 scores'
                for seed, score in zip(seeds, scores):
                    origin = metadata.get('mixed4_source_runs', {}).get(str(seed))
                    original = json.loads((Path(origin)/'result.json').read_text()) if origin else None
                    counts.append(dict(agents=4, mode=mode, seed=seed,
                        interaction_groups={name: value.get('groups') for name, value in score['region'].items()},
                        coupled_by_size=({size: value['groups'] for size, value in original['shared_center_scores'].items()}
                                         if original else None)))
            else:
                manifest = json.loads((folder/'manifest.json').read_text())
                if not manifest['completed'] or manifest['seeds'] != seeds:
                    raise ValueError('Incomplete evaluation: '+str(folder))
                if manifest['checkpoint_sha256'] != metadata.setdefault('checkpoint_sha256', manifest['checkpoint_sha256']):
                    raise ValueError('Checkpoint mismatch: '+str(folder))
                results = [json.loads((folder/('general_seed'+str(seed))/'result.json').read_text())
                           for seed in seeds]
                for row in results:
                    row['comparison_protocol'] = 'cvpr2027_'+mode+'_v1'
                    row['group_sampler'] = ('all_independent_uniform_skills' if mode == 'independent'
                                            else 'sequential_uniform_feasible_I_E_C')
                    if people > 4:
                        row['group_spawn_over4']['sampler'] = row['group_sampler']
                    destination = folder/('general_seed'+str(row['seed']))
                    conditions = json.loads((destination/'new_conditions.json').read_text())
                    conditions.update(scene_mode=mode, group_sampler=row['group_sampler'])
                    (destination/'new_conditions.json').write_text(json.dumps(conditions, indent=2))
                    (destination/'result.json').write_text(json.dumps(row, indent=2))
                scores = [row['interaction_group_scores'] for row in results]
                source = str(folder)
                for row in results:
                    center = row['shared_center_scores']
                    counts.append(dict(agents=people, mode=mode, seed=row['seed'],
                        interaction_groups={name: values.get('groups')
                            for name, values in row['interaction_group_scores']['region'].items()},
                        coupled_by_size={size: value['groups'] for size, value in center.items()}))
            summaries[str(people)][mode] = dict(source=source, scores=scores,
                summary=summarize_interaction_groups(scores))
    data = dict(seeds=seeds, agents=[2, 4, 8, 16], envs=512, horizon=900,
        xy_extent_percent=100, reused_mixed_4h_source=str(cache),
        reused_source_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        summaries=summaries, sampling_counts=counts)
    (output/'summary.json').write_text(json.dumps(data, indent=2))
    lines = ['# V2 2k · 인원수별 Independent / Mixed 평가', '',
        '512환경 × 임시 seed '+', '.join(map(str, seeds))+' · 900step · XY100% · M=2N.',
        'Independent 전용 장면과 Mixed 장면을 각각 평가했다. 4H Mixed는 기존 결과를 재사용했다.', '',
        '| 인원 | Independent Interaction Success Rate (%) | Mixed Interaction Group Success Rate (%) |',
        '| --- | ---: | ---: |']
    for people in (2, 4, 8, 16):
        values = []
        for mode, name in (('independent', INTERACTION_METRICS[0]), ('mixed', INTERACTION_METRICS[3])):
            item = summaries[str(people)][mode]['summary']['region'][name]
            values.append('{:.2f}±{:.2f}'.format(item['mean']*100, item['std']*100))
        lines.append('| {}H | {} | {} |'.format(people, *values))
    for mode, names in (('independent', INTERACTION_METRICS[:1]), ('mixed', INTERACTION_METRICS)):
        if mode == 'mixed':
            lines += ['', '## Mixed 관계별 평균', '', '| 인원 | '+' | '.join(names)+' |',
                      '| --- | '+' | '.join(['---:']*len(names))+' |']
            for people in (2, 4, 8, 16):
                stats = summaries[str(people)][mode]['summary']['region']
                lines.append('| {}H | '.format(people)+' | '.join(
                    '{:.2f}±{:.2f}'.format(stats[name]['mean']*100, stats[name]['std']*100)
                    for name in names)+' |')
        lines += ['', '## '+mode, '', '| 인원 | Seed | '+' | '.join(names)+' |',
                  '| --- | --- | '+' | '.join(['---:']*len(names))+' |']
        for people in (2, 4, 8, 16):
            for seed, scores in zip(seeds, summaries[str(people)][mode]['scores']):
                lines.append('| {}H | {} | '.format(people, seed)+' | '.join(
                    '{:.2f}'.format(scores['region'][name]['rate']*100) for name in names)+' |')
    lines += ['', '## Mixed 샘플 수 (세 seed 합계)', '', '| 인원 | Independent | Enabling | Coupled 2H | Coupled 3H | Coupled 4H |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for people in (2, 4, 8, 16):
        rows = [row for row in counts if row['agents'] == people and row['mode'] == 'mixed']
        if any(row['coupled_by_size'] is None for row in rows):
            continue
        independent, enabling = [sum(row['interaction_groups'][name] for row in rows) for name in INTERACTION_METRICS[:2]]
        coupled = [sum(row['coupled_by_size'].get(str(size), 0) for row in rows) for size in (2, 3, 4)]
        if independent+2*enabling+sum(size*count for size, count in zip((2, 3, 4), coupled)) != 512*people*len(seeds):
            raise ValueError('Interaction group counts do not cover all people')
        lines.append('| {}H | {} | {} | {} | {} | {} |'.format(people, independent, enabling, *coupled))
    lines += ['', '평균±표본표준편차는 seed별 값으로 계산했다. Mixed는 장면 전체의 I/E/C 그룹 성공률을 동일 가중 평균한다.',
              'Enable은 같은 frame 선행·후행 동시 성공, Coupled는 실제 공유 상자 중심의 목표 도달이다.',
              '5명 이상 그룹 중심 간격은 5m이며, 관측용 distractor는 그룹당 2k 토큰의 잔여분으로 배분하고 물리에서 제거했다.',
              '임시 seed와 XY100%는 이번 보조 실행에만 적용했고, 논문 MD·메인 기본 기준은 변경하지 않았다.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--mixed4-cache', type=Path, required=True)
    args = parser.parse_args()
    report(args.output.resolve(), args.seeds, args.mixed4_cache.resolve())

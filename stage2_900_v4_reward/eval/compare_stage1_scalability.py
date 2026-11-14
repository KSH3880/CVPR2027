import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tokenhsi'))
import torch
from utils.evaluation_protocol import INTERACTION_METRICS, summarize_interaction_groups


def read(path):
    return json.loads(path.read_text())


def compare(stage1, full, previous):
    torch.set_num_threads(1)
    cached = read(previous/'summary.json')
    original = read(previous/'manifest.json')
    seeds = cached['seeds']
    summaries, checks, counts = {}, [], []
    stage1_digest = None
    for people in (2, 4, 8, 16):
        summaries[str(people)] = {}
        for mode in ('independent', 'mixed'):
            name = mode+'_'+str(people)+'h'
            folder = stage1/name
            manifest = read(folder/'manifest.json')
            if not manifest['completed'] or manifest['seeds'] != seeds:
                raise ValueError('Incomplete Stage1 evaluation: '+name)
            stage1_digest = stage1_digest or manifest['checkpoint_sha256']
            if manifest['checkpoint_sha256'] != stage1_digest:
                raise ValueError('Stage1 checkpoint differs: '+name)
            verification = read(folder/'verification.json')
            if len(verification) != len(seeds):
                raise ValueError('Incomplete Stage1 verification: '+name)
            rows = [read(folder/('general_seed'+str(seed))/'result.json') for seed in seeds]
            for row in rows:
                policy = row['stage1_policy']
                if row['xy_extent_percent'] != 100 or row['enabling_success'] != 'simultaneous':
                    raise ValueError('Unexpected scoring: '+name)
                if policy['coordination_calls'] or not policy['native_actor_exact'] or not policy['weights_unchanged']:
                    raise ValueError('Stage1 policy verification failed: '+name)
                if row['encoder_attention_checks'] != dict(actor=901, critic=900):
                    raise ValueError('Stage1 attention checks incomplete: '+name)
                checks.append(dict(agents=people, mode=mode, seed=row['seed'],
                    initial_states_exact=True, initial_observation_exact=True,
                    native_stage1_exact=True, coordination_calls=0, xy_extent_percent=100))
                counts.append(dict(agents=people, mode=mode, seed=row['seed'],
                    interaction_groups={key: value.get('groups') for key, value in row['interaction_group_scores']['region'].items()},
                    coupled_by_size={key: value['groups'] for key, value in row['shared_center_scores'].items()}))
            scores = [row['interaction_group_scores'] for row in rows]
            item = dict(stage1=dict(source=str(folder), scores=scores, summary=summarize_interaction_groups(scores)))
            if people <= 4:
                item['full'] = cached['summaries'][str(people)][mode]
            else:
                other = full/name
                saved = read(other/'manifest.json')
                if not saved['completed'] or saved['seeds'] != seeds:
                    raise ValueError('Incomplete Full evaluation: '+name)
                full_rows = [read(other/('general_seed'+str(seed))/'result.json') for seed in seeds]
                if len(read(other/'verification.json')) != len(seeds):
                    raise ValueError('Incomplete Full verification: '+name)
                for row, baseline in zip(full_rows, rows):
                    if row['distractor_attention'] != 'all_observed_tokens' or row['encoder_attention_checks'] != dict(actor=900, critic=900):
                        raise ValueError('Full distractor attention verification failed: '+name)
                    if row['xy_extent_percent'] != 100 or row['enabling_success'] != 'simultaneous':
                        raise ValueError('Unexpected Full scoring: '+name)
                    run = 'general_seed'+str(row['seed'])
                    first = torch.load(folder/run/'initial.pt', map_location='cpu', weights_only=False)
                    second = torch.load(other/run/'initial.pt', map_location='cpu', weights_only=False)
                    if first.keys() != second.keys() or any(not torch.equal(first[key], second[key]) for key in first):
                        raise ValueError('Stage1/Full initial state differs: '+name)
                    for metric in INTERACTION_METRICS[:3]:
                        if row['interaction_group_scores']['region'][metric]['groups'] != baseline['interaction_group_scores']['region'][metric]['groups']:
                            raise ValueError('Stage1/Full group counts differ: '+name)
                full_scores = [row['interaction_group_scores'] for row in full_rows]
                item['full'] = dict(source=str(other), scores=full_scores, summary=summarize_interaction_groups(full_scores))
                if saved['checkpoint_sha256'] != original['checkpoint_sha256']:
                    raise ValueError('Full checkpoint differs: '+name)
            summaries[str(people)][mode] = item
    data = dict(seeds=seeds, envs=512, horizon=900, xy_extent_percent=100,
        stage1_checkpoint_sha256=stage1_digest, full_checkpoint_sha256=original['checkpoint_sha256'],
        distractor_attention='all_observed_tokens', summaries=summaries, sampling_counts=counts,
        reused_full_2h_4h=str(previous), reused_summary_sha256=hashlib.sha256((previous/'summary.json').read_bytes()).hexdigest())
    (stage1/'comparison.json').write_text(json.dumps(data, indent=2))
    (stage1/'comparison_verification.json').write_text(json.dumps(checks, indent=2))

    def value(people, mode, policy, metric):
        stats = summaries[str(people)][mode][policy]['summary']['region'][metric]
        return '{:.2f}±{:.2f}'.format(100*stats['mean'], 100*stats['std'])

    lines = ['# Stage1 26k / Full V2 2k 평가', '',
        '512환경 × 임시 seed '+', '.join(map(str, seeds))+' · 900step · XY100% · M=2N.',
        'Independent와 Mixed를 별도로 평가했다. 평균±표본표준편차, 단위 %. 모든 관측 토큰을 attention에 포함하고 distractor 물리만 제거했다.', '',
        '| 인원 | Stage1 Independent | Full Independent | Stage1 Mixed | Full Mixed |',
        '| --- | ---: | ---: | ---: | ---: |']
    for people in (2, 4, 8, 16):
        values = [value(people, mode, policy, INTERACTION_METRICS[0 if mode == 'independent' else 3])
                  for mode in ('independent', 'mixed') for policy in ('stage1', 'full')]
        lines.append('| {}H | {} |'.format(people, ' | '.join(values)))
    lines += ['', 'Independent = Independent Interaction Success Rate. Mixed = Mixed Interaction Group Success Rate.',
        'Full 2·4H는 기존 결과 재사용, 8·16H는 distractor attention을 보존해 재평가했다. 이전 8·16H의 distractor attention 제외 결과와 구분한다.',
        'Stage1은 원본26k 가중치·정규화를 사용하고 Stage2 coordination·외부 대기 제어를 사용하지 않는다. Enable은 목표 정렬 가상 받침, Coupled는 동일 공유 상자의 개인별 PUSH 구역을 입력한다.']
    lines += ['', '## Mixed 관계별 평균', '', '| 인원 | 모델 | '+' | '.join(INTERACTION_METRICS)+' |',
              '| --- | --- | '+' | '.join(['---:']*len(INTERACTION_METRICS))+' |']
    for people in (2, 4, 8, 16):
        for policy in ('stage1', 'full'):
            values = [value(people, 'mixed', policy, key) for key in INTERACTION_METRICS]
            lines.append('| {}H | {} | {} |'.format(people, policy, ' | '.join(values)))
    for mode, metrics in (('independent', INTERACTION_METRICS[:1]), ('mixed', INTERACTION_METRICS)):
        lines += ['', '## '+mode, '', '| 인원 | 모델 | Seed | '+' | '.join(metrics)+' |',
                  '| --- | --- | --- | '+' | '.join(['---:']*len(metrics))+' |']
        for people in (2, 4, 8, 16):
            for policy in ('stage1', 'full'):
                for seed, score in zip(seeds, summaries[str(people)][mode][policy]['scores']):
                    values = ['{:.2f}'.format(100*score['region'][key]['rate']) for key in metrics]
                    lines.append('| {}H | {} | {} | {} |'.format(people, policy, seed, ' | '.join(values)))
    lines += ['', '## Mixed 샘플 수 (세 seed 합계)', '',
        '| 인원 | Independent | Enabling | Coupled 2H | Coupled 3H | Coupled 4H |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for people in (2, 4, 8, 16):
        rows = [row for row in counts if row['agents'] == people and row['mode'] == 'mixed']
        independent, enabling = [sum(row['interaction_groups'][key] for row in rows) for key in INTERACTION_METRICS[:2]]
        coupled = [sum(row['coupled_by_size'].get(str(size), 0) for row in rows) for size in (2, 3, 4)]
        if independent+2*enabling+sum(size*count for size, count in zip((2, 3, 4), coupled)) != 512*people*len(seeds):
            raise ValueError('Group counts do not cover all people')
        lines.append('| {}H | {} | {} | {} | {} | {} |'.format(people, independent, enabling, *coupled))
    (stage1/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:13]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage1', type=Path)
    parser.add_argument('full', type=Path)
    parser.add_argument('previous', type=Path)
    args = parser.parse_args()
    compare(args.stage1.resolve(), args.full.resolve(), args.previous.resolve())

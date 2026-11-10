import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tokenhsi'))
import torch
from evaluate import summarize
from rescore import rescore
from utils.evaluation_protocol import INTERACTION_METRICS, TASK_NAMES


def report(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    checkpoint = Path(manifest['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    reference = ROOT / manifest['reference']
    checks = []
    for seed in manifest['seeds']:
        path = directory / ('general_seed' + str(seed))
        data = json.loads((path / 'result.json').read_text())
        expected = (seed, 512, 4, 8, 900, 900, 'simultaneous', 'common_intersection')
        actual = tuple(data[key] for key in ('seed', 'num_envs', 'num_agents', 'num_objects',
                       'horizon', 'simulated_steps', 'enabling_success', 'enable_size_mode'))
        if actual != expected:
            raise ValueError('Unexpected evaluation configuration: ' + str(path))
        before = torch.load(reference / path.name / 'initial.pt', map_location='cpu')
        after = torch.load(path / 'initial.pt', map_location='cpu')
        for key in before:
            if key != 'boxes' and not torch.equal(before[key], after[key]):
                raise ValueError('Initial scene differs: ' + key)
        parked = torch.load(path / 'physics_only_reset.pt', map_location='cpu')
        mask = parked['mask']
        if not torch.equal(before['boxes'], parked['observed']):
            raise ValueError('Observed boxes differ from baseline')
        if not torch.equal(before['boxes'][~mask], after['boxes'][~mask]):
            raise ValueError('Required physical boxes differ from baseline')
        observation = torch.load(reference / path.name / 'observed_scene.pt', map_location='cpu')
        if not torch.equal(observation, parked['observed_scene']):
            raise ValueError('Full initial observation differs from baseline')
        physics = json.loads((path / 'physics_only_verification.json').read_text())
        conditions = json.loads((path / 'new_conditions.json').read_text())
        if physics['checked_frames'] != 901 or not conditions['conditions_verified'] or conditions['near50']:
            raise ValueError('Runtime conditions failed')
        score = rescore(path)
        if not score['errors_match_recorded']:
            raise ValueError('CPU/GPU Error differs')
        checks.append(dict(seed=seed, initial_task_states_exact=True, initial_observation_exact=True,
            frames=score['frames'], finite=score['finite'], cpu_gpu_sr_exact=score['cpu_gpu_exact'],
            cpu_gpu_counts_exact=score['json_counts_exact'], cpu_gpu_errors_match=True,
            physical_removal_checks=physics['checked_frames']))
    summary = summarize(directory)
    (directory / 'verification.json').write_text(json.dumps(checks, indent=2))
    lines = ['# Stage2 새 프로토콜 평가', '',
             '{} · epoch {} · GPU{} · 4명·8토큰·512환경×{} seeds·900step·ordered.'.format(
                 manifest['experiment'], manifest['epoch'], manifest['gpu'], len(checks)), '',
             'Enabling은 같은 frame의 AT·후행 동시 성공, Coupled는 실제 공유 상자 중심 성공이다. Independent는 Mixed 장면 안의 solo다.', '',
             '| 지표 (%) | ' + ' | '.join('Seed ' + str(seed) for seed in manifest['seeds']) + ' | Mean ± Std. |',
             '| --- | ' + ' | '.join(['---:'] * (len(checks) + 1)) + ' |']
    for name in INTERACTION_METRICS:
        row = summary['interaction_group_scores']['region'][name]
        lines.append('| ' + name + ' | ' + ' | '.join('{:.2f}'.format(value * 100) for value in row['seeds'])
                     + ' | {:.2f}±{:.2f} |'.format(row['mean'] * 100, row['std'] * 100))
    baseline = ROOT / manifest.get('comparison_summary', 'output_etc/check/stage2_point_region_errors_20261108/saved_512/summary.json')
    previous = json.loads(baseline.read_text())
    lines += ['', '| 지표 (%) | ' + manifest.get('comparison_label', '이전 V1 1.5k')
              + ' | 이번 평가 | 변화(pp) |', '| --- | ---: | ---: | ---: |']
    for name in INTERACTION_METRICS:
        a, b = [source['interaction_group_scores']['region'][name] for source in (previous, summary)]
        lines.append('| {} | {:.2f}±{:.2f} | {:.2f}±{:.2f} | {:+.2f} |'.format(
            name, a['mean'] * 100, a['std'] * 100, b['mean'] * 100, b['std'] * 100,
            (b['mean'] - a['mean']) * 100))
    lines += ['', '| 행동 | Point Error (cm) | Region Error (cm) | 성공 표본 수 |',
              '| --- | ---: | ---: | ---: |']
    for name in TASK_NAMES:
        row = summary['errors']['per_task'][name]
        point, region = row['point_error_cm'], row['region_error_cm']
        lines.append('| {} | {:.2f}±{:.2f} | {:.2f}±{:.2f} | {} |'.format(
            name, point['mean'], point['std'], region['mean'], region['std'], row['successes']))
    lines += ['', 'SR·Error는 seed별 산출 후 평균±표본표준편차다. Error는 Region 성공 trial만 사용하며 Enable은 동시 성공한 쌍의 두 행동, Coupled는 그룹당 한 번 포함한다.',
              '공통 교집합·near50 해제·순차 균등 I/E/C·균등 행동·distractor 관측 유지/물리 제거를 적용했다.',
              'Seed별 기준 snapshot과 초기 과제·물리 작업 상태·관측 전체 exact 및 매 frame 관측/물리 분리, CPU 재채점을 확인했다.',
              'Time·t_coop는 산출하지 않았다. 기존 학습 프로세스와 원본 결과는 변경하지 않았다.']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report(args.directory.resolve())

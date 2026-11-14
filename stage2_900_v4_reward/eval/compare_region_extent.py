import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tokenhsi'))
from env.tasks.multi_agent import edge_ontop_reward
from utils.evaluation_protocol import INTERACTION_METRICS, interaction_group_scores, spatial_success, summarize_interaction_groups


def compare(source, output):
    torch.set_num_threads(1)
    results = {'80': [], '100': []}
    checks = []
    original_geometry = edge_ontop_reward.inner_xy_region_error
    for path in sorted(source.glob('general_seed*/result.json')):
        raw = path.read_bytes()
        recorded = json.loads(raw)
        with np.load(path.parent / 'trace.npz') as trace:
            data = {key: torch.from_numpy(trace[key]) for key in (
                'packet', 'predecessor', 'groups', 'roots', 'boxes', 'sizes', 'goals',
                'valid', 'ever', 'simultaneous', 'shared_center_ever')}
        packet, parent = data['packet'], data['predecessor']
        n, m = parent.shape
        rows = torch.arange(n)[:, None]
        follower = parent >= 0
        for extent, margin in (('80', .1), ('100', 0.)):
            ever = torch.zeros(n, m, 2, dtype=torch.bool)
            same = torch.zeros_like(ever)
            def geometry(point, support, size, margin_fraction):
                return original_geometry(point, support, size, margin)
            with patch.object(edge_ontop_reward, 'inner_xy_region_error', geometry):
                for frame in range(len(data['roots'])):
                    current = spatial_success(packet, data['roots'][frame], data['boxes'][frame],
                        data['sizes'], data['goals']) & data['valid'][frame][..., None]
                    ever |= current
                    same |= current & current[rows, parent.clamp_min(0)] & follower[..., None]
            if extent == '80':
                if not torch.equal(ever, data['ever']) or not torch.equal(same, data['simultaneous']):
                    raise ValueError('80% trace reproduction differs')
            elif not torch.equal(ever[..., 0], data['ever'][..., 0]) or (
                    data['ever'] & ~ever).any() or (data['simultaneous'] & ~same).any():
                raise ValueError('Expanded region lost successes or changed Point success')
            scores = interaction_group_scores(packet[..., 0].bool(), parent, data['groups'],
                ever, data['shared_center_ever'], same)
            if extent == '80' and scores != recorded['interaction_group_scores']:
                raise ValueError('80% group scores differ from recorded results')
            results[extent].append(scores)
        if path.read_bytes() != raw:
            raise ValueError('Source result changed')
        checks.append(dict(seed=recorded['seed'], frames=len(data['roots']),
            baseline_exact=True, point_success_unchanged=True, expanded_success_superset=True,
            result_sha256=hashlib.sha256(raw).hexdigest()))
        print('seed {}: baseline exact; 100% success is a superset'.format(recorded['seed']), flush=True)
    summaries = {key: summarize_interaction_groups(value) for key, value in results.items()}
    output.mkdir(parents=True, exist_ok=True)
    comparison = dict(source=str(source), seeds=[row['seed'] for row in checks],
                      checks=checks, summaries=summaries, results=results,
                      z_tolerance_cm=20, gpu_or_simulation_used=False)
    (output / 'comparison.json').write_text(json.dumps(comparison, indent=2))
    lines = ['# XY 성공 영역 80%·100% 비교', '',
             '같은 checkpoint·512환경×{} seeds의 저장 궤적을 CPU 재채점했다. 높이 ±20cm·AT/PUSH·낙상·Enable 동시 성공 기준은 유지했다.'.format(len(checks)), '',
             '| 지표 (%) | 중앙 80% | 상판 100% | 변화(pp) |', '| --- | ---: | ---: | ---: |']
    for name in INTERACTION_METRICS:
        a, b = [summaries[key]['region'][name] for key in ('80', '100')]
        lines.append('| {} | {:.2f}±{:.2f} | {:.2f}±{:.2f} | {:+.2f} |'.format(
            name, a['mean'] * 100, a['std'] * 100, b['mean'] * 100, b['std'] * 100,
            (b['mean'] - a['mean']) * 100))
    lines += ['', '| Seed | Independent 100% | Enabling 100% | Coupled 100% | Mixed 100% |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row, scores in zip(checks, results['100']):
        lines.append('| {} | '.format(row['seed']) + ' | '.join(
            '{:.2f}'.format(scores['region'][name]['rate'] * 100) for name in INTERACTION_METRICS) + ' |')
    lines += ['', '80%의 기존 SR·최종 ever/simultaneous flag를 정확히 재현했고, Point 성공 불변·100% 성공의 포함 관계를 확인했다.',
              '성공률 비교만 수행했다. Error·메인 평가 기준·프로토콜·원본 결과·정책 궤적은 변경하지 않았다.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    compare(args.source.resolve(), args.output.resolve())

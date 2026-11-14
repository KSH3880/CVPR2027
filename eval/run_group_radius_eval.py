from argparse import Namespace
import json
import math
import os
from pathlib import Path

import torch

import run_group_spacing_ready as ready
from run_group_spacing import Suite, ROOT, NAMES, write, sha
from run_group_spacing_comparison import Comparison

OUTPUT = ROOT/'output_etc/check/table1_group_radius_8r3_16r7p5_env64_seed1_20261112'
MODELS = ('stage1', 'tokenhsi', 'ours')
RADII = {8: 3., 16: 7.5}


class RadiusSuite(Suite):
    def __init__(self):
        super().__init__(Namespace(output=OUTPUT, workers_per_gpu=10, reserve_mib=10000,
            envs=64, models=MODELS))
        self.jobs = [dict(j, radius=RADII[j['people']]) for j in self.jobs if j['seed'] == 1]
        self.references = [dict(j, radius=RADII[j['people']]) for j in self.references if j['seed'] == 1]
        self.extra_hashes = {str(p): sha(p) for p in (Path(__file__), ROOT/'eval/run_group_spacing_comparison.py')}
        previous = self.output/'radius_source_hashes.json'
        if previous.exists() and json.loads(previous.read_text()) != self.extra_hashes:
            raise ValueError('Radius runner sources changed; use a new output folder')
        write(previous, self.extra_hashes)
        self.state.update(expected_runs=12, expected_references=4, pending=16,
            seeds=[1], envs=64, group_spawn_radii_m=RADII)
        self.manifest.update(protocol='table1_fixed_group_radius_pilot', seeds=[1], jobs=self.jobs,
            expected_runs=12, references=4, group_placement='fixed_radius_group_center_ring',
            group_spawn_radii_m=RADII, fsm_independent_reused_from=None,
            known_limitations=['TokenHSI fixed397D input'], extra_source_hashes=self.extra_hashes,
            workers_per_gpu=10, reserve_mib=10000)
        write(self.output/'manifest.json', self.manifest)
        self.radius_ready = True
        self.save()

    def save(self):
        super().save()
        if getattr(self, 'radius_ready', False):
            self.partial()

    def partial(self):
        lines = ['# 그룹 중심 고정 반경 · 64환경 seed1', '',
            f"8H R=3m /16H R=7.5m ·900step·XY100%·M=2N. 검증 완료 {len(self.state['verified'])}/12회.", '',
            '| Method | 8H Independent | 8H Mixed | 16H Independent | 16H Mixed |',
            '|---|---:|---:|---:|---:|']
        for model in MODELS:
            values = []
            for n in (8, 16):
                for mode in ('independent', 'mixed'):
                    job = dict(model=model, name=f'{mode}_{n}h', seed=1)
                    if not self.proof(job).exists():
                        values.append('—');continue
                    result = json.loads((self.destination(job)/'result.json').read_text())
                    metric = 'Independent Interaction Success Rate' if mode == 'independent' else 'Mixed Interaction Group Success Rate'
                    values.append(f"{100*result['interaction_group_scores']['region'][metric]['rate']:.2f}")
            lines.append('| '+NAMES[model]+' | '+' | '.join(values)+' |')
        lines += ['', '단위%. 단일 seed이며 표준편차 없음. TokenHSI는 고정397D 입력 차이를 유지한다.']
        (self.output/'partial_report.md').write_text('\n'.join(lines)+'\n')

    def validate(self, job):
        spacing = Comparison.spacing_metadata(self, job)
        path = self.destination(job)
        initial = torch.load(path/'initial.pt', map_location='cpu', weights_only=False)
        groups = initial['groups'].clone()
        rows, follower = (initial['predecessor'] >= 0).nonzero(as_tuple=True)
        groups[rows, follower] = groups[rows, initial['predecessor'][rows, follower]]
        same = groups[:, :, None] == groups[:, None, :]
        sizes = same.sum(-1, keepdim=True)
        centers = torch.bmm(same.float(), initial['roots'][..., :2])/sizes
        count = (1/sizes.squeeze(-1).float()).sum(-1)
        origin = (centers/sizes).sum(1)/count[:, None]
        radius_error = float(((centers-origin[:, None]).norm(dim=-1)-job['radius']).abs().max())
        expected_gap = 2*job['radius']*math.sin(math.pi/int(count.round().max()))
        if spacing.get('group_spawn_radius') != job['radius'] or radius_error > .002:
            raise ValueError('Configured and actual group radii differ')
        if abs(spacing['minimum_initial_group_gap']-expected_gap) > .002:
            raise ValueError('Group-center ring geometry mismatch')
        if (path/'result.json').exists():
            result = json.loads((path/'result.json').read_text())
            result['comparison_protocol'] = 'cvpr2027_table1_fixed_group_radius_env64'
            write(path/'result.json', result)
        super().validate(job)
        proof = json.loads(self.proof(job).read_text())
        proof.update(group_radius_verified=True, group_spawn_radius_m=job['radius'],
            maximum_radius_error_m=radius_error, group_ring_analytic_gap_verified=True)
        write(self.proof(job), proof)


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU supervisor requires CUDA_VISIBLE_DEVICES=')
    suite = RadiusSuite()
    ready.run(suite, {'running': {}})
    if any(sha(p) != h for p, h in suite.extra_hashes.items()):
        raise ValueError('Radius runner sources changed')
    summary = json.loads((suite.output/'summary.json').read_text())
    summary.update(group_spawn_radii_m=RADII, group_placement='fixed_radius_group_center_ring',
        fsm_independent_reused_from=None)
    write(suite.output/'summary.json', summary)
    proof = json.loads((suite.output/'completion_verification.json').read_text())
    proof.pop('fsm_legacy_first_observation_difference_disclosed', None)
    proof.update(gpus=[6, 7], group_spawn_radii_m=RADII, all_group_radii_verified=True)
    write(suite.output/'completion_verification.json', proof)
    report = (suite.output/'report.md').read_text().replace('그룹 간격 확인', '그룹 중심 고정 반경', 1)
    report = report.replace(' SCT+FSM Independent는 SCT 결과 공유.', '')
    report = report.replace(' SCT+FSM은 기존 reset 시 관측 재계산 동작을 보존한 참고값이며 실제 첫 입력 차이는 각 first_policy_input.json에 기록한다.', '')
    (suite.output/'report.md').write_text(report+'\n그룹 중심 원 반경:8H3m /16H7.5m. GPU6/7만 사용했다.\n')

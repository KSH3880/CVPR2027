from argparse import Namespace
import json
import math
import os
from pathlib import Path

import torch

from run_group_spacing import Suite, ROOT, write, sha
from run_group_spacing_formal import Formal, MODELS, GPUS
from run_group_spacing_comparison import Comparison

OUTPUT = ROOT/'output_etc/check/table1_group_radius_8r3_16r7p5_env512_seeds167_baselines_20261112'
RADII = {8: 3., 16: 7.5}


class RadiusFormal(Formal):
    gap = None

    def __init__(self):
        Suite.__init__(self, Namespace(output=OUTPUT, workers_per_gpu=8, reserve_mib=12000,
            envs=512, models=MODELS, entry=ROOT/'eval/group_spacing_2m_eval.py'))
        self.jobs = [dict(j, radius=RADII[j['people']]) for j in self.jobs]
        self.references = [dict(j, radius=RADII[j['people']]) for j in self.references]
        self.extra_hashes = {str(p): sha(p) for p in (
            Path(__file__).resolve(), ROOT/'eval/run_group_spacing_formal.py',
            ROOT/'eval/run_group_spacing_comparison.py', ROOT/'eval/run_group_spacing_ready.py',
            ROOT/'eval/group_spacing_1m_eval.py', ROOT/'eval/group_spacing_2m_eval.py',
            ROOT.parent/'sct_coopft/tokenhsi/utils/config.py')}
        previous = self.output/'formal_source_hashes.json'
        if previous.exists() and json.loads(previous.read_text()) != self.extra_hashes:
            raise ValueError('Formal radius sources changed; use a new output folder')
        write(previous, self.extra_hashes)
        self.state.update(gpus=list(GPUS), expected_runs=len(self.jobs),
            expected_references=len(self.references), pending=len(self.jobs)+len(self.references),
            envs=512, seeds=[1, 6, 7], group_spawn_radii_m=RADII, gpu5_reserve_mib=16000)
        self.manifest.update(protocol='table1_fixed_group_radius_formal', group_spawn_gap=None,
            group_placement='fixed_radius_group_center_ring', group_spawn_radii_m=RADII,
            gpus=list(GPUS), jobs=self.jobs, expected_runs=len(self.jobs), references=len(self.references),
            people=list(self.people), extra_source_hashes=self.extra_hashes, excluded_models=['ours'],
            coopft_gpu_guard='GPU5 in-memory adapter; GPUs6/7 native', workers_per_gpu=8,
            reserve_mib=12000, gpu5_reserve_mib=16000,
            startup_reservation_mib={'8h': 10000, '16h': 14000})
        write(self.output/'manifest.json', self.manifest)
        self.ready = True
        self.save()

    def save(self):
        Suite.save(self)
        if getattr(self, 'ready', False):
            self.partial()

    def validate(self, job):
        spacing = Comparison.spacing_metadata(self, job)
        path = self.destination(job)
        name = 'initial_before_fsm.pt' if job['model'] == 'fsm' else 'initial.pt'
        initial = torch.load(path/name, map_location='cpu', weights_only=False)
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
            result['comparison_protocol'] = 'cvpr2027_table1_fixed_group_radius_env512'
            write(path/'result.json', result)
        Suite.validate(self, job)
        proof = json.loads(self.proof(job).read_text())
        proof.update(group_radius_verified=True, group_spawn_radius_m=job['radius'],
            maximum_radius_error_m=radius_error, group_ring_analytic_gap_verified=True)
        write(self.proof(job), proof)

    def run(self):
        super().run()
        proof = json.loads((self.output/'completion_verification.json').read_text())
        proof.update(group_spawn_radii_m=RADII, all_group_radii_verified=True)
        write(self.output/'completion_verification.json', proof)
        summary = json.loads((self.output/'summary.json').read_text())
        summary.update(group_spawn_radii_m=RADII, group_placement='fixed_radius_group_center_ring')
        write(self.output/'summary.json', summary)
        report = self.output/'report.md'
        report.write_text(report.read_text()+'\n그룹 중심 원 반경:8H3m /16H7.5m. GPU5/6/7만 사용했다.\n')


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU supervisor requires CUDA_VISIBLE_DEVICES=')
    RadiusFormal().run()

from argparse import Namespace
import json
import os
from pathlib import Path
import shutil
import torch

import run_group_spacing_ready as ready
from run_group_spacing import Suite, ROOT, NAMES, write, sha, summarize

OUTPUT = ROOT/'output_etc/check/table1_group_spacing_env64_seed1_20261111'
PREVIOUS = ROOT/'output_etc/check/table1_no_group_spacing_env64_seeds167_20261111_retry1024'


class Comparison(Suite):
    def __init__(self):
        super().__init__(Namespace(output=OUTPUT,workers_per_gpu=10,reserve_mib=10000))
        self.jobs=[dict(j,spacing=spacing,gap=gap,key=spacing+'/'+j['key'])
                   for spacing,gap in (('native',None),('3m',3.)) for j in self.jobs if j['seed']==1]
        self.references=[dict(j,spacing=spacing,gap=gap,key=spacing+'/'+j['key'])
                         for spacing,gap in (('native',None),('3m',3.)) for j in self.references if j['seed']==1]
        self.state.update(expected_runs=36,expected_references=8,pending=44)
        self.manifest.update(protocol='table1_group_spacing_comparison_pilot',seeds=[1],
            expected_runs=36,references=8,jobs=self.jobs,group_spawn_gap='native or3m',
            group_placement='native agent offsets or3m group-center ring',
            implementation_sha256=sha(__file__),adopted_native_runs=str(PREVIOUS),
            adopted_native_entry_sha256=json.loads((PREVIOUS/'source_hashes.json').read_text())[str(ROOT/'eval/group_spacing_eval.py')])
        for job in self.references+self.jobs:
            folder=self.output/job['spacing']
            (folder/'logs').mkdir(parents=True,exist_ok=True)
            if job['spacing']!='native':
                continue
            source=PREVIOUS/job['model']/job['name']/'general_seed1'
            target=self.destination(job)
            if source.exists() and not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                target.symlink_to(source.resolve(),target_is_directory=True)
            source_proof=PREVIOUS/'checks'/job['model']/job['name']/'seed1.json'
            if source_proof.exists() and not self.proof(job).exists():
                self.proof(job).parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source_proof,self.proof(job))
        for job in self.references+self.jobs:
            if self.proof(job).exists():
                metadata=self.spacing_metadata(job)
                proof=json.loads(self.proof(job).read_text())
                proof['group_spacing']=metadata
                write(self.proof(job),proof)
        write(self.output/'manifest.json',self.manifest)
        self.save()

    def view(self,spacing):
        view=object.__new__(Suite)
        view.__dict__.update(self.__dict__)
        view.output=self.output/spacing
        view.save=self.save
        view.validate=self.validate
        return view

    def destination(self,job):
        return self.output/job['spacing']/job['model']/job['name']/f"general_seed{job['seed']}"

    def proof(self,job):
        return self.output/job['spacing']/'checks'/job['model']/job['name']/f"seed{job['seed']}.json"

    def validate(self,job):
        self.spacing_metadata(job)
        return Suite.validate(self.view(job['spacing']),job)

    def spacing_metadata(self,job):
        path=self.destination(job)
        snapshot=torch.load(path/('initial_before_fsm.pt' if job['model']=='fsm' else 'initial.pt'),
                            map_location='cpu',weights_only=False)
        groups=snapshot['groups'].clone()
        rows,followers=(snapshot['predecessor']>=0).nonzero(as_tuple=True)
        groups[rows,followers]=groups[rows,snapshot['predecessor'][rows,followers]]
        same=groups[:,:,None]==groups[:,None,:]
        centers=torch.bmm(same.float(),snapshot['roots'][...,:2])/same.sum(-1,keepdim=True)
        distances=(centers[:,:,None]-centers[:,None]).norm(dim=-1).masked_fill(same,float('inf'))
        minimum=distances.amin((1,2))
        metadata=json.loads((path/'group_spacing.json').read_text())
        metadata.update(minimum_initial_group_gap=float(minimum.min()),
            scenes_with_group_gap_below_5m=int((minimum<5).sum()),group_center_definition='I singleton; E pair; C group')
        torch.save(dict(group_centers=centers,per_scene_minimum_gap=minimum),path/'natural_spacing.pt')
        write(path/'group_spacing.json',metadata)
        for name in ('result.json','scalability_protocol.json','layout.json'):
            if (path/name).exists():
                value=json.loads((path/name).read_text()); value.update(metadata)
                write(path/name,value)
        return metadata

    def run_job(self,job,gpu):
        return self.view(job['spacing']).run_job(job,gpu)


def summarize_comparison(output,jobs):
    summaries={}
    for spacing in ('native','3m'):
        summarize(output/spacing,[j for j in jobs if j['spacing']==spacing])
        summaries[spacing]=json.loads((output/spacing/'summary.json').read_text())
    write(output/'summary.json',dict(envs=64,seeds=[1],horizon=900,results=summaries))
    lines=['# Table 1 · 강제 이격 해제 /3m 비교', '',
        '64환경 × seed1 × 900step·XY100%·M=2N. 각 조건18회, 총36회. SCT+FSM Independent는 SCT 공유.', '',
        '| Method | Group placement | 8H Independent | 8H Mixed | 16H Independent | 16H Mixed |',
        '|---|---|---:|---:|---:|---:|']
    for model,label in NAMES.items():
        for spacing in ('native','3m'):
            scores=summaries[spacing]['results'][model]
            values=[scores[f'{mode}_{people}h']['mean'] for people in (8,16) for mode in ('independent','mixed')]
            lines.append('| '+label+' | '+spacing+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines+=['','단위%. 단일 seed이며 표준편차를 계산하지 않는다. native는 기존 사람별 spawn을 유지하고 그룹 중심 강제 이동을 해제한다.3m는 기존 그룹 중심 원형 배치의 최소 간격을3m로 바꾼다.',
        '관계·거리·상자 크기·ordered·distractor·성공 기준은 유지하며 배치 재시도 한도는 두 조건 모두1024회다.',
        'TokenHSI의 고정397D 입력 차이와 FSM의 기존 reset 관측 차이는 보존·명시한다. seed별 네 가지 SR·Error는 조건별 summary.json/report.md에 저장한다.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU supervisor requires CUDA_VISIBLE_DEVICES=')
    suite=Comparison()
    adoption=OUTPUT/'adoption_spacing_diagnostic.json'
    previous=json.loads((adoption if adoption.exists() else PREVIOUS/'seed1_adoption_status.json').read_text())
    if not adoption.exists():
        previous['running']={'native/'+key:value for key,value in previous['running'].items()}
    ready.summarize=summarize_comparison
    ready.run(suite,previous)
    if sha(__file__)!=suite.manifest['implementation_sha256']:
        raise ValueError('Comparison implementation changed')

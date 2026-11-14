from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

from run_group_spacing import Suite, ROOT, NAMES, LOCK, write, sha, summarize
from run_group_spacing_comparison import Comparison
from run_group_spacing_ready import wait_reference

OUTPUT = ROOT/'output_etc/check/table1_3m_env512_seeds167_baselines_20261111'
MODELS = ('tokenhsi', 'stage1', 'fsm', 'coopft')
GPUS = (5, 6, 7)


def memory():
    output = subprocess.check_output(['nvidia-smi', '-i', '5,6,7',
        '--query-gpu=index,uuid,memory.free,memory.used', '--format=csv,noheader,nounits'], text=True)
    result = {}
    for line in output.splitlines():
        index, uuid, free, used = [v.strip() for v in line.split(',')]
        result[int(index)] = dict(uuid=uuid, free=int(free), used=int(used), processes={})
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True)
    for line in output.splitlines():
        uuid, pid, size = [v.strip() for v in line.split(',')]
        for row in result.values():
            if uuid == row['uuid'] and size.isdigit():
                row['processes'][int(pid)] = int(size)
    return result


class Formal(Suite):
    people = (8, 16)
    gap = 3.
    output_path = OUTPUT
    runner_path = Path(__file__)

    def __init__(self):
        super().__init__(Namespace(output=self.output_path, workers_per_gpu=8, reserve_mib=12000,
            envs=512, models=MODELS, entry=ROOT/'eval/group_spacing_2m_eval.py'))
        self.jobs = [dict(j, gap=self.gap) for j in self.jobs if j['people'] in self.people]
        self.references = [dict(j, gap=self.gap) for j in self.references if j['people'] in self.people]
        self.extra_hashes = {str(p): sha(p) for p in (
            self.runner_path, ROOT/'eval/run_group_spacing_formal.py', ROOT/'eval/run_group_spacing_comparison.py',
            ROOT/'eval/group_spacing_1m_eval.py', ROOT/'eval/group_spacing_2m_eval.py',
            ROOT.parent/'sct_coopft/tokenhsi/utils/config.py')}
        previous = self.output/'formal_source_hashes.json'
        if previous.exists() and json.loads(previous.read_text()) != self.extra_hashes:
            raise ValueError('Formal evaluation sources changed; use a new output folder')
        write(previous, self.extra_hashes)
        self.state.update(gpus=list(GPUS), expected_runs=len(self.jobs), expected_references=len(self.references),
            pending=len(self.jobs)+len(self.references), envs=512, seeds=[1, 6, 7],
            group_spawn_gap=self.gap, gpu5_reserve_mib=16000)
        self.manifest.update(protocol=f'table1_group_spacing_{self.gap:g}m_formal', group_spawn_gap=self.gap,
            group_placement='group_center_ring', gpus=list(GPUS), jobs=self.jobs,
            expected_runs=len(self.jobs), references=len(self.references), people=list(self.people),
            extra_source_hashes=self.extra_hashes, excluded_models=['ours'],
            coopft_gpu_guard='GPU5 in-memory adapter; GPUs6/7 native',
            workers_per_gpu=8, reserve_mib=12000, gpu5_reserve_mib=16000,
            startup_reservation_mib={'8h': 10000, '16h': 14000})
        write(self.output/'manifest.json', self.manifest)
        self.save()

    def validate(self, job):
        Comparison.spacing_metadata(self, job)
        path = self.destination(job)
        if job['model'] != 'reference':
            result = json.loads((path/'result.json').read_text())
            result['comparison_protocol'] = f'cvpr2027_table1_group_spacing_{self.gap:g}m_env512'
            write(path/'result.json', result)
        return super().validate(job)

    def partial(self):
        placement = f'{self.gap:g}m' if self.gap is not None else '그룹 중심 고정 반경'
        lines = [f'# Table 1 · {placement} · 정식 평가 중간 결과', '',
            f"512환경 × seed1/6/7 · 900step · XY100% · M=2N. 검증 완료 {len(self.state['verified'])}/{len(self.jobs)}회.", '',
            '| Method | '+' | '.join(f'{n}H {mode}' for n in self.people for mode in ('Independent','Mixed'))+' |',
            '|---|'+'---:|'*(2*len(self.people))]
        for model in MODELS:
            values = []
            for people in self.people:
                for mode in ('independent', 'mixed'):
                    source = 'stage1' if model == 'fsm' and mode == 'independent' else model
                    metric = ('Independent Interaction Success Rate' if mode == 'independent'
                              else 'Mixed Interaction Group Success Rate')
                    rates = []
                    for seed in (1, 6, 7):
                        job = dict(model=source, name=f'{mode}_{people}h', seed=seed)
                        if self.proof(job).exists():
                            result = json.loads((self.destination(job)/'result.json').read_text())
                            rates.append(100*result['interaction_group_scores']['region'][metric]['rate'])
                    value = '—' if not rates else f'{statistics.mean(rates):.2f}'
                    if len(rates) > 1:
                        value += f'±{statistics.stdev(rates):.2f}'
                    if rates:
                        value += f' ({len(rates)}/3)'
                    values.append(value)
            lines.append('| '+NAMES[model]+' | '+' | '.join(values)+' |')
        lines += ['', '단위%. 평균±표본표준편차(ddof=1); 괄호는 완료 seed 수.',
            'FSM Independent는 SCT 공유. TokenHSI397D 입력 차이와 FSM 기존 초기 관측 차이를 유지·기록한다.']
        (self.output/'partial_report.md').write_text('\n'.join(lines)+'\n')

    def run(self):
        jobs = self.references+self.jobs
        adoption = self.output/'concurrency_adoption.json'
        adopted = json.loads(adoption.read_text())['running'] if adoption.exists() else {}
        pending = [j for j in jobs if not self.proof(j).exists() and j['key'] not in adopted]
        for job in jobs:
            if self.proof(job).exists():
                target = 'references_verified' if job['model'] == 'reference' else 'verified'
                self.state[target].append(dict(key=job['key'], reused=True))
        self.state['phase'] = 'references_and_ready_evaluations'
        futures = {}
        self.partial()
        with ThreadPoolExecutor(max_workers=24) as pool:
            for job in jobs:
                if job['key'] in adopted and not self.proof(job).exists():
                    detail = dict(adopted[job['key']], adopted=True,
                        reservation_mib=10000 if job['people'] == 8 else 14000)
                    self.state['running'][job['key']] = detail
                    futures[pool.submit(wait_reference, self, job, detail['pid'])] = detail['gpu'], job
            while pending or futures:
                snapshot = memory()
                for gpu in sorted(GPUS, key=lambda g: -snapshot[g]['free']):
                    active = [j for _, (g, j) in futures.items() if g == gpu]
                    while len(active) < 8 and not self.state['failed']:
                        reserved = sum(max(self.state['running'][j['key']]['reservation_mib']-
                            snapshot[gpu]['processes'].get(self.state['running'][j['key']].get('pid'), 0), 0)
                            for j in active)
                        ready = [j for j in pending if j['model'] == 'reference'
                                 or self.proof(dict(j, model='reference')).exists()]
                        ready.sort(key=lambda j: (j['model'] != 'reference', j['seed'],
                            j['mode'] != 'mixed', j['people']))
                        reserve = 16000 if gpu == 5 else 12000
                        job = next((j for j in ready if snapshot[gpu]['free']-reserved-reserve >=
                                    (10000 if j['people'] == 8 else 14000)), None)
                        if job is None:
                            break
                        pending.remove(job)
                        with LOCK:
                            self.state['running'][job['key']] = dict(gpu=gpu,
                                reservation_mib=10000 if job['people'] == 8 else 14000,
                                phase='starting', started_unix=time.time())
                            self.save()
                        futures[pool.submit(self.run_job, job, gpu)] = gpu, job
                        active.append(job)
                done, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED) if futures else (set(), set())
                for future in done:
                    gpu, job = futures.pop(future)
                    with LOCK:
                        detail = self.state['running'].pop(job['key'])
                        try:
                            future.result()
                            target = 'references_verified' if job['model'] == 'reference' else 'verified'
                            self.state[target].append(dict(key=job['key'], **detail))
                        except Exception:
                            self.state['failed'].append(dict(key=job['key'], error=traceback.format_exc()))
                            print(self.state['failed'][-1]['error'], flush=True)
                        self.save()
                    self.partial()
                self.state['pending'] = len(pending)
                self.state['gpu_memory'] = {g: {k: v for k, v in row.items() if k != 'processes'}
                                            for g, row in snapshot.items()}
                self.save()
                if self.state['failed'] and not futures:
                    raise RuntimeError('Queue stopped after failure; other processes were not interrupted')
                if not futures and pending:
                    time.sleep(5)
        if len(self.state['verified']) != len(self.jobs) or len(self.state['references_verified']) != len(self.references):
            raise ValueError('Incomplete formal evaluation')
        summarize(self.output, self.jobs, envs=512)
        report = self.output/'report.md'
        placement = f'그룹 간격{self.gap:g}m' if self.gap is not None else '그룹 중심 고정 반경'
        report.write_text(report.read_text().replace('그룹 간격 확인', f'{placement} · 정식 평가', 1))
        summary = json.loads((self.output/'summary.json').read_text())
        summary.update(group_spawn_gap=self.gap, people=list(self.people), excluded_models=['ours'])
        write(self.output/'summary.json', summary)
        if any(sha(p) != value for p, value in {**self.hashes, **self.extra_hashes}.items()):
            raise ValueError('Sources or checkpoints changed')
        write(self.output/'completion_verification.json', dict(completed=True, new_runs=len(self.jobs), references=len(self.references),
            envs=512, seeds=[1, 6, 7], horizon=900, group_spawn_gap=self.gap, gpus=list(GPUS),
            excluded_models=['ours'], initial_scenes_exact=True, cpu_sr_and_error_verified=True,
            checkpoint_hashes_unchanged=True, fsm_legacy_first_observation_difference_disclosed=True))
        self.manifest['completed'] = True
        write(self.output/'manifest.json', self.manifest)
        self.state.update(completed=True, phase='complete', pending=0)
        self.save()


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU supervisor requires CUDA_VISIBLE_DEVICES=')
    Formal().run()

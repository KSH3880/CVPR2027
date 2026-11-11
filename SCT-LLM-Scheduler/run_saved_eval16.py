"""Run saved raw schedules without retries, repairs or predictive validation gates."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from run_eval16_batch import ROOT, RT, gpu_job, save
from sct_scheduler.client import loads
from sct_scheduler.core import validate_response
from sct_scheduler.executable import validate_executable


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    source=args.source.resolve();base=args.output.resolve()
    assert (ROOT/'output_etc/check').resolve() in base.parents
    original=json.loads((source/'status.json').read_text())
    assert original['api_done']==1536
    reference=Path(original['gpu_output'])/'reference'
    rows=[]
    for seed in (1,6,7):
        for i in range(512):
            folder=source/'api'/f'general_seed{seed}'/f'env_{i:04d}'
            req=loads((folder/'request.json').read_text())
            assert req==loads((reference/f'general_seed{seed}'/'requests'/f'env_{i:04d}'/'request.json').read_text())
            answer=loads((folder/'response.raw.txt').read_text())
            row=dict(seed=seed,environment=i,executable=True,validation_error=None)
            try:validate_response(req,answer)
            except Exception as exc:row['validation_error']=str(exc)
            try:validate_executable(req,answer)
            except Exception as exc:row.update(executable=False,exclusion_reason=str(exc))
            rows.append(row)
    manifest=dict(source=str(source),scenes=rows,evaluated=sum(r['executable'] for r in rows),
        excluded=[r for r in rows if not r['executable']],new_api_calls=0,schedules_modified=False)
    if args.prepare_only:
        print(json.dumps({k:v for k,v in manifest.items() if k!='scenes'},indent=2));return
    # Atomic new output directory also acts as a duplicate-launch claim.
    base.mkdir(parents=True,exist_ok=False)
    gpu=RT/'output_etc/check'/base.name;gpu.mkdir(parents=True,exist_ok=False)
    (base/'api').symlink_to(source/'api',target_is_directory=True)
    save(base/'execution_manifest.json',manifest)
    status=dict(stage='eval',completed=False,gpu_output=str(gpu),source=str(source),
        evaluated_scenes=manifest['evaluated'],excluded_scenes=len(manifest['excluded']),physical_gpu=7,new_api_calls=0)
    save(base/'status.json',status)
    os.environ['SCT_EXECUTION_ONLY']='1'
    try:
        for seed in (1,6,7):
            status.update(seed=seed);save(base/'status.json',status)
            gpu_job(base,seed,512,reference,gpu/'eval'/f'general_seed{seed}')
        status.update(stage='cpu_verification');save(base/'status.json',status)
        with (base/'verification.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'verify_eval16_batch.py'),str(base)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status.update(stage='complete',completed=True);save(base/'status.json',status)
    except Exception as exc:
        status.update(stage='failed',error=str(exc));save(base/'status.json',status);raise


if __name__=='__main__':main()

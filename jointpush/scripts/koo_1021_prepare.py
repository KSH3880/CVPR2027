"""Verify the pinned koo_1021 Stage1BasePush snapshot without using a GPU."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PORT=ROOT/'ports/koo_1021'
def verify(checkpoint=None, mode='test'):
    if checkpoint: checkpoint=Path(checkpoint).expanduser().resolve(strict=True)
    manifest=json.loads((ROOT/'koo_1021_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path=PORT/name
        if 'symlink' in expected:
            if not path.is_symlink() or os.readlink(path)!=expected['symlink']:
                raise ValueError('Source symlink differs: '+name)
        elif not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:
            raise ValueError('Source file differs: '+name)
    # Check original native data and the approved bank before using the new port.
    from prepare import prepare
    prepare()
    import yaml
    task=PORT/'ports/stage1_base_push'
    env=yaml.safe_load((task/'tokenhsi/data/cfg/multi_agent/stage1_base_push.yaml').read_text())['env']
    data=PORT/'tokenhsi/data'
    dataset=yaml.safe_load((task/'tokenhsi/data/dataset_stage1_base_push.yaml').read_text())
    missing=[]
    for skill in ('loco','push'):
        for row in dataset['motions'][skill]:
            if not (task/'tokenhsi/data'/row['file']).exists():missing.append(row['file'])
    if missing:raise FileNotFoundError('Missing loco/PUSH motions: '+repr(missing))
    report=dict(branch=manifest['branch'],commit=manifest['commit'],profile='stage1_base_push',source_entries=len(manifest['files']),agents=2,boxes=4,checkpoint=None,gpu_used=False)
    child_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(task/'tokenhsi')+':'+str(task))
    audit="import yaml; from utils.stage1_push_spec import validate_env,checkpoint_contract; e=yaml.safe_load(open('tokenhsi/data/cfg/multi_agent/stage1_base_push.yaml'))['env']; e['motion_file']='tokenhsi/data/dataset_stage1_base_push.yaml'; validate_env(e); c=checkpoint_contract(e); print('Stage1BasePush contract:',c['observation'],c['amp_input'],c['bank_sha256'])"
    subprocess.run([sys.executable,'-c',audit],cwd=task,env=child_env,check=True)
    if checkpoint:
        checkpoint=Path(checkpoint).expanduser().resolve(strict=True)
        child_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(task/'tokenhsi')+':'+str(task))
        subprocess.run([sys.executable,str(task/'preflight.py'),str(checkpoint),mode],cwd=task,env=child_env,check=True)
        report['checkpoint']=str(checkpoint)
    out=ROOT.parent/'runs/jointpush/koo_1021/preparation.json'
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--mode',choices=['train','test'],default='test')
    a=p.parse_args();verify(a.checkpoint,a.mode)

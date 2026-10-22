"""Prepare exact source paths without changing YAML, bank, assets or checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
PORT=ROOT/'ports/koo_native_y2m'
DATA=PORT/'tokenhsi/data'
MANIFEST=DATA/'dataset_push_native_phase_xy.yaml'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(data_roots=(),checkpoint=None,output=None):
    provenance=json.loads((ROOT/'source_manifest.json').read_text())
    for relative,signature in provenance['files'].items():
        p=PORT/relative
        if 'symlink' in signature:
            if not p.is_symlink() or os.readlink(p)!=signature['symlink']:
                raise ValueError('Source symlink changed: '+relative)
        elif not p.is_file() or sha(p)!=signature['sha256']:
            raise ValueError('Source file changed: '+relative)
    roots=[Path(p).expanduser().resolve() for p in data_roots]
    roots += [PROJECT/'stage2_jointpush/ports/koo_push/tokenhsi/data',Path('/home/hwanhee/koo_cvpr/stage1_fix/tokenhsi/data')]
    motions=yaml.safe_load(MANIFEST.read_text())['motions']
    bank=json.loads((DATA/'push_native_phase_y2m_rsi_bank.json').read_text())
    linked=[];missing=[]
    for skill,rows in motions.items():
        for index,row in enumerate(rows):
            target=DATA/row['file']
            relative=row['file'].split('stage1_push_motion_data_20261020/tokenhsi/data/')[-1]
            if not target.exists():
                source=next((r/relative for r in roots if (r/relative).is_file()),None)
                if source is None:
                    missing.append(row['file']);continue
                if skill=='push' and sha(source)!=bank['contract']['motions'][index]:
                    raise ValueError('PUSH motion content differs from native bank: '+str(source))
                if target.is_symlink():
                    raise ValueError('Dangling data link exists: '+str(target))
                target.parent.mkdir(parents=True,exist_ok=True)
                target.symlink_to(source.resolve())
            linked.append(dict(skill=skill,path=row['file'],source=str(target.resolve())))
    if missing:raise FileNotFoundError('Missing motion data; pass --data-root <tokenhsi/data>: '+json.dumps(missing))
    # Execute source hash contract from its original relative working directory.
    os.chdir(PORT);sys.path.insert(0,str(PORT/'tokenhsi'))
    from utils.native_push_rsi import native_contract
    env=yaml.safe_load((DATA/'cfg/multi_agent/push_native_phase_y2m.yaml').read_text())['env']
    env['motion_file']='tokenhsi/data/dataset_push_native_phase_xy.yaml'
    contract=native_contract(env)
    if contract!=bank['contract']:raise ValueError('Native RSI contract differs')
    result=dict(source_commit=provenance['commit'],source_files_verified=len(provenance['files']),
        motions=linked,native_bank_candidates=len(bank['candidates']['push']),native_bank_contract_matches=True,
        data_manifest_sha256=sha(MANIFEST),bank_sha256=sha(DATA/'push_native_phase_y2m_rsi_bank.json'),
        checkpoint=None,checkpoint_validation='not requested',gpu_used=False)
    if checkpoint:
        import torch
        from utils.relation_task_spec import checkpoint_metadata,check_checkpoint_metadata
        checkpoint=Path(checkpoint).expanduser().resolve()
        weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
        check_checkpoint_metadata(weights,checkpoint_metadata(env['relationReward']))
        saved=weights['relation_experiment_config']
        if any(saved['env'].get(k)!=v for k,v in env.items() if k!='motion_file'):
            raise ValueError('Checkpoint environment differs from frozen Y2m config')
        train=yaml.safe_load((DATA/'cfg/train/rlg/push_native_phase_y2m.yaml').read_text())
        saved_train=json.loads(json.dumps(saved['train']))
        for p in [train['params'],saved_train['params']]:
            for key in ['seed','load_checkpoint','load_path']:p.pop(key,None)
        if train!=saved_train:raise ValueError('Checkpoint train config differs')
        saved_contract=weights['open_push_contract']
        if saved_contract['manifest_sha256']!=sha(MANIFEST) or saved_contract['rsi_bank_sha256']!=result['bank_sha256']:
            raise ValueError('Checkpoint manifest or bank hash differs')
        if saved_contract['push_box']!=env['pushBox'] or weights['relation_task_instance']!=dict(graph=env['relationGraph'],num_agents=2,num_objects=2):
            raise ValueError('Checkpoint box/task/count contract differs')
        result.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),epoch=int(weights['epoch']),
                      frame=int(weights['frame']),checkpoint_validation='metadata/environment/train/data checked; strict network restore remains at runtime')
    if output:
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='motions'},indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',action='append',default=[])
    parser.add_argument('--checkpoint',type=lambda x:Path(x).expanduser().resolve())
    parser.add_argument('--report',type=lambda x:Path(x).expanduser().resolve())
    args=parser.parse_args()
    prepare(args.data_root,args.checkpoint,args.report)

"""Evaluate the public TokenHSI Table policy in a frozen original checkout."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import torch
import yaml
from evaluation.hold_at.metrics import sha256
from evaluation.tokenhsi_native_table.validate import score
from evaluation.climb_paired.run_eval import wait_memory,write_json

ROOT=Path(__file__).resolve().parents[2]
BASELINE=ROOT.parent/'TokenHSI'
CACHE=Path('/home/injesus1010/.cache/huggingface/hub/models--lianganimation--TokenHSI/snapshots/7b040865d11a9ea4eb7afbfd9b2d65f1fbfd6318/tokenhsi')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=CACHE/'ckpt_stage2_objShape_table.pth')
    parser.add_argument('--base-checkpoint',type=Path,default=CACHE/'ckpt_stage1.pth')
    parser.add_argument('--gpu',default='0');parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true',help='4env, same full20s x3; not performance')
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    sources=dict(table=args.checkpoint.resolve(),base=args.base_checkpoint.resolve())
    for path in sources.values():
        if not path.is_file():parser.error(f'Checkpoint missing: {path}')
    table=torch.load(sources['table'],map_location='cpu',weights_only=False)
    assert table['epoch']==40000 and len(table['model'])==102
    assert tuple(table['running_mean_std']['running_mean'].shape)==(265,)
    assert all(torch.isfinite(value).all() for value in table['model'].values())
    assert torch.load(sources['base'],map_location='cpu',weights_only=False)['epoch']==40000
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    out=(args.output or ROOT/'output_etc/check/hold_at_eval'/('tokenhsi_table_native20s_'+stamp+('_smoke' if args.smoke else ''))).resolve()
    if ROOT/'output_etc/check' not in out.parents:parser.error('Output must be below stage1_combined/output_etc/check')
    out.mkdir(parents=True,exist_ok=False)
    code=out/'code';code.mkdir();token=code/'tokenhsi';token.mkdir()
    for path in (BASELINE/'tokenhsi').iterdir():
        if path.name in ['data','__pycache__']:continue
        target=token/path.name
        if path.is_dir():shutil.copytree(path,target,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        else:shutil.copy2(path,target)
    # Dataset/assets remain read-only through existing local links.
    (token/'data').symlink_to(ROOT/'tokenhsi/data',target_is_directory=True)
    for name in ['lpanlib','body_models']:(code/name).symlink_to(ROOT/name,target_is_directory=True)
    (code/'dataset_dynamic_objects').symlink_to(ROOT/'tokenhsi/data/dataset_dynamic_objects',target_is_directory=True)
    # The shared runtime is local infrastructure; the original task/player/run files stay exact.
    scripts=token/'scripts/multi_agent';scripts.mkdir(exist_ok=True)
    for name in ['runtime_env.sh','mps_auto_env.sh']:shutil.copy2(ROOT/'tokenhsi/scripts/multi_agent'/name,scripts/name)
    evaluation=code/'evaluation/tokenhsi_native_table';evaluation.mkdir(parents=True)
    for name in ['public_worker.py','validate.py']:shutil.copy2(Path(__file__).with_name(name),evaluation/name)
    policy=out/'policy';policy.mkdir();shutil.copy2(sources['table'],policy/'checkpoint.pth');shutil.copy2(sources['base'],policy/'base.pth')
    original=out/'original';original.mkdir()
    files=dict(env='tokenhsi/data/cfg/adapt_interaction_skills/amp_humanoid_adapt_carry_box2objs_table.yaml',
        train='tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt_disc3.yaml')
    for key,path in files.items():shutil.copy2(BASELINE/path,original/(key+'.yaml'))
    cfg=yaml.safe_load((original/'env.yaml').read_text());count=4 if args.smoke else 512
    assert cfg['env']['episodeLength']==600 and cfg['env']['box']['build']['objRoot']=='dataset_dynamic_objects'
    assert cfg['env']['box']['build']['objSpecified']=='Table_Circle'
    assert len(list((code/'dataset_dynamic_objects/test/Table_Circle').glob('*/config.json')))==9
    copied={str(path.relative_to(code)):sha256(path) for path in token.rglob('*.py')}
    # Every copied Python source is byte-identical to the untouched public baseline.
    for name,digest in copied.items():assert sha256(BASELINE/name)==digest,name
    hashes={key:sha256(path) for key,path in sources.items()}
    manifest=dict(protocol='tokenhsi_public_native_table_20s_continuous3_v1',policy='public_tokenhsi_table',source_epoch=40000,
        checkpoints={key:dict(source=str(path),sha256=hashes[key]) for key,path in sources.items()},num_envs=count,seed=0,num_repeats=3,seconds=20,
        episode_steps=600,expected_timeout_actions=599,success_distance=.2,upright_dot_threshold=.86615,
        primary_included_repeats=[1,2],native_environment=True,native_policy_observations=265,original_player_get_action=True,
        original_run_and_seed_and_sim_helpers=True,math_attention=True,smoke=args.smoke,baseline_python_sha256=copied,
        trace_worker_sha256=sha256(evaluation/'public_worker.py'))
    write_json(out/'manifest.json',manifest)
    if args.prepare_only:print('PREPARED',out,flush=True);return
    wait_memory(args.gpu,6500,out/'gpu_wait.json')
    command=['bash','-c','. tokenhsi/scripts/multi_agent/runtime_env.sh; exec python "$@"','public_native_table',
        'evaluation/tokenhsi_native_table/public_worker.py','--task','HumanoidAdaptCarryBox2Objs',
        '--cfg_train',str(original/'train.yaml'),'--cfg_env',str(original/'env.yaml'),
        '--motion_file','tokenhsi/data/dataset_loco_sit_carry_climb.yaml','--hrl_checkpoint',str(policy/'base.pth'),
        '--checkpoint',str(policy/'checkpoint.pth'),'--test','--num_envs',str(count),'--headless','--eval','--eval_task','Table_Circle',
        '--seed','0','--output_path',str(out/'measurement')]
    env=os.environ.copy()
    for key in ['CUDA_VISIBLE_DEVICES','CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY']:env.pop(key,None)
    env.update(TOKENHSI_GPU=args.gpu,TOKENHSI_CONDA_ENV='tokenhsi118',PYTORCH_JIT='0',PYTHONDONTWRITEBYTECODE='1',
        TORCH_EXTENSIONS_DIR='/tmp/table_hold_at_torch_extensions',TOKENHSI_NATIVE_PUBLIC_OUTPUT=str(out/'measurement'),PYTHONUNBUFFERED='1',PYTHONPATH=str(code))
    state=dict(status='starting',output=str(out),supervisor_pid=os.getpid(),num_envs=count,seed=0,seconds=20,smoke=args.smoke)
    active=ROOT.parent/'runs/tokenhsi_table_native20s_active.json'
    def state_write(**values):
        state.update(values);write_json(out/'state.json',state)
        if not args.smoke:write_json(active,state)
    try:
        print('START public Table native20s',count,'envs',out,flush=True)
        with (out/'run.log').open('w') as log:
            child=subprocess.Popen(command,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT)
            state_write(status='running',worker_pid=child.pid)
            rc=child.wait()
        if rc:raise RuntimeError(f'Native public Table worker rc={rc}; see {out/"run.log"}')
        metrics_files=list((out/'metrics/checkpoint/Table_Circle').glob('metrics_*.json'));assert len(metrics_files)==1
        metrics=json.loads(metrics_files[0].read_text());result=score(out/'measurement',metrics)
        result.update(policy='public_tokenhsi_table',source_epoch=40000,smoke=args.smoke)
        trace_hashes={str(path.relative_to(out)):sha256(path) for path in (out/'measurement').glob('*.npz')}
        for name,digest in copied.items():assert sha256(code/name)==sha256(BASELINE/name)==digest
        assert sha256(policy/'checkpoint.pth')==sha256(sources['table'])==hashes['table']
        assert sha256(policy/'base.pth')==sha256(sources['base'])==hashes['base']
        for i in range(3):
            import numpy as np
            with np.load(out/f'measurement/initial_{i}.npz') as initial:
                if count==512:assert len(np.unique(initial['model_ids']))==9
        write_json(out/'summary.json',result)
        write_json(out/'validation.json',dict(ok=True,independent_native_rescore=True,all_native_code_byte_exact=True,checkpoints_unchanged=True,
            trace_sha256=trace_hashes,all9_models_present=count==512,model_rms_amp_rms_exact_and_frozen=True))
        lines=['# 공개 TokenHSI Table carry 원본 평가','',f'512환경(test9)·seed0·20초·연속3회. 실제환경수={count}, smoke={args.smoke}.',
            '원본 HumanoidAdaptCarryBox2Objs/env YAML/seed/sim/run/player/get_action/265D 관측 그대로 사용. RTX5090 호환 math attention 및 관측·행동을 바꾸지 않는 기록 wrapper만 추가.',
            '성공은 물체중심–목표3D거리≤20cm AND native up-dot≥0.86615, 첫episode 종료frame포함ever. 600설정step/시간종료599action. 원본루프가 부분reset하며 반복사이 재시드하지 않음.',
            f"대표 repeat1/2: **{result['excluding_first_mean_percent']:.2f} ± {result['excluding_first_repeat_std_pp']:.2f}%** (동일seed 반복표본std).",
            f"3회전체: {result['mean_percent']:.2f} ± {result['repeat_sample_std_pp']:.2f}%.",
            '', '| 반복 | 성공 수 | 성공률 % | 성공 최소오차 cm |','|---|---:|---:|---:|']
        for row in result['repeats']:lines.append(f"| {row['repeat']} | {row['success_count']}/{count} | {row['success_percent']:.4f} | {row['success_precision_cm']:.4f} |")
        lines+=['','배포 Table pth 내부 epoch40000. 원논문83.6%와구분: 공개코드는9×9m목표를사용하며논문조건전체재현으로표현하지않음. 기존우리10k 원본환경평가는84.47±1.52%였으나별도측정이며초기배열동일성을가정하지않음.']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
        write_json(out/'completed.json',dict(rc=0,all_traces_rescored=True,smoke=args.smoke))
        state_write(status='complete',rc=0,worker_pid=None)
        print('NATIVE_PUBLIC_TABLE',result['excluding_first_mean_percent'],'+/-',result['excluding_first_repeat_std_pp'],flush=True)
        print('RESULT',out,flush=True)
    except Exception as exc:state_write(status='failed',error=str(exc));raise


if __name__=='__main__':main()

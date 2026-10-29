"""Compare public Chair carry and our residual8500/10000 on common AT40s."""
import argparse
import copy
import csv
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
import yaml
from evaluation.climb_paired.run_eval import ROOT,write_json,wait_memory
from evaluation.climb_paired.continuous_carry import PROTOCOL,TABLE_PROTOCOL,score_trace
from evaluation.climb_paired.repeat_metrics import repeat_statistics
from evaluation.hold_at.metrics import sha256
from tokenhsi.utils.hold_at_ft import object_catalog

DEFAULT_NN=ROOT/'output/hold_at_adaptation/chair/chair_residual_after_table_20261028_recovery/train/ChairAdaptResidual_28-11-08-36/nn'
TABLE_NN=ROOT/'output/hold_at_adaptation/table/table_adaptation_compare_20261027/residual/train/TableAdaptResidual_27-17-21-38/nn'
PUBLIC_DIR=Path('/home/injesus1010/.cache/huggingface/hub/models--lianganimation--TokenHSI/snapshots/7b040865d11a9ea4eb7afbfd9b2d65f1fbfd6318/tokenhsi')
BASELINE=ROOT.parent/'TokenHSI/tokenhsi'


def verify_case(case,epoch,count,checkpoint_hash,public_hash=None,protocol=PROTOCOL,model_count=27):
    result=json.loads((case/'results.json').read_text())
    assert result['protocol']==protocol and result['num_repeats']==3 and result['num_envs']==count
    assert result['max_steps']==1200 and abs(result['time_limit_seconds']-40)<1e-4
    assert not result['reseed_between_repeats'] and not result['initial_frame_scored']
    assert result['scene_checkpoint_exact'] and result['scene_model_and_rms_unchanged']
    assert result['policy']['source_epoch']==epoch and result['policy']['weights_frozen']
    if public_hash:
        assert result['policy']['checkpoint_sha256']==public_hash and result['public_model_and_rms_unchanged']
        assert result['policy']['observation_size']==265 and result['policy']['strict_model_tensors']==102
    checks=[]
    for i,row in enumerate(result['repeats']):
        trace=case/f'repeat_{i}/trace.npz';check=score_trace(trace)
        assert sha256(trace)==row['trace_sha256']
        assert check==json.loads((trace.parent/'independent_validation.json').read_text())
        assert check['num_envs']==count and check['point_count']==row['point_count']==row['region_count']
        assert check['max_end_frame']==1199
        with np.load(trace.parent/'initial.npz') as initial:
            if count==512:assert len(np.unique(initial['model_ids']))==model_count
        checks.append(check)
    assert sha256(case/'checkpoint.pth')==checkpoint_hash
    assert repeat_statistics(result['repeats'])==result['repeat_statistics']
    write_json(case/'validation.json',dict(ok=True,all_traces_rescored=True,statistics_recomputed=True,checks=checks))
    write_json(case/'completed.json',dict(rc=0,all_traces_rescored=True))
    return result


def main(category='Chair'):
    assert category in ['Chair','Table']
    is_table=category=='Table'
    native_category='Table_Circle' if is_table else 'ArmChair_Normal'
    model_count=9 if is_table else 27
    protocol=TABLE_PROTOCOL if is_table else PROTOCOL
    scene_key='ours10000' if is_table else 'ours8500'
    policy_order=['tokenhsi','ours10000'] if is_table else ['tokenhsi','ours8500','ours10000']
    p=argparse.ArgumentParser(description=f'{category} carry common AT40s continuous3 comparison')
    if not is_table:p.add_argument('--checkpoint8500',type=Path,default=DEFAULT_NN/'ChairAdaptResidual_00008500.pth')
    p.add_argument('--checkpoint10000','--checkpoint',type=Path,default=TABLE_NN/'TableAdaptResidual_00010000.pth' if is_table else DEFAULT_NN/'ChairAdaptResidual_00010000.pth')
    p.add_argument('--public-checkpoint',type=Path,default=PUBLIC_DIR/f'ckpt_stage2_objShape_{category.lower()}.pth')
    p.add_argument('--public-base-checkpoint',type=Path,default=PUBLIC_DIR/'ckpt_stage1.pth')
    p.add_argument('--object-root',type=Path,default=ROOT/'tokenhsi/data/dataset_dynamic_objects')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--gpu',default='0')
    p.add_argument('--output',type=Path);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--smoke',action='store_true',help='4env full40s x3 repeats per policy; not performance')
    args=p.parse_args()
    if args.seed<0:p.error('Nonnegative seed required')
    selected=dict(ours10000=args.checkpoint10000,public=args.public_checkpoint,base=args.public_base_checkpoint)
    if not is_table:selected['ours8500']=args.checkpoint8500
    paths={key:path.resolve() for key,path in selected.items()}
    for path in paths.values():
        if not path.is_file():p.error(f'Checkpoint missing: {path}')
    loaded={key:torch.load(path,map_location='cpu',weights_only=False) for key,path in paths.items()}
    expected_epochs=dict(ours10000=10000,public=40000 if is_table else 10000,base=40000)
    if not is_table:expected_epochs['ours8500']=8500
    for key,epoch in expected_epochs.items():
        assert loaded[key]['epoch']==epoch,(key,loaded[key]['epoch'])
        assert all(torch.isfinite(value).all() for value in loaded[key]['model'].values())
    saved=loaded[scene_key]['relation_experiment_config']
    for key in ['env','sim']:assert saved[key]==loaded['ours10000']['relation_experiment_config'][key]
    assert saved['env']['holdAtFT']['category']==native_category
    assert saved['env']['holdAtFT']['trainable']=='residual_only_actor_frozen_encoders'
    entries=object_catalog(args.object_root,'test',native_category);assert len(entries)==model_count
    count=4 if args.smoke else 512
    cfg=copy.deepcopy({key:saved[key] for key in ['env','sim']})
    cfg['env'].update(numAgents=1,numObjects=1,numEnvs=count,episodeLength=1200)
    cfg['env']['eval'].update(skill=['loco'],skillInitProb=[1.])
    cfg['env']['debug']['reward']=False
    cfg['env']['holdAtFT'].update(asset_split='test',object_root=str(args.object_root.resolve()))
    assert cfg['env']['enableEarlyTermination'] and cfg['env']['contactBodies']==['right_foot','left_foot']
    assert cfg['env']['holdAtFT']['ground_only'] and cfg['env']['holdAtFT']['face_human']==(not is_table)
    train=copy.deepcopy(saved['train']);train['params']['algo']['name']='ma'
    assert train['params']['network']['name']=='amp_table_adaptation'
    train['params']['load_checkpoint']=False;train['params'].pop('load_path',None)
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    out=(args.output or ROOT/'output_etc/check/hold_at_eval'/(category.lower()+'_compare_40s_'+stamp+('_smoke' if args.smoke else ''))).resolve()
    if ROOT/'output_etc/check' not in out.parents:p.error('Output must be below stage1_combined/output_etc/check')
    out.mkdir(parents=True,exist_ok=False)
    pinned={key:out/(key+'.pth') for key in paths}
    for key,path in paths.items():shutil.copy2(path,pinned[key])
    hashes={key:sha256(path) for key,path in pinned.items()}
    for key,path in paths.items():assert hashes[key]==sha256(path)
    originals=out/'original';originals.mkdir()
    originals_map=dict(human_source='env/tasks/humanoid.py',
        carry_source='env/tasks/adapt_interaction_skills/humanoid_adapt_carry_box2objs.py',
        train_config='data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt_disc3.yaml')
    public_cfg={}
    for key,path in originals_map.items():
        target=originals/Path(path).name;shutil.copy2(BASELINE/path,target);public_cfg[key]=str(target)
    public_cfg.update(checkpoint=str(pinned['public']),base_checkpoint=str(pinned['base']),policy_name=f'public_tokenhsi_{category.lower()}_adapt')
    network_files=['learning/amp_network_builder.py','learning/transformer/composer.py',
        'learning/transformer/amp_network_builder_transformer.py','learning/transformer/amp_network_builder_transformer_adapt.py']
    for path in network_files:assert sha256(BASELINE/path)==sha256(ROOT/'tokenhsi'/path),path
    source_files=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'evaluation/hold_at').glob('*.py'))
    source_files+=list((ROOT/'tokenhsi').rglob('*.py'))
    source_files+=[BASELINE/path for path in originals_map.values()]+[BASELINE/'learning/transformer/trans_players.py']
    source_hashes={str(path):sha256(path) for path in source_files}
    code=out/'code';code.mkdir()
    for index,path in enumerate(source_files):shutil.copy2(path,code/f'{index}_{path.name}')
    asset_hashes={str(path):sha256(path) for entry in entries for path in Path(entry['path']).rglob('*') if path.is_file()}
    write_json(out/'asset_sha256.json',asset_hashes)
    (out/'env.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));(out/'train.yaml').write_text(yaml.safe_dump(train,sort_keys=False))
    manifest=dict(protocol=protocol,seed=args.seed,num_envs=count,seconds=40,max_steps=1200,num_repeats=3,
        primary_included_repeats=[1,2],reseed_between_repeats=False,smoke=args.smoke,gpu=args.gpu,
        category=native_category,test_models=model_count,initial_frame_scored=False,policies=policy_order,
        criterion='object-center 3D distance<=0.20m; first-episode ever; no upright/contact/holding/settling requirement',
        checkpoints={key:dict(source=str(paths[key]),pinned=str(pinned[key]),sha256=hashes[key],epoch=int(loaded[key]['epoch'])) for key in paths},
        code_sha256=source_hashes,public_network_baseline_exact=True)
    write_json(out/'manifest.json',manifest)
    if args.prepare_only:print('PREPARED',out,flush=True);return
    active=ROOT.parent/f'runs/{category.lower()}_carry_40s_active.json'
    state=dict(status='starting',supervisor_pid=os.getpid(),output=str(out),seed=args.seed,num_envs=count,seconds=40,smoke=args.smoke,completed_policies=[])
    def state_write(**values):
        state.update(values);write_json(out/'state.json',state)
        if not args.smoke:write_json(active,state)
    state_write()
    results={}
    try:
        for policy in policy_order:
            for path,value in source_hashes.items():assert sha256(path)==value,path
            wait_memory(args.gpu,6500,out/'gpu_wait.json')
            case=out/policy;case.mkdir()
            scene=scene_key if policy=='tokenhsi' else policy
            shutil.copy2(pinned[scene],case/'checkpoint.pth')
            local_cfg=copy.deepcopy(cfg)
            if policy=='tokenhsi':local_cfg['env']['holdAtFT']['public_policy']=public_cfg
            (case/'env.yaml').write_text(yaml.safe_dump(local_cfg,sort_keys=False))
            command=['bash','-c','. tokenhsi/scripts/multi_agent/runtime_env.sh; exec python "$@"','chair_carry_eval',str(Path(__file__).with_name('launch_carry.py')),
                '--task','HumanoidTableHoldAtFT' if is_table else 'HumanoidChairHoldAtFT','--cfg_env',str(case/'env.yaml'),'--cfg_train',str(out/'train.yaml'),
                '--checkpoint',str(case/'checkpoint.pth'),'--motion_file','tokenhsi/data/dataset_table_hold_at.yaml',
                '--num_agents','1','--num_objects','1','--num_envs',str(count),'--test','--eval','--headless','--no_video',
                '--eval_repeats','3','--eval_skills','loco','--eval_skill_probs','1','--task_graph','holding_at',
                '--seed',str(args.seed),'--episode_length','1200','--output_path',str(case)]
            if (ROOT/'tokenhsi/data/dataset_tokenhsi_hold_at.yaml').is_file():command[command.index('tokenhsi/data/dataset_table_hold_at.yaml')]='tokenhsi/data/dataset_tokenhsi_hold_at.yaml'
            env=os.environ.copy()
            for key in ['HOLD_AT_OBJECT_ROOT','CHECK','MAX_ITERATIONS','RESUME_CHECKPOINT','CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','CUDA_VISIBLE_DEVICES']:env.pop(key,None)
            env.update(TOKENHSI_GPU=args.gpu,TOKENHSI_CONDA_ENV='tokenhsi118',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTORCH_JIT='0',TORCH_EXTENSIONS_DIR='/tmp/table_hold_at_torch_extensions')
            print('START',policy,out,flush=True)
            with (case/'run.log').open('w') as log:
                worker=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                state_write(status='running',policy=policy,worker_pid=worker.pid)
                rc=worker.wait()
            if rc:raise RuntimeError(f'{policy}: worker rc={rc}; inspect {case/"run.log"}')
            result=verify_case(case,expected_epochs['public'] if policy=='tokenhsi' else int(loaded[scene]['epoch']),count,hashes[scene],hashes['public'] if policy=='tokenhsi' else None,protocol,model_count)
            results[policy]=result
            for repeat in range(3):
                with np.load(out/f'tokenhsi/repeat_{repeat}/initial.npz') as left,np.load(case/f'repeat_{repeat}/initial.npz') as right:
                    equality={key:bool(np.array_equal(left[key],right[key])) for key in left.files}
                if repeat==0:assert all(equality.values()),(policy,equality)
                write_json(case/f'repeat_{repeat}/initial_comparison_to_tokenhsi.json',equality)
            state_write(completed_policies=list(results),worker_pid=None)
            metric=result['repeat_statistics']['region']
            print('DONE',policy,metric['mean_percent'],'+/-',metric['within_seed_sample_std_percentage_points'],flush=True)
        assert all(sha256(path)==hashes[key] for key,path in pinned.items())
        assert all(sha256(path)==value for path,value in source_hashes.items())
        assert all(sha256(path)==value for path,value in asset_hashes.items())
        write_json(out/'summary.json',dict(protocol=protocol,seed=args.seed,num_envs=count,seconds=40,smoke=args.smoke,
            representative_repeats=[1,2],std_scope='within one seed across repeats1/2, sample std ddof1',
            policies={key:value['repeat_statistics'] for key,value in results.items()}))
        with (out/'repeats.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['policy','seed','repeat','success_count','num_envs','region_percent','included_in_primary_mean'])
            for policy,result in results.items():
                for row in result['repeats']:writer.writerow([policy,args.seed,row['repeat'],row['region_count'],count,100*row['region_ever'],row['repeat'] in [1,2]])
        lines=[f'# {category} HOLD+AT 우리 기준40초 비교','',f'seed={args.seed}, {count}환경, test{model_count}, {native_category}동적물체1개/사람1명, loco시작, deterministic. smoke={args.smoke}.',
            '물체중심과목표점의3D거리≤20cm, 첫episode action후부터종료frame까지ever. AT 점/영역동일. 파지·직립·접촉·손놓기·정지·유지시간추가조건없음.',
            '원본제어루프로시드재설정없이연속3회. 대표repeat1/2 평균±반복표본std, repeat0 원시결과와3회전체통계도보존.',
            f'우리모델은78k기반encoder고정+action residual학습모델. 공개정책은TokenHSI {category.lower()} carry adaptation epoch{expected_epochs["public"]}/265D.',
            '첫반복초기전배열동일검증. 후속반복은정책별부분reset에따른RNG소비로초기장면이달라질수있으며각repeat에실측비교보존. 원본논문조건전체재현은아님.','',
            '| 정책 | repeat0 % | repeat1 % | repeat2 % | 대표 평균±std % | 3회전체 평균±std % |','|---|---:|---:|---:|---:|---:|']
        for policy,result in results.items():
            m=result['repeat_statistics']['region'];rates=m['repeat_percent']
            lines.append(f"| {policy} | {rates[0]:.4f} | {rates[1]:.4f} | {rates[2]:.4f} | {m['mean_percent']:.4f} ± {m['within_seed_sample_std_percentage_points']:.4f} | {m['all_three_mean_percent']:.4f} ± {m['all_three_sample_std_percentage_points']:.4f} |")
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
        write_json(out/'validation.json',dict(ok=True,all_traces_rescored=True,num_traces=len(results)*3,all_policy_first_initial_states_exact=True,
            checkpoint_source_code_and_asset_hashes_verified=True,all_models_and_rms_frozen=True,statistics_recomputed=True))
        write_json(out/'completed.json',dict(rc=0,num_policies=len(results),num_repeats_per_policy=3,smoke=args.smoke))
        state_write(status='complete',rc=0,policy=None,worker_pid=None)
        print('RESULT',out,flush=True)
    except Exception as exc:
        state_write(status='failed',error=str(exc));raise


if __name__=='__main__':main()

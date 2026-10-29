"""One-seed Chair SIT checkpoint evaluation, common region40s, continuous3."""
import argparse
import copy
import csv
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch
import yaml
from evaluation.climb_paired.metrics import score_trace
from evaluation.climb_paired.repeat_metrics import repeat_statistics
from evaluation.climb_paired.run_eval import ROOT,catalog,sha,write_json,wait_memory
from evaluation.public_point.run_eval import PUBLIC


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--policy',choices=['ours','tokenhsi'],default='ours')
    parser.add_argument('--public-checkpoint',type=Path,default=PUBLIC)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--expected-epoch',type=int,help='Optional checkpoint identity check; actual epoch is always recorded')
    parser.add_argument('--seconds',type=int,default=40)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true',help='4-env implementation check, permits non20k structural test checkpoint; never a performance result')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.seed<0 or args.seconds<=0:parser.error('Nonnegative seed and positive seconds required')
    source=args.checkpoint.resolve()
    if not source.is_file():parser.error(f'Checkpoint does not exist: {source}')
    weights=torch.load(source,map_location='cpu',weights_only=False)
    epoch=int(weights['epoch'])
    if args.expected_epoch is not None and epoch!=args.expected_epoch:parser.error(f'Expected epoch{args.expected_epoch}, found {epoch}')
    original30d='a2c_network.actor_encoder.tokenizers.1.0.weight' in weights['model'] and weights['model']['a2c_network.actor_encoder.tokenizers.1.0.weight'].shape[1]==30
    if not original30d:assert weights.get('chair_ft_contract',{}).get('shape_encoder')=='bbox35_mlp'
    assert all(torch.isfinite(tensor).all() for tensor in weights['model'].values())
    policy_epoch=epoch
    public_source=None
    if args.policy=='tokenhsi':
        public_source=args.public_checkpoint.resolve()
        if not public_source.is_file():parser.error(f'Public checkpoint does not exist: {public_source}')
        policy_epoch=int(torch.load(public_source,map_location='cpu',weights_only=False)['epoch'])
    entries=catalog('Chair');assert len(entries)==26
    for entry in entries:
        for name in ['asset.urdf','config.json']:assert (Path(entry['path'])/name).is_file(),entry['path']
    count=4 if args.smoke else 512;steps=args.seconds*30
    cfg=copy.deepcopy({key:weights['relation_experiment_config'][key] for key in ['env','sim']})
    cfg['env'].update(numAgents=1,numObjects=1,numEnvs=count,episodeLength=steps,debug=dict(reward=False))
    cfg['env']['chairFT'].update(asset_split='test',point_evaluation=dict(category='Chair',action='SIT',catalog=entries))
    cfg['env']['asset']['assetRoot']='tokenhsi/data/assets'
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    label=f'chair_epoch{epoch}' if args.policy=='ours' else f'tokenhsi_sit_epoch{policy_epoch}'
    out=(args.output or ROOT/'output_etc/check/chair_sit_eval'/(f'{label}_{args.seconds}s_continuous3_'+stamp+('_smoke' if args.smoke else ''))).resolve()
    if ROOT/'output_etc/check' not in out.parents:parser.error('Output must be below stage1_combined/output_etc/check')
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(source,out/'checkpoint.pth');checkpoint=out/'checkpoint.pth'
    checkpoint_hash=sha(checkpoint)
    public_hash=None
    if public_source is not None:
        shutil.copy2(public_source,out/'public.pth')
        public_hash=sha(out/'public.pth')
        cfg['env']['chairFT']['public_checkpoint']=str(out/'public.pth')
    (out/'env.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    files=list(Path(__file__).parent.glob('*.py'))+[ROOT/'evaluation/finetuned/backend.py',ROOT/'evaluation/public_point/policy.py',ROOT.parent/'TokenHSI/tokenhsi/learning/transformer/trans_players.py']
    code_hashes={str(path):sha(path) for path in files}
    for index,path in enumerate(files):
        target=out/'code'/f'{index}_{path.name}';target.parent.mkdir(exist_ok=True);shutil.copy2(path,target)
    write_json(out/'manifest.json',dict(protocol='chair_sit_common40s_native_continuous3_v1',checkpoint=str(source),checkpoint_sha256=checkpoint_hash,
        epoch=policy_epoch,scene_checkpoint_epoch=epoch,architecture='original30d' if original30d else 'bbox35_mlp',policy=args.policy,
        public_checkpoint=str(public_source) if public_source else None,public_checkpoint_sha256=public_hash,category='Chair',action='SIT',test_models=26,
        num_envs=count,seed=args.seed,seconds=args.seconds,max_steps=steps,num_repeats=3,reseed_between_repeats=False,
        primary_included_repeats=[1,2],all_repeats_preserved=[0,1,2],smoke=args.smoke,gpu=args.gpu,code_sha256=code_hashes))
    if args.prepare_only:print('PREPARED',out,flush=True);return
    wait_memory(args.gpu,6500,out/'state.json')
    command=['bash','-c','. tokenhsi/scripts/multi_agent/runtime_env.sh; exec python "$@"','chair_sit_eval',str(Path(__file__).with_name('launch.py')),
        '--task','EvaluationTask','--cfg_train','tokenhsi/data/cfg/train/rlg/tokenhsi_climb_30d_full_ft.yaml' if original30d else 'tokenhsi/data/cfg/train/rlg/chair_table_eval_full_ft.yaml',
        '--cfg_env',str(out/'env.yaml'),'--checkpoint',str(checkpoint),'--num_agents','1','--num_objects','1','--num_envs',str(count),
        '--task_graph','sit','--eval_skills','loco','--eval_skill_probs','1','--eval_repeats','3','--episode_length',str(steps),
        '--output_path',str(out),'--seed',str(args.seed),'--headless','--no_video','--test','--eval','--motion_file','tokenhsi/data/dataset_loco_sit_carry_climb.yaml']
    env=os.environ.copy();env.update(TOKENHSI_GPU=args.gpu,TOKENHSI_CONDA_ENV='tokenhsi118',PYTORCH_JIT='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        TORCH_EXTENSIONS_DIR='/tmp/table_hold_at_torch_extensions',CLIMB_PAIRED_POLICY=args.policy,CLIMB_PAIRED_ORIGINAL30D='1' if original30d else '0',CLIMB_PAIRED_CONTINUOUS3='1')
    try:
        print('START Chair SIT',args.policy,'epoch',policy_epoch,'seed',args.seed,'envs',count,'seconds',args.seconds,'continuous repeats',3,'output',out,flush=True)
        with (out/'run.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write_json(out/'state.json',dict(status='running',worker_pid=child.pid,seed=args.seed,num_repeats=3))
            rc=child.wait()
        if rc:raise RuntimeError(f'Chair SIT worker rc={rc}; see {out/"run.log"}')
        result=json.loads((out/'results.json').read_text())
        assert result['num_repeats']==3 and result['reseed_between_repeats'] is False
        assert result['policy']['epoch']==policy_epoch and result['max_steps']==steps and abs(result['time_limit_seconds']-args.seconds)<1e-4
        assert result['scene_checkpoint_exact'] and result['scene_model_and_rms_unchanged']
        for repeat,row in enumerate(result['repeats']):
            trace=out/f'repeat_{repeat}/trace.npz';check=score_trace(trace,'SIT')
            assert sha(trace)==row['trace_sha256'] and check['num_envs']==count and check['max_end_frame']<=steps-1
            for metric in ['point','region']:assert row[metric+'_ever']==check[metric+'_count']/count
        stats=repeat_statistics(result['repeats']);assert stats==result['repeat_statistics']
        assert sha(checkpoint)==checkpoint_hash
        if public_hash is not None:assert sha(out/'public.pth')==public_hash==result['policy']['checkpoint_sha256']
        write_json(out/'summary.json',dict(seed=args.seed,num_envs=count,seconds=args.seconds,epoch=policy_epoch,scene_checkpoint_epoch=epoch,policy=args.policy,smoke=args.smoke,metrics=stats,std_scope='within one seed across repeats1/2; not between seeds'))
        with (out/'repeats.csv').open('w') as stream:
            writer=csv.writer(stream);writer.writerow(['seed','repeat','point_percent','region_percent','included_in_primary_mean'])
            for row in result['repeats']:writer.writerow([args.seed,row['repeat'],row['point_ever']*100,row['region_ever']*100,row['repeat'] in [1,2]])
        lines=[f'# Chair SIT {args.policy} 우리 영역 {args.seconds}초·연속3회','',f'Policy epoch={policy_epoch}, scene checkpoint epoch={epoch}, seed={args.seed}, environments={count}, seconds={args.seconds}, smoke={args.smoke}.',
            '의자test26균등복원추출,1명/고정의자1개,loco시작,deterministic. 골반이좌판로컬XY내부80%및좌판+10cm목표높이±20cm이면영역성공. 첫episode초기/종료frame포함ever. 별도접촉·방향·유지시간조건없음.',
            '한프로세스에서시드재설정없이원본루프로3회연속평가. 대표평균±표본std는repeat1·2,repeat0포함전체통계도보존. 표준편차는동일seed반복산포이며seed간산포가아님.', '',
            '| 반복 | 점 성공률 % | 영역 성공률 % | 대표집계 포함 |','|---|---:|---:|---|']
        for row in result['repeats']:lines.append(f'| {row["repeat"]} | {row["point_ever"]*100:.4f} | {row["region_ever"]*100:.4f} | {row["repeat"] in [1,2]} |')
        lines+=['',json.dumps(stats,indent=2)]
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
        write_json(out/'completed.json',dict(rc=0,all_traces_rescored=True,num_repeats=3,seed=args.seed,smoke=args.smoke))
        write_json(out/'state.json',dict(status='complete',rc=0,seed=args.seed,num_repeats=3))
        print('Region mean +/- repeat std:',stats['region']['mean_percent'],stats['region']['within_seed_sample_std_percentage_points'],flush=True)
        print('RESULT',out,flush=True)
    except Exception as exc:
        write_json(out/'state.json',dict(status='failed',error=str(exc)))
        raise


if __name__=='__main__':main()

"""Table40s, seeds0..9, native continuous3 per seed; retain every repeat."""
import argparse
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
from evaluation.climb_paired.metrics import score_trace
from evaluation.climb_paired.repeat_metrics import repeat_statistics,seed_statistics
from evaluation.climb_paired.run_eval import ROOT,PUBLIC,catalog,sha,write_json,wait_memory

CHECKPOINT=ROOT/'output/climb_adaptation/tokenhsi_climb_30d_100k_20261026_121912/train/TokenHSIClimb30DFullFT_26-12-19-15/nn/TokenHSIClimb30DFullFT_00002000.pth'


def save(out,rows,seeds,smoke):
    metrics=seed_statistics(rows)
    write_json(out/'records.json',rows)
    write_json(out/'summary.json',dict(metrics=metrics,evaluated_seeds=[r['seed'] for r in rows],planned_seeds=seeds,
        complete=len(rows)==len(seeds),smoke=smoke,num_repeats=3,primary_included_repeats=[1,2],
        no_reseed_between_repeats=True,primary_metric='region'))
    with (out/'repeats.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['seed','policy','repeat','point_percent','region_percent','included_in_primary_mean'])
        for pair in rows:
            for policy in ['tokenhsi','ours']:
                for record in pair[policy]['repeats']:
                    writer.writerow([pair['seed'],policy,record['repeat'],100*record['point_ever'],100*record['region_ever'],record['repeat'] in [1,2]])
    with (out/'seeds.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['seed','tokenhsi_region_mean_percent','tokenhsi_within_seed_std_pp','ours_region_mean_percent','ours_within_seed_std_pp','delta_pp'])
        for row in rows:
            public,ours=[row[p]['repeat_statistics']['region'] for p in ['tokenhsi','ours']]
            writer.writerow([row['seed'],public['mean_percent'],public['within_seed_sample_std_percentage_points'],ours['mean_percent'],ours['within_seed_sample_std_percentage_points'],ours['mean_percent']-public['mean_percent']])
    lines=['# Table CLIMB 40초 · 시드별 연속3회', '',
        'TokenHSI Stage1 40k와우리원본30D FT2k,Table test8,1명/고정물체1개,512환경. 각seed는정책별새프로세스에서한번초기화하고원본run_eval의reset/수집루프로3회연속평가한다. 반복사이seed재설정없음. 점20cm/영역80%XY·Z±20cm의첫episode ever를동일trace에서채점하며초기·종료frame포함. 원본논문환경/시간의재현이아닌공통40초평가다.',
        '원본운영집계대로repeat0은보존하되대표평균에서제외,repeat1·2평균과반복표본std를기록한다. 전체seed집계는각seed평균들의평균±seed간표본std다. 3회전체평균과std도각case results.json에보존한다.',
        '첫반복의초기장면은모델간exact검증한다. 이후반복은정책별종료/reset시점에따라RNG소비와초기장면이달라질수있어exact를가정하지않으며실제비교를기록한다.',
        f'완료seed: {[r["seed"] for r in rows]} / 계획: {seeds}. Smoke={smoke}.', '',
        '| Seed | TokenHSI 영역 평균±반복std % | 우리 영역 평균±반복std % | 차이 %p |',
        '|---|---:|---:|---:|']
    for row in rows:
        public,ours=[row[p]['repeat_statistics']['region'] for p in ['tokenhsi','ours']]
        lines.append(f'| {row["seed"]} | {public["mean_percent"]:.4f} ± {public["within_seed_sample_std_percentage_points"]:.4f} | {ours["mean_percent"]:.4f} ± {ours["within_seed_sample_std_percentage_points"]:.4f} | {ours["mean_percent"]-public["mean_percent"]:.4f} |')
    lines+=['','전체seed집계(±는seed간std이며위표의반복std와구분):',json.dumps(metrics,indent=2),'',
        '계획한모든seed와repeat0/1/2를보존한다. 이기는seed만대표결과로선택하지않으며모든repeat의trace를독립SciPy로재채점한다.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(10)))
    parser.add_argument('--seconds',type=int,default=40)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--min-free-mib',type=int,default=6500)
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0 or args.seconds<=0:parser.error('Distinct nonnegative seeds and positive seconds required')
    count=4 if args.smoke else 512;seeds=[args.seeds[0]] if args.smoke else args.seeds
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    out=(args.output or ROOT/'output_etc/check/climb_eval'/('table_30d_2k_continuous3_'+stamp+('_smoke' if args.smoke else ''))).resolve()
    assert ROOT/'output_etc/check' in out.parents
    out.mkdir(parents=True,exist_ok=False)
    for policy,path in [('ours',args.checkpoint.resolve()),('tokenhsi',PUBLIC)]:shutil.copy2(path,out/(policy+'.pth'))
    weights=torch.load(out/'ours.pth',map_location='cpu',weights_only=False)
    assert weights['epoch']==2000 and weights['model']['a2c_network.actor_encoder.tokenizers.1.0.weight'].shape[1]==30
    public=torch.load(out/'tokenhsi.pth',map_location='cpu',weights_only=False);assert public['epoch']==40000
    del weights,public
    cfg=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/tokenhsi_climb_30d_full_ft.yaml').read_text())
    entries=catalog('Table');assert len(entries)==8
    for entry in entries:entry['feature']=[1.,0.,0.,0.,0.,0.]
    steps=args.seconds*30
    cfg['env'].update(numAgents=1,numObjects=1,numEnvs=count,episodeLength=steps,debug=dict(reward=False))
    cfg['env']['chairFT'].update(asset_split='test',point_evaluation=dict(category='Table',action='CLIMB',catalog=entries),public_checkpoint=str(out/'tokenhsi.pth'))
    code_files=list(Path(__file__).parent.glob('*.py'))+[ROOT/'evaluation/finetuned/backend.py',ROOT/'evaluation/public_point/policy.py',ROOT.parent/'TokenHSI/tokenhsi/learning/transformer/trans_players.py']
    hashes={str(p):sha(p) for p in code_files}
    for index,path in enumerate(code_files):
        target=out/'code'/f'{index}_{path.name}';target.parent.mkdir(exist_ok=True);shutil.copy2(path,target)
    checkpoint_hashes={policy:sha(out/(policy+'.pth')) for policy in ['tokenhsi','ours']}
    write_json(out/'manifest.json',dict(protocol='table_climb_common40s_native_continuous3_v1',num_envs=count,seeds=seeds,seconds=args.seconds,max_steps=steps,
        num_repeats=3,reseed_between_repeats=False,primary_included_repeats=[1,2],all_repeats_preserved=[0,1,2],
        gpu=args.gpu,smoke=args.smoke,checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=checkpoint_hashes,code_sha256=hashes))
    (out/'env.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));state=out/'state.json';rows=[]
    try:
        for seed in seeds:
            pair=dict(seed=seed)
            for policy in ['tokenhsi','ours']:
                for path in code_files:assert sha(path)==hashes[str(path)],'Evaluation code changed mid-run'
                wait_memory(args.gpu,args.min_free_mib,state)
                case=out/policy/f'seed_{seed}';case.mkdir(parents=True)
                (case/'env.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
                command=['bash','-c','. tokenhsi/scripts/multi_agent/runtime_env.sh; exec python "$@"','continuous3',str(Path(__file__).with_name('launch.py')),
                    '--task','EvaluationTask','--cfg_train','tokenhsi/data/cfg/train/rlg/tokenhsi_climb_30d_full_ft.yaml','--cfg_env',str(case/'env.yaml'),
                    '--checkpoint',str(out/'ours.pth'),'--num_agents','1','--num_objects','1','--num_envs',str(count),'--task_graph','climb',
                    '--eval_skills','loco','--eval_skill_probs','1','--eval_repeats','3','--episode_length',str(steps),'--output_path',str(case),
                    '--seed',str(seed),'--headless','--no_video','--test','--eval','--motion_file','tokenhsi/data/dataset_loco_sit_carry_climb.yaml']
                env=os.environ.copy();env.update(TOKENHSI_GPU=args.gpu,TOKENHSI_CONDA_ENV='tokenhsi118',PYTORCH_JIT='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                    TORCH_EXTENSIONS_DIR='/tmp/table_hold_at_torch_extensions',CLIMB_PAIRED_POLICY=policy,CLIMB_PAIRED_ORIGINAL30D='1',CLIMB_PAIRED_CONTINUOUS3='1')
                print('START',policy,'seed',seed,'repeats',3,flush=True)
                with (case/'run.log').open('w') as log:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    write_json(state,dict(status='running',policy=policy,seed=seed,worker_pid=child.pid,completed_seed_pairs=len(rows)))
                    rc=child.wait()
                if rc:raise RuntimeError(f'{policy} seed{seed} rc={rc}: {case}')
                record=json.loads((case/'results.json').read_text())
                assert record['num_repeats']==3 and record['reseed_between_repeats'] is False
                assert record['max_steps']==steps and abs(record['time_limit_seconds']-args.seconds)<1e-4
                assert record['policy']['epoch']==(40000 if policy=='tokenhsi' else 2000)
                assert record['scene_checkpoint_exact'] and record['scene_model_and_rms_unchanged']
                for repeat,repeat_record in enumerate(record['repeats']):
                    path=case/f'repeat_{repeat}'/'trace.npz';check=score_trace(path)
                    assert sha(path)==repeat_record['trace_sha256']
                    assert check['num_envs']==count and check['max_end_frame']<=steps-1
                    for metric in ['point','region']:assert repeat_record[metric+'_ever']==check[metric+'_count']/count
                assert repeat_statistics(record['repeats'])==record['repeat_statistics']
                pair[policy]=record
                print('DONE',policy,'seed',seed,json.dumps(record['repeat_statistics']),flush=True)
            initial_comparison=[]
            for repeat in range(3):
                paths=[out/p/f'seed_{seed}'/f'repeat_{repeat}'/'initial.npz' for p in ['tokenhsi','ours']]
                with np.load(paths[0]) as a,np.load(paths[1]) as b:
                    equal={key:bool(np.array_equal(a[key],b[key])) for key in a.files}
                    initial_comparison.append(dict(repeat=repeat,arrays_exact=equal,all_exact=all(equal.values())))
            assert initial_comparison[0]['all_exact'],'First repeat initial scene mismatch'
            pair['initial_comparison_by_repeat']=initial_comparison
            rows.append(pair);save(out,rows,seeds,args.smoke)
            print('SEED_DONE',seed,'region delta pp',pair['ours']['repeat_statistics']['region']['mean_percent']-pair['tokenhsi']['repeat_statistics']['region']['mean_percent'],flush=True)
        for policy in checkpoint_hashes:assert sha(out/(policy+'.pth'))==checkpoint_hashes[policy]
        write_json(state,dict(status='complete',rc=0,completed_seed_pairs=len(rows)))
        write_json(out/'completed.json',dict(rc=0,num_seeds=len(seeds),num_repeats_per_seed=3,all_traces_rescored=True))
        print('RESULT',out,flush=True)
    except Exception as exc:
        write_json(state,dict(status='failed',error=str(exc),completed_seed_pairs=len(rows)))
        raise


if __name__=='__main__':main()

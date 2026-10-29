"""Bounded paired Table CLIMB comparison; all planned seeds are retained."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
import yaml
from evaluation.climb_paired.metrics import score_trace, summarize
from evaluation.finetuned.run_eval import ROOT, catalog
from evaluation.public_point.run_eval import PUBLIC

DEFAULT_CHECKPOINT = ROOT/'checkpoints/Stage1V2Fixed_00078000.pth'


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_summary(out, rows, seeds, smoke, epoch):
    summary = summarize(rows)
    summary.update(planned_seeds=seeds, complete=len(rows)==len(seeds), smoke=smoke,
                   primary_metric='region', selection='all planned seeds, no early stopping or filtering')
    write_json(out/'summary.json', summary)
    write_json(out/'records.json', rows)
    with (out/'seeds.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['seed', 'tokenhsi_point_percent', 'ours_point_percent', 'point_delta_pp', 'tokenhsi_region_percent', 'ours_region_percent', 'region_delta_pp'])
        for row in rows:
            values = [row['seed']]
            for metric in ['point', 'region']:
                public, ours = [row[p][metric+'_ever']*100 for p in ['tokenhsi', 'ours']]
                values.extend([public, ours, ours-public])
            writer.writerow(values)
    lines = ['# Table CLIMB paired evaluation', '',
        '40s by default; 1 human/1 fixed object, native Table test8, uniform sampling with replacement. Loco start, deterministic actions, first episode ever including initial/terminal frame.',
        'Primary: pelvis in object-local inner80% XY and target Z±20cm. Secondary: pelvis-to-target 3D distance≤20cm. Target=metadata+.94m. No feet/contact/hold condition.',
        f'Public TokenHSI Stage1 40k vs our original30D FT{epoch}. This common evaluation is not a reproduction of the original paper protocol.',
        'Identical scene checkpoint/config for both policies; public policy creation preserves RNG. Every initial array is compared exactly. Every trace is independently rescored using SciPy rotation.', '',
        f'Planned seeds: {seeds}. Completed pairs: {len(rows)}. Smoke: {smoke}.', '',
        '| Seed | TokenHSI point % | Ours point % | Delta pp | TokenHSI region % | Ours region % | Delta pp |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values=[]
        for metric in ['point','region']:
            public, ours = [row[p][metric+'_ever']*100 for p in ['tokenhsi','ours']]
            values.extend([public,ours,ours-public])
        lines.append('| '+str(row['seed'])+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines += ['', 'All evaluated seeds contribute to the aggregate. Winning seeds are exploratory diagnostics and must not be presented as representative performance. Std is across evaluation seeds, not training seeds.']
    for metric, stats in summary['metrics'].items():
        lines += ['', f'{metric}: {json.dumps(stats)}']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def wait_memory(gpu, minimum, state):
    while True:
        free = int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True).strip())
        if free >= minimum:
            return
        write_json(state, dict(status='waiting_for_gpu_memory', gpu=gpu, free_MiB=free, required_MiB=minimum))
        print('WAIT_GPU_MEMORY', free, 'required', minimum, flush=True)
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--public-checkpoint',type=Path,default=PUBLIC)
    parser.add_argument('--seconds',type=int,default=40)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(10)))
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--min-free-mib',type=int,default=6500)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--public-only',action='store_true',help='Evaluate only TokenHSI; the supplied checkpoint constructs the scene only')
    parser.add_argument('--reference-dir',type=Path,help='Reuse validated public-only results with identical protocol and initial scenes')
    args=parser.parse_args()
    if args.seconds<=0 or not args.seeds or len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0:
        parser.error('positive seconds and distinct nonnegative seeds required')
    seeds=[args.seeds[0]] if args.smoke else args.seeds
    count=4 if args.smoke else 512
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    out=(args.output or ROOT/'output_etc/check/climb_eval'/('table_paired_40s_'+stamp+('_smoke' if args.smoke else ''))).resolve()
    if ROOT/'output_etc/check' not in out.parents:
        parser.error('output must be below stage1_combined/output_etc/check')
    out.mkdir(parents=True,exist_ok=False)
    originals={'ours':args.checkpoint.resolve(),'tokenhsi':args.public_checkpoint.resolve()}
    for policy,path in originals.items():shutil.copy2(path,out/(policy+'.pth'))
    weights=torch.load(out/'ours.pth',map_location='cpu',weights_only=False)
    original30d='a2c_network.actor_encoder.tokenizers.1.0.weight' in weights['model'] and weights['model']['a2c_network.actor_encoder.tokenizers.1.0.weight'].shape[1]==30
    assert original30d, 'This comparison requires the user-requested original30D checkpoint'
    epoch=int(weights['epoch'])
    public=torch.load(out/'tokenhsi.pth',map_location='cpu',weights_only=False)
    assert public['epoch']==40000
    del public
    cfg=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/tokenhsi_climb_30d_full_ft.yaml').read_text())
    entries=catalog('Table');assert len(entries)==8
    # Shape descriptors are unused by the original30D policy.
    for entry in entries:entry['feature']=[1.,0.,0.,0.,0.,0.]
    steps=args.seconds*30
    cfg['env'].update(numAgents=1,numObjects=1,numEnvs=count,episodeLength=steps,debug=dict(reward=False))
    cfg['env']['chairFT'].update(asset_split='test',point_evaluation=dict(category='Table',action='CLIMB',catalog=entries),public_checkpoint=str(out/'tokenhsi.pth'))
    code_files=list(Path(__file__).parent.glob('*.py'))+[ROOT/'evaluation/finetuned/backend.py',ROOT/'evaluation/public_point/policy.py']
    hashes={str(path.relative_to(ROOT)):sha(path) for path in code_files}
    for path in code_files:
        target=out/'code'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    manifest=dict(protocol='table_climb_common_40s_paired_v1',seconds=args.seconds,max_steps=steps,num_envs=count,seeds=seeds,gpu=args.gpu,smoke=args.smoke,public_only=args.public_only,architecture='original30d_no_expansion',epoch=epoch,
        primary_metric='region',policy_order=['tokenhsi','ours'],reference_dir=str(args.reference_dir.resolve()) if args.reference_dir else None,checkpoints={key:dict(source=str(path),sha256=sha(out/(key+'.pth'))) for key,path in originals.items()},
        code_sha256=hashes,category='Table',test_models=entries,shape_descriptor='unused by original30D policy; no parameter expansion',selection='fixed seed list before evaluation; all results retained')
    write_json(out/'manifest.json',manifest)
    (out/'env.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    if args.prepare_only:
        print('PREPARED',out,flush=True);return
    state=out/'state.json';rows=[];public_rows=[]
    try:
        for seed in seeds:
            pair=dict(seed=seed)
            for policy in (['tokenhsi'] if args.public_only else ['tokenhsi','ours']):
                for path in code_files:assert sha(path)==hashes[str(path.relative_to(ROOT))], 'Evaluation code changed mid-run'
                case=out/policy/('seed_'+str(seed));case.mkdir(parents=True)
                (case/'env.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
                if policy=='tokenhsi' and args.reference_dir:
                    reference=args.reference_dir.resolve()
                    ref_manifest=json.loads((reference/'manifest.json').read_text())
                    for key in ['protocol','seconds','max_steps','num_envs','smoke','architecture','category']:
                        assert ref_manifest[key]==manifest[key],key
                    assert ref_manifest['checkpoints']['tokenhsi']['sha256']==manifest['checkpoints']['tokenhsi']['sha256']
                    reference_cfg=yaml.safe_load((reference/'env.yaml').read_text())
                    reference_cfg['env']['chairFT']['public_checkpoint']=cfg['env']['chairFT']['public_checkpoint']
                    assert reference_cfg==cfg, 'Reference scene config changed'
                    previous=reference/policy/('seed_'+str(seed))
                    for name in ['initial.npz','trace.npz','results.json','independent_validation.json']:
                        shutil.copy2(previous/name,case/name)
                    record=json.loads((case/'results.json').read_text());check=score_trace(case/'trace.npz')
                    for metric in ['point','region']:assert record[metric+'_ever']==check[metric+'_count']/count
                    assert sha(case/'trace.npz')==record['trace_sha256']
                    pair[policy]=record
                    print('REUSED',policy,seed,flush=True)
                    continue
                wait_memory(args.gpu,args.min_free_mib,state)
                command=['bash','-c','. tokenhsi/scripts/multi_agent/runtime_env.sh; exec python "$@"','paired_climb',str(Path(__file__).with_name('launch.py')),
                    '--task','EvaluationTask','--cfg_train','tokenhsi/data/cfg/train/rlg/tokenhsi_climb_30d_full_ft.yaml','--cfg_env',str(case/'env.yaml'),
                    '--checkpoint',str(out/'ours.pth'),'--num_agents','1','--num_objects','1','--num_envs',str(count),'--task_graph','climb',
                    '--eval_skills','loco','--eval_skill_probs','1','--eval_repeats','1','--episode_length',str(steps),'--output_path',str(case),
                    '--seed',str(seed),'--headless','--no_video','--test','--eval','--motion_file','tokenhsi/data/dataset_loco_sit_carry_climb.yaml']
                env=os.environ.copy();env.update(TOKENHSI_GPU=args.gpu,TOKENHSI_CONDA_ENV='tokenhsi118',PYTORCH_JIT='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                    TORCH_EXTENSIONS_DIR='/tmp/table_hold_at_torch_extensions',CLIMB_PAIRED_POLICY=policy,CLIMB_PAIRED_ORIGINAL30D='1')
                write_json(state,dict(status='running',seed=seed,policy=policy,completed_pairs=len(rows)))
                print('START',policy,'seed',seed,'envs',count,'steps',steps,flush=True)
                with (case/'run.log').open('w') as log:
                    worker=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    write_json(state,dict(status='running',worker_pid=worker.pid,seed=seed,policy=policy,completed_pairs=len(rows)))
                    rc=worker.wait()
                if rc:raise RuntimeError(f'{policy} seed{seed} rc={rc}: {case}')
                record=json.loads((case/'results.json').read_text())
                check=score_trace(case/'trace.npz')
                assert record['num_envs']==count and record['seed']==seed
                assert record['max_steps']==steps and abs(record['time_limit_seconds']-args.seconds)<1e-4
                assert record['max_end_frame']<=steps and record['ended']==count
                assert record['policy']['epoch']==(40000 if policy=='tokenhsi' else epoch)
                for metric in ['point','region']:assert record[metric+'_ever']==check[metric+'_count']/count
                assert sha(case/'trace.npz')==record['trace_sha256']
                write_json(case/'independent_validation.json',dict(ok=True,scipy_rescore=check))
                pair[policy]=record
                print('DONE',policy,'seed',seed,'point',record['point_ever']*100,'region',record['region_ever']*100,flush=True)
            if args.public_only:
                public_rows.append(dict(pair['tokenhsi']))
                write_json(out/'records.json',public_rows)
                stats={metric:dict(seed_percent=[r[metric+'_ever']*100 for r in public_rows],
                    mean_percent=float(np.mean([r[metric+'_ever']*100 for r in public_rows])),
                    sample_std_percentage_points=float(np.std([r[metric+'_ever']*100 for r in public_rows],ddof=1)) if len(public_rows)>1 else None) for metric in ['point','region']}
                write_json(out/'summary.json',dict(policy='tokenhsi',public_only=True,smoke=args.smoke,planned_seeds=seeds,evaluated_seeds=[r['seed'] for r in public_rows],metrics=stats))
                (out/'REPORT.md').write_text('# TokenHSI Table CLIMB common evaluation\n\n'+json.dumps(dict(seconds=args.seconds,num_envs=count,smoke=args.smoke,metrics=stats),indent=2)+'\n\nAll evaluated seeds retained. Evaluation seed sample std, not training seed std.\n')
                continue
            with np.load(out/'tokenhsi'/('seed_'+str(seed))/'initial.npz') as public_initial, np.load(out/'ours'/('seed_'+str(seed))/'initial.npz') as our_initial:
                assert set(public_initial.files)==set(our_initial.files)
                for key in public_initial.files:np.testing.assert_array_equal(public_initial[key],our_initial[key],err_msg=key)
            pair['all_initial_arrays_exact']=True
            rows.append(pair);save_summary(out,rows,seeds,args.smoke,epoch)
            print('PAIRED',seed,'region delta pp',100*(pair['ours']['region_ever']-pair['tokenhsi']['region_ever']),flush=True)
        for policy in originals:assert sha(out/(policy+'.pth'))==manifest['checkpoints'][policy]['sha256']
        write_json(state,dict(status='complete',rc=0,completed_pairs=len(rows),completed_public_seeds=len(public_rows)))
        write_json(out/'completed.json',dict(rc=0,paired_initial_exact=None if args.public_only else True,all_traces_rescored=True))
        print('RESULT',out,flush=True)
    except Exception as exc:
        write_json(state,dict(status='failed',error=str(exc),completed_pairs=len(rows)))
        raise


if __name__=='__main__':main()

"""Run three independent seeds. A general scene is never a single-task scene."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tokenhsi'))
from utils.experiments import NAMES,experiment_horizon
from utils.output_paths import resolve_output_path
from utils.evaluation_protocol import HORIZON


def summarize(directory):
    runs = [json.loads(p.read_text()) for p in sorted(directory.glob('*/result.json'))]
    summary = dict(runs=len(runs), protocol=runs[0]['protocol'] if runs else None, scores={})
    for criterion in ('point', 'region'):
        summary['scores'][criterion] = {}
        for task in ('AT', 'ON_TOP', 'SIT', 'CLIMB', 'PUSH'):
            rows = [r['scores'][criterion][task] for r in runs if r['scores'][criterion].get(task, {}).get('trials', 0)]
            values = [r['rate'] for r in rows]
            if not values:
                continue
            item = dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values)>1 else None,
                        successes=sum(r['success'] for r in rows), trials=sum(r['trials'] for r in rows), seeds=values)
            for metric in ('follower_individual', 'both_ever', 'simultaneous'):
                pairs = [r[metric+'_success']/r['pair_trials'] for r in rows if r['pair_trials']]
                item[metric] = dict(mean=statistics.mean(pairs), std=statistics.stdev(pairs) if len(pairs)>1 else None,
                    successes=sum(r[metric+'_success'] for r in rows), trials=sum(r['pair_trials'] for r in rows)) if pairs else None
            summary['scores'][criterion][task] = item
        for count in (4,5):
            seed_macro=[]
            names=('AT','ON_TOP','SIT','CLIMB','PUSH')[:count]
            for seed in (0,1,2):
                values=[r['scores'][criterion][task]['rate'] for task in names for r in runs
                    if r['seed']==seed and r['scores'][criterion].get(task,{}).get('trials',0)]
                if len(values)==count:seed_macro.append(statistics.mean(values))
            if seed_macro:
                item=dict(mean=statistics.mean(seed_macro),std=statistics.stdev(seed_macro) if len(seed_macro)>1 else None,seeds=seed_macro)
                summary['scores'][criterion]['macro'+str(count)]=item
                summary['scores'][criterion]['macro']=item
    (directory/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=NAMES)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--profile', choices=['general','stage1_single'], default='general')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--envs', type=int, default=512, help='512 for report; smaller only for checks')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    parser.add_argument('--tasks', nargs='+', choices=['AT','ON_TOP','SIT','CLIMB','PUSH'], default=None)
    args=parser.parse_args()
    if args.tasks is None:
        args.tasks=['AT','ON_TOP','SIT','CLIMB']+(['PUSH'] if args.experiment in ('stage1_push','stage2_before_push','stage2_coupled_push','stage2_before_push_near50','stage2_coupled_push_region_goals') else [])
    if 'PUSH' in args.tasks and args.experiment not in ('stage1_push','stage2_before_push','stage2_coupled_push','stage2_before_push_near50','stage2_coupled_push_region_goals'):
        parser.error('PUSH requires a PUSH experiment')
    if args.envs <= 0 or len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset({0,1,2}):
        parser.error('Use positive environment count and distinct seeds from 0,1,2')
    if args.profile=='stage1_single' and args.experiment not in ('stage1','stage1_push'):
        parser.error('Single-task evaluation is Stage1-only')
    checkpoint=args.checkpoint.resolve(strict=True)
    directory=Path(resolve_output_path(args.experiment,'eval',str(args.output)))
    directory.mkdir(parents=True,exist_ok=True)
    manifest=vars(args).copy()
    manifest["horizon"] = experiment_horizon(args.experiment)
    manifest.update(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), resolved_checkpoint=str(checkpoint),
                    formal_report=(args.envs==512 and args.seeds==[0,1,2] and (args.profile=='general' or set(args.tasks)==({'AT','ON_TOP','SIT','CLIMB','PUSH'} if args.experiment in ('stage1_push','stage2_before_push','stage2_coupled_push','stage2_before_push_near50','stage2_coupled_push_region_goals') else {'AT','ON_TOP','SIT','CLIMB'}))))
    old_path=directory/'manifest.json'
    if old_path.exists():
        old=json.loads(old_path.read_text())
        for key in ('experiment','profile','envs','seeds','tasks','checkpoint_sha256','horizon'):
            if old.get(key) != manifest.get(key):
                raise ValueError(f'Existing evaluation has different {key}; use a new output directory')
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    tasks=args.tasks if args.profile=='stage1_single' else ['general']
    for task in tasks:
        for seed in args.seeds:
            out=directory/f'{task}_seed{seed}'
            if (out/'result.json').exists() and (out/'trace.npz').exists():
                continue
            out.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,str(ROOT/'tokenhsi/run.py'),'--task','HumanoidMACoupledPush' if args.experiment in ('stage2_coupled_push','stage2_coupled_push_region_goals') else 'HumanoidMABeforePush' if args.experiment in ('stage2_before_push','stage2_before_push_near50') else 'HumanoidMAPush' if args.experiment=='stage1_push' else 'HumanoidMACarry' if args.experiment=='stage1' else 'HumanoidMABefore',
                 '--cfg_train',f'tokenhsi/data/cfg/train/rlg/{args.experiment}.yaml', '--cfg_env','',
                 '--motion_file',('tokenhsi/data/dataset_stage1_jointpush.yaml' if args.experiment in ('stage1_push','stage2_before_push','stage2_coupled_push','stage2_before_push_near50','stage2_coupled_push_region_goals') else 'tokenhsi/data/dataset_loco_sit_carry_climb.yaml'), '--checkpoint',str(checkpoint),
                 '--test','--eval','--headless','--no_video','--eval_protocol',args.profile,
                 '--eval_repeats','1','--num_envs',str(args.envs),'--seed',str(seed),'--output_path',str(out),
                 '--task_graph','random_scenario']
            if task!='general':cmd += ['--eval_single_task',task]
            print(f'Running {args.profile}/{task} seed={seed}: {out}',flush=True)
            with (out/'run.log').open('w') as log:
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Run failed: {out}/run.log (exit {result.returncode})')
    print(json.dumps(summarize(directory),indent=2),flush=True)

if __name__=='__main__':main()

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
from utils.experiments import NAMES,experiment_horizon,STAGE2_NAMES
from utils.output_paths import resolve_output_path
from utils.evaluation_protocol import ENABLING_SUCCESS, HORIZON, summarize_interaction_groups
from utils.evaluation_errors import ERROR_PROTOCOL, summarize_errors
from env.tasks.multi_agent.stage2_group_layout import EVAL_RADII


def summarize(directory):
    runs = [json.loads(p.read_text()) for p in sorted(directory.glob('*/result.json'))]
    criteria = {r.get('enabling_success', 'both_ever') for r in runs if 'interaction_group_scores' in r}
    if len(criteria) > 1:
        raise ValueError('Cannot summarize different enabling success criteria together')
    errors = summarize_errors(runs) if any('errors' in r for r in runs) else None
    summary = dict(runs=len(runs), protocol=runs[0]['protocol'] if runs else None, scores={})
    if criteria:
        summary['enabling_success'] = next(iter(criteria))
    if errors is not None:
        summary['errors'] = errors
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
            for seed in sorted({r['seed'] for r in runs}):
                values=[r['scores'][criterion][task]['rate'] for task in names for r in runs
                    if r['seed']==seed and r['scores'][criterion].get(task,{}).get('trials',0)]
                if len(values)==count:seed_macro.append(statistics.mean(values))
            if seed_macro:
                item=dict(mean=statistics.mean(seed_macro),std=statistics.stdev(seed_macro) if len(seed_macro)>1 else None,seeds=seed_macro)
                summary['scores'][criterion]['macro'+str(count)]=item
                summary['scores'][criterion]['macro']=item
    coupled = [r['coupled_scores'] for r in runs if 'coupled_scores' in r]
    if coupled:
        summary['coupled_scores'] = {}
        for criterion in ('point', 'region'):
            summary['coupled_scores'][criterion] = {}
            for size in sorted({size for r in coupled for size in r[criterion]}, key=int):
                rows = [r[criterion][size] for r in coupled if size in r[criterion]]
                item = {key: sum(row[key] for row in rows) for key in rows[0]}
                for metric in ('individual_ever', 'individual_final', 'simultaneous_ever', 'simultaneous_final'):
                    denominator = item['people' if metric.startswith('individual') else 'groups']
                    item[metric+'_rate'] = item[metric]/denominator if denominator else None
                summary['coupled_scores'][criterion][size] = item
    centers = [r['shared_center_scores'] for r in runs if 'shared_center_scores' in r]
    if centers:
        summary['shared_center_scores'] = {}
        for size in sorted({size for row in centers for size in row}, key=int):
            item = {key: sum(row[size][key] for row in centers if size in row)
                    for key in ('groups', 'ever_success', 'final_success')}
            for metric in ('ever', 'final'):
                item[metric+'_rate'] = item[metric+'_success']/item['groups'] if item['groups'] else None
            summary['shared_center_scores'][size] = item
    interaction = [r['interaction_group_scores'] for r in runs if 'interaction_group_scores' in r]
    if interaction:
        summary['interaction_group_scores'] = summarize_interaction_groups(interaction)
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
    parser.add_argument('--agents', type=int, default=0)
    parser.add_argument('--objects', type=int, default=0)
    parser.add_argument('--group_spawn_over4', action='store_true')
    parser.add_argument('--enable_size_mode', choices=['native','per_action_intersection'], default='native')
    parser.add_argument('--coupled_assignment', choices=['random','ordered','distance','optimal'], default=None)
    args=parser.parse_args()
    if args.experiment in STAGE2_NAMES and args.coupled_assignment is None:
        args.coupled_assignment='ordered'
    if args.enable_size_mode != 'native' and (args.experiment not in STAGE2_NAMES or args.profile != 'general'):
        parser.error('--enable_size_mode requires unified stage2 general evaluation')
    if args.experiment not in STAGE2_NAMES and (args.agents or args.objects or args.group_spawn_over4 or args.coupled_assignment):
        parser.error('Scene and assignment overrides are exclusive to unified stage2 evaluation')
    if args.tasks is None:
        args.tasks=['AT','ON_TOP','SIT','CLIMB']+(['PUSH'] if args.experiment in ('stage2','stage2_distill','stage2_900','stage2_distill_900','stage2_distill_900_v2','stage2_distill_900_v3','stage2_distill_900_v4','stage2_900_v4_reward','stage1_push','stage2_before_push','stage2_coupled_push','stage2_before_push_near50','stage2_before_push_near50_pair50_source5','stage2_coupled_push_goal5','stage2_coupled_push_individual_goal','stage2_coupled_push_individual_goal_solo5','stage2_coupled_push_individual_goal_solo5_carry5','stage2_coupled_push_individual_goal_solo5_carry5_ground','stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg','stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg','stage2_coupled_push_individual_goal_solo5_countnorm_coupled','stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none') else [])
    if 'PUSH' in args.tasks and args.experiment not in ('stage2','stage2_distill','stage2_900','stage2_distill_900','stage2_distill_900_v2','stage2_distill_900_v3','stage2_distill_900_v4','stage2_900_v4_reward','stage1_push','stage2_before_push','stage2_coupled_push','stage2_before_push_near50','stage2_before_push_near50_pair50_source5','stage2_coupled_push_goal5','stage2_coupled_push_individual_goal','stage2_coupled_push_individual_goal_solo5','stage2_coupled_push_individual_goal_solo5_carry5','stage2_coupled_push_individual_goal_solo5_carry5_ground','stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg','stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg','stage2_coupled_push_individual_goal_solo5_countnorm_coupled','stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none'):
        parser.error('PUSH requires a PUSH experiment')
    if args.envs <= 0 or not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error('Use positive environment count and distinct nonnegative seeds')
    if args.profile=='stage1_single' and args.experiment not in ('stage1','stage1_push'):
        parser.error('Single-task evaluation is Stage1-only')
    checkpoint=args.checkpoint.resolve(strict=True)
    directory=Path(resolve_output_path(args.experiment,'eval',str(args.output)))
    directory.mkdir(parents=True,exist_ok=True)
    manifest=vars(args).copy()
    manifest['enabling_success'] = ENABLING_SUCCESS
    manifest['error_protocol'] = ERROR_PROTOCOL
    manifest['group_spawn_radius'] = EVAL_RADII.get(args.agents) if args.group_spawn_over4 else None
    manifest["horizon"] = experiment_horizon(args.experiment)
    manifest.update(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), resolved_checkpoint=str(checkpoint),
                    formal_report=(args.enable_size_mode=='native' and args.envs==512 and not args.agents and not args.objects and not args.group_spawn_over4 and args.seeds==[0,1,2] and (args.profile=='general' or set(args.tasks)==({'AT','ON_TOP','SIT','CLIMB','PUSH'} if args.experiment in ('stage2','stage2_distill','stage2_900','stage2_distill_900','stage2_distill_900_v2','stage2_distill_900_v3','stage2_distill_900_v4','stage2_900_v4_reward','stage1_push','stage2_before_push','stage2_coupled_push','stage2_before_push_near50','stage2_before_push_near50_pair50_source5','stage2_coupled_push_goal5','stage2_coupled_push_individual_goal','stage2_coupled_push_individual_goal_solo5','stage2_coupled_push_individual_goal_solo5_carry5','stage2_coupled_push_individual_goal_solo5_carry5_ground','stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg','stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg','stage2_coupled_push_individual_goal_solo5_countnorm_coupled','stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none') else {'AT','ON_TOP','SIT','CLIMB'}))))
    old_path=directory/'manifest.json'
    if old_path.exists():
        old=json.loads(old_path.read_text())
        for key in ('experiment','profile','envs','seeds','tasks','checkpoint_sha256','horizon','agents','objects','group_spawn_over4','group_spawn_radius','coupled_assignment','enable_size_mode','enabling_success','error_protocol'):
            if old.get(key, 'native' if key == 'enable_size_mode' else None) != manifest.get(key):
                raise ValueError(f'Existing evaluation has different {key}; use a new output directory')
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    tasks=args.tasks if args.profile=='stage1_single' else ['general']
    for task in tasks:
        for seed in args.seeds:
            out=directory/f'{task}_seed{seed}'
            if (out/'result.json').exists() and (out/'trace.npz').exists():
                saved=json.loads((out/'result.json').read_text())
                if saved.get('errors', {}).get('protocol') != ERROR_PROTOCOL:
                    raise ValueError(f'Existing evaluation has no matching error metrics: {out}; use a new output directory')
                if 'interaction_group_scores' in saved and saved.get('enabling_success', 'both_ever') != ENABLING_SUCCESS:
                    raise ValueError(f'Existing evaluation uses a different enabling success criterion: {out}; use a new output directory')
                continue
            out.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,str(ROOT/'tokenhsi/run.py'),'--task','HumanoidMACoupledPush' if args.experiment in ('stage2','stage2_distill','stage2_900','stage2_distill_900','stage2_distill_900_v2','stage2_distill_900_v3','stage2_distill_900_v4','stage2_900_v4_reward','stage2_coupled_push','stage2_coupled_push_goal5','stage2_coupled_push_individual_goal','stage2_coupled_push_individual_goal_solo5','stage2_coupled_push_individual_goal_solo5_carry5','stage2_coupled_push_individual_goal_solo5_carry5_ground','stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg','stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg','stage2_coupled_push_individual_goal_solo5_countnorm_coupled','stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none') else 'HumanoidMABeforePush' if args.experiment in ('stage2_before_push','stage2_before_push_near50','stage2_before_push_near50_pair50_source5') else 'HumanoidMAPush' if args.experiment=='stage1_push' else 'HumanoidMACarry' if args.experiment=='stage1' else 'HumanoidMABefore',
                 '--cfg_train',f'tokenhsi/data/cfg/train/rlg/{args.experiment}.yaml', '--cfg_env','',
                 '--motion_file',('tokenhsi/data/dataset_stage1_jointpush.yaml' if args.experiment in ('stage2','stage2_distill','stage2_900','stage2_distill_900','stage2_distill_900_v2','stage2_distill_900_v3','stage2_distill_900_v4','stage2_900_v4_reward','stage1_push','stage2_before_push','stage2_coupled_push','stage2_before_push_near50','stage2_before_push_near50_pair50_source5','stage2_coupled_push_goal5','stage2_coupled_push_individual_goal','stage2_coupled_push_individual_goal_solo5','stage2_coupled_push_individual_goal_solo5_carry5','stage2_coupled_push_individual_goal_solo5_carry5_ground','stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_learnedbg','stage2_coupled_push_individual_goal_solo5_carry5_ground_relfusion_zerobg','stage2_coupled_push_individual_goal_solo5_countnorm_coupled','stage2_coupled_push_individual_goal_solo5_countnorm_coupled_none') else 'tokenhsi/data/dataset_loco_sit_carry_climb.yaml'), '--checkpoint',str(checkpoint),
                 '--test','--eval','--headless','--no_video','--eval_protocol',args.profile,
                 '--eval_repeats','1','--num_envs',str(args.envs),'--seed',str(seed),'--output_path',str(out),
                 '--task_graph','random_scenario']
            if args.agents:cmd += ['--num_agents',str(args.agents)]
            if args.objects:cmd += ['--num_objects',str(args.objects)]
            if args.group_spawn_over4:cmd += ['--group_spawn_over4']
            if args.enable_size_mode != 'native':cmd += ['--enable_size_mode',args.enable_size_mode]
            if args.coupled_assignment:cmd += ['--coupled_assignment',args.coupled_assignment]
            if task!='general':cmd += ['--eval_single_task',task]
            print(f'Running {args.profile}/{task} seed={seed}: {out}',flush=True)
            with (out/'run.log').open('w') as log:
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Run failed: {out}/run.log (exit {result.returncode})')
    print(json.dumps(summarize(directory),indent=2),flush=True)

if __name__=='__main__':main()

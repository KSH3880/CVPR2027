"""MD single-person comparison: source78000 versus PUSH fine-tune9000."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from utils.finetune_evaluation import atomic_json, summarize_rows, distance_bins


def summarize(root):
    data={'protocol':'single_match_region_v1','people_per_environment':1,'models':{}}
    lines=['# MD 기준 1인 평가: 원본78000 vs PUSH fine-tune9000','',
           '기존 2인 결과와 분리한다. 기존4과제900step, PUSH만1800step. 512환경×3seed의첫episode ever.', '',
           '| 과제 | 모델 | 완료seed | 시도 | 점 ever 평균±std | 영역 ever 평균±std |','|---|---|---|---:|---:|---:|']
    for label in ('source78000','push_ft9000'):
        model={}
        for task in ('holding_at','holding_ontop','sit','climb','push_at'):
            paths=sorted((root/label).glob(task+'_seed*/results.json'))
            runs=[json.loads(p.read_text()) for p in paths if (p.parent/'validation.json').exists()]
            if not runs:continue
            rows=[r for run in runs for r in run['agents']]
            item=dict(completed_seeds=[r['seed'] for r in runs],pooled=summarize_rows(rows))
            for metric in ('point_ever_rate','region_ever_rate'):
                rates=[r['summary'][metric] for r in runs]
                item[metric+'_mean']=float(np.mean(rates))
                item[metric+'_std_ddof1']=float(np.std(rates,ddof=1)) if len(rates)>1 else None
            if task=='push_at':item['by_initial_box_goal_distance']=distance_bins(rows)
            model[task]=item
            def format_rate(metric):
                mean=item[metric+'_mean']*100;std=item[metric+'_std_ddof1']
                return f'{mean:.2f}% ± {std*100:.2f}%p' if std is not None else f'{mean:.2f}%'
            lines.append(f"| {task} | {label} | {item['completed_seeds']} | {len(rows)} | {format_rate('point_ever_rate')} | {format_rate('region_ever_rate')} |")
        data['models'][label]=model
    data['comparison']={}
    for task in ('holding_at','holding_ontop','sit','climb'):
        a=data['models']['source78000'].get(task);b=data['models']['push_ft9000'].get(task)
        if a and b and len(a['completed_seeds'])==len(b['completed_seeds'])==3:
            data['comparison'][task]={metric+'_delta_percentage_points':
                (b[metric+'_mean']-a[metric+'_mean'])*100 for metric in ('point_ever_rate','region_ever_rate')}
    if data['comparison']:
        lines+=['','## 파인튜닝 − 원본 변화','', '| 과제 | 점 ever 변화 | 영역 ever 변화 |','|---|---:|---:|']
        for task,d in data['comparison'].items():
            lines.append(f"| {task} | {d['point_ever_rate_delta_percentage_points']:+.2f}%p | {d['region_ever_rate_delta_percentage_points']:+.2f}%p |")
    push=data['models']['push_ft9000'].get('push_at')
    if push:
        lines+=['','## 1인 PUSH 초기 상자→목표 거리별','', '| 거리 | 시도 | 성공률 |','|---|---:|---:|']
        for label,r in push['by_initial_box_goal_distance'].items():
            if r['agent_trials']:lines.append(f"| {label} | {r['agent_trials']} | {r['point_ever_rate']*100:.2f}% |")
    atomic_json(root/'summary.json',data)
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference-checkpoint',required=True)
    p.add_argument('--finetune-checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--num-envs',type=int,default=512)
    p.add_argument('--push-horizon',type=int,default=1800)
    args=p.parse_args();root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'run_manifest.json').exists():raise FileExistsError('Use a new single-person comparison directory')
    reference=str(Path(args.reference_checkpoint).resolve());finetune=str(Path(args.finetune_checkpoint).resolve())
    manifest=dict(protocol='single_match_region_v1',status='running',people=1,num_envs=args.num_envs,
        source=reference,finetune=finetune,seeds=[0,1,2],existing_horizon=900,push_horizon=args.push_horizon,
        planned_policy_runs=27,runs=[],started=datetime.datetime.now().isoformat())
    atomic_json(root/'run_manifest.json',manifest)
    for task in ('holding_at','holding_ontop','sit','climb','push_at'):
        objects=2 if task=='holding_ontop' else 1;horizon=args.push_horizon if task=='push_at' else 900
        for seed in (0,1,2):
            plan=root/'initial_states'/f'{task}_seed{seed}.npz';plan.parent.mkdir(exist_ok=True)
            common=['evaluate_single_match.py','--benchmark-task',task,'--reference-checkpoint',reference,
                '--initial-plan',str(plan),'--task','HumanoidMACarry','--num_agents','1','--num_objects',str(objects),
                '--num_envs',str(args.num_envs),'--episode_length',str(horizon),'--headless','--test','--eval','--no_video',
                '--motion_file','tokenhsi/data/dataset_loco_sit_carry_climb.yaml','--seed',str(seed)]
            # Generate each shared initial state once, with the original reset code.
            cpu_env=os.environ.copy();cpu_env.update(TOKENHSI_GPU='-1',CUDA_VISIBLE_DEVICES='-1')
            prepare=[sys.executable]+common+['--prepare-only','--sim_device','cpu','--pipeline','cpu',
                '--rl_device','cpu','--num_threads','1','--output_path',str(root/'initial_states')]
            print('[single batch] prepare',task,seed,flush=True)
            with plan.with_suffix('.log').open('w') as log:
                code=subprocess.run(prepare,env=cpu_env,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:
                manifest.update(status='failed',failure_stage='prepare',failed_task=task,failed_seed=seed)
                atomic_json(root/'run_manifest.json',manifest);raise SystemExit(f'Initial state generation failed: {plan.with_suffix(".log")}')
            models=[('push_ft9000',finetune)] if task=='push_at' else [('source78000',reference),('push_ft9000',finetune)]
            for label,checkpoint in models:
                out=root/label/f'{task}_seed{seed}';out.mkdir(parents=True)
                command=[sys.executable]+common+['--checkpoint',checkpoint,'--output_path',str(out),
                    '--sim_device','cuda:0','--pipeline','gpu','--rl_device','cuda:0']
                print('[single batch] start',label,task,seed,'people=1','objects='+str(objects),flush=True)
                with (out/'eval.log').open('w') as log:
                    code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
                manifest['runs'].append(dict(model=label,task=task,seed=seed,exit_code=code,output=str(out)))
                summarize(root)
                if code:
                    manifest['status']='failed';atomic_json(root/'run_manifest.json',manifest)
                    raise SystemExit(f'Policy evaluation failed: {out}/eval.log. No automatic GPU retry.')
                atomic_json(root/'run_manifest.json',manifest)
                print('[single batch] complete',label,task,seed,flush=True)
    manifest.update(status='complete',finished=datetime.datetime.now().isoformat())
    atomic_json(root/'run_manifest.json',manifest)


if __name__=='__main__':main()

"""Run each task/seed in its own process and retain completed results on failure."""
import argparse
import datetime
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from utils.finetune_evaluation import TASKS, atomic_json, summarize_batch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--num-envs',type=int,default=512)
    parser.add_argument('--tasks',nargs='+',choices=TASKS,default=list(TASKS))
    parser.add_argument('--push-episode-length',type=int,default=1200)
    args=parser.parse_args()
    if args.push_episode_length < 1 or len(set(args.tasks))!=len(args.tasks):
        parser.error('Use a positive PUSH horizon and unique tasks')
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    if (root/'run_manifest.json').exists():
        raise FileExistsError('Use a fresh evaluation output directory')
    checkpoint=Path(args.checkpoint).resolve()
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,
        gpu=os.environ['TOKENHSI_GPU'],num_envs=args.num_envs,seeds=[0,1,2],
        tasks=args.tasks,push_episode_length=args.push_episode_length,planned_runs=3*len(args.tasks),
        started=datetime.datetime.now().isoformat(),status='running',runs=[])
    atomic_json(root/'run_manifest.json',manifest)
    for task in args.tasks:
        for seed in (0,1,2):
            output=root/f'{task}_seed{seed}'
            output.mkdir()
            horizon=args.push_episode_length if task=='push_at' else 900
            command=[sys.executable,'evaluate_finetune.py','--task','HumanoidMAStage1Push',
                '--cfg_env','tokenhsi/data/cfg/multi_agent/stage1_push_finetune.yaml',
                '--cfg_train','tokenhsi/data/cfg/train/rlg/stage1_push_finetune.yaml',
                '--motion_file','tokenhsi/data/dataset_stage1_push_finetune.yaml',
                '--num_agents','2','--num_envs',str(args.num_envs),'--num_objects','4',
                '--headless','--no_video','--test','--eval','--eval_repeats','1',
                '--eval_skills','loco','--eval_skill_probs','1.0','--task_graph',task,
                '--episode_length',str(horizon),'--seed',str(seed),
                '--checkpoint',str(checkpoint),'--output_path',str(output)]
            print(f'[batch] start {task} seed={seed} horizon={horizon}',flush=True)
            with (output/'eval.log').open('w') as log:
                process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            manifest['runs'].append(dict(task=task,seed=seed,exit_code=process.returncode,
                output=str(output),command=command))
            summarize_batch(root)
            if process.returncode:
                manifest['status']='failed'
                atomic_json(root/'run_manifest.json',manifest)
                raise SystemExit(f'Evaluation failed ({task}, seed {seed}); inspect {output}/eval.log. No GPU retry.')
            atomic_json(root/'run_manifest.json',manifest)
            print(f'[batch] complete {task} seed={seed}',flush=True)
    manifest['status']='complete'
    manifest['finished']=datetime.datetime.now().isoformat()
    atomic_json(root/'run_manifest.json',manifest)


if __name__=='__main__':
    main()

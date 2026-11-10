import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(enable_sit=2,enable_climb=2,enable_ontop=2,coupled_2=2,coupled_3=3,coupled_4=4)

parser = argparse.ArgumentParser()
parser.add_argument('setting',choices=SETTINGS)
parser.add_argument('--envs',type=int,default=1)
parser.add_argument('--episodes',type=int,default=10)
parser.add_argument('--steps',type=int,default=900)
parser.add_argument('--seed',type=int,default=1)
parser.add_argument('--headless',action='store_true')
parser.add_argument('--capture',action='store_true')
parser.add_argument('--collision-probe',action='store_true')
parser.add_argument('--output',type=Path)
parser.add_argument('--stage1-checkpoint',type=Path,default=ROOT/'pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth')
args = parser.parse_args()
if args.collision_probe and (args.setting != 'enable_sit' or args.envs != 1 or args.steps <= 30):
    parser.error('Collision probe requires enable_sit, one env and more than 30 steps')
if min(args.envs,args.episodes,args.steps)<=0 or args.steps>900 or args.seed<0:
    parser.error('Use positive envs/episodes, 1..900 steps and a nonnegative seed')
checkpoint = ROOT/'output/stage2_distill_900_v2/Stage2Distill900V2_08-19-13-51/nn/Stage2Distill900V2_00002000.pth'
for path in (checkpoint,args.stage1_checkpoint):
    if not path.is_file():
        parser.error('Missing checkpoint: '+str(path))
output = (args.output or ROOT/'output_etc/check'/('fsm_'+args.setting+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
if ROOT/'output_etc/check' not in output.parents:
    parser.error('Output must be under output_etc/check')
if output.exists() and any(output.iterdir()):
    parser.error('Output already contains files; use a new directory')
output.mkdir(parents=True,exist_ok=True)
people = SETTINGS[args.setting]
command = [sys.executable,str(ROOT/'FSM/entry.py'),'--task','HumanoidMACoupledPush',
    '--cfg_train','tokenhsi/data/cfg/train/rlg/stage2_distill_900_v2.yaml','--cfg_env','',
    '--motion_file','tokenhsi/data/dataset_stage1_jointpush.yaml','--checkpoint',str(checkpoint),
    '--seed',str(args.seed),'--num_envs',str(args.envs),'--num_agents',str(people),'--num_objects',str(2*people),
    '--test','--eval','--no_video','--eval_protocol','general','--eval_repeats','1',
    '--task_graph','random_scenario','--enable_size_mode','per_action_intersection','--coupled_assignment','ordered',
    '--output_path',str(output),'--task_camera','stack']
if args.headless:
    command.append('--headless')
environment = dict(os.environ,FSM_SETTING=args.setting,FSM_STEPS=str(args.steps),FSM_EPISODES=str(args.episodes),
                   FSM_CAPTURE=str(int(args.capture)),FSM_STAGE1_CHECKPOINT=str(args.stage1_checkpoint.resolve()),
                   FSM_COLLISION_PROBE=str(int(args.collision_probe)))
(output/'command.json').write_text(json.dumps(dict(command=command,stage1_checkpoint=environment['FSM_STAGE1_CHECKPOINT'],
    setting=args.setting,steps=args.steps,episodes=args.episodes,collision_probe=args.collision_probe,
    gpu=os.environ.get('TOKENHSI_GPU')),indent=2))
print('[FSM] output='+str(output),flush=True)
sys.exit(subprocess.call(command,cwd=ROOT,env=environment))

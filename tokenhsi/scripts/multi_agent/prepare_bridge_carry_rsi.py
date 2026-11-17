import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import isaacgym
import torch

from env.tasks.multi_agent.bridge_carry_rsi_cache import signature
from env.tasks.multi_agent.bridge_carry_rsi_probe import CarryProbe, profiles, screen
from utils.config import get_args, parse_sim_params, set_seed
from utils.experiments import native_config

parser=argparse.ArgumentParser()
parser.add_argument('--cache-output',required=True)
options,remaining=parser.parse_known_args()
sys.argv=[sys.argv[0]]+remaining
args=get_args()
args.eval=args.test=args.headless=True
args.output_path=options.cache_output
Path(args.output_path).mkdir(parents=True,exist_ok=True)
target=native_config('stage2_sensor_900_bridge_before_fixed_carry_rsi')
cfg=native_config('stage1'); train=native_config('stage1',True)
for name in ('asset','plane','powerScale','enableSelfCollisionDetection','controlFrequencyInv'):
    cfg['env'][name]=target['env'][name]
cfg['sim']=target['sim']
cfg['carry_profiles']=profiles(target['env'])
cfg['env'].update(numEnvs=len(cfg['carry_profiles']),motion_file='tokenhsi/data/dataset_stage1_jointpush.yaml')
cfg['args']=args
set_seed(42);torch.set_num_threads(2)
task=CarryProbe(cfg,parse_sim_params(args,cfg,train),args.physics_engine,args.device,args.device_id,True)
key=signature(target,task._motion_lib['carryWith'])
records=screen(task)
data=dict(signature=key,records=records)
path=Path(options.cache_output)/(key+'.pt')
torch.save(data,path.with_suffix('.pending'));path.with_suffix('.pending').replace(path)
summary=dict(path=str(path),profiles=len(records),supported=sum(bool(v) for v in records.values()),frames=sum(map(len,records.values())))
(path.parent/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
task.gym.destroy_sim(task.sim)
if not summary['supported']: raise RuntimeError('No valid carry RSI profiles')

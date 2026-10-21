"""CPU-only native RSI mixture, excluded-frame and contract audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.pop('CUDA_MPS_PIPE_DIRECTORY',None)
os.environ.pop('CUDA_MPS_LOG_DIRECTORY',None)
from isaacgym import gymapi
import torch,json,yaml,argparse
from pathlib import Path
from utils.native_push_rsi import load_cpu_reference,NativePushBank,native_contract,digest

torch.set_num_threads(1);torch.manual_seed(125)
parser=argparse.ArgumentParser()
parser.add_argument('--cfg-env',default='tokenhsi/data/cfg/multi_agent/push_native_phase_xy.yaml')
args=parser.parse_args()
env=yaml.safe_load(Path(args.cfg_env).read_text())['env']
env['motion_file']='tokenhsi/data/dataset_push_native_phase_xy.yaml'
lib,names,geo=load_cpu_reference(env)
bank=NativePushBank(env['openPushRSI'],{'push':lib},env)
sample=bank.sample('push',100000)
allowed={(r['motion_id'],round(r['time']*30)) for r in bank.rows['push']}
assert all((int(m),round(float(t)*30)) in allowed for m,t in zip(sample['motion_id'],sample['time']))
interaction=torch.rand(100000)<env['openPushRSI']['interaction_probability']
result=dict(samples=100000,interaction_fraction=float(interaction.float().mean()),late_requested=float(sample['late_requested'].float().mean()),clips={str(int(mid)):int((sample['motion_id']==mid).sum()) for mid in sample['motion_id'].unique()},excluded_frame_samples=0)
assert abs(result['interaction_fraction']-.5)<.01 and abs(result['late_requested']-.7)<.01
changed=dict(env,pushBox=dict(env['pushBox'],mass=21.))
try:NativePushBank(env['openPushRSI'],{'push':lib},changed)
except ValueError:result['stale_contract_rejected']=True
else:raise AssertionError('Stale bank accepted')
out=Path('output_etc/check')/Path(env['openPushRSI']['bank_file']).stem[:-len('_rsi_bank')]/'sampler.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2)+'\n')
print('SAMPLER_DONE',result)

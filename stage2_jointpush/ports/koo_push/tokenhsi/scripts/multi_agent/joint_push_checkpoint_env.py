"""Materialize the saved JointPush environment for strict checkpoint evaluation."""
import argparse
from pathlib import Path
import tempfile
import torch
import yaml

p=argparse.ArgumentParser()
p.add_argument('checkpoint');p.add_argument('template');p.add_argument('directory');p.add_argument('--koo-rsi',action='store_true');p.add_argument('--require-aligned',action='store_true')
a=p.parse_args()
checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
contract=checkpoint.get('open_push_contract',{})
saved=checkpoint.get('relation_experiment_config',{}).get('env')
if not saved or contract.get('policy')!='push_at_ca4':
    raise ValueError('Expected a JointPush Stage2 checkpoint with saved environment')
if a.require_aligned and saved['jointPush'].get('goal_mode')!='box_forward':
    raise ValueError('Resume requires an aligned-goal checkpoint; start a new aligned run from Stage1')
if saved.get('pushBox')!=contract.get('push_box'):
    raise ValueError('Saved environment and checkpoint box contract disagree')
config=yaml.safe_load(Path(a.template).read_text())
config['env']=saved
if a.koo_rsi:
    template=yaml.safe_load(Path(a.template).read_text())['env']
    config['env']['stateInit']=template['stateInit']
    config['env']['jointPush']['rsi']=template['jointPush']['rsi']
    config['env']['jointPush']['koo_rsi']=template['jointPush']['koo_rsi']
config.pop('experiment',None)
directory=Path(a.directory);directory.mkdir(parents=True,exist_ok=True)
with tempfile.NamedTemporaryFile(mode='w',prefix='checkpoint_env_',suffix='.yaml',dir=directory,delete=False) as f:
    yaml.safe_dump(config,f,sort_keys=False)
    print(f.name)

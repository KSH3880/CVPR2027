"""Explicitly migrate a 30D PUSH checkpoint without changing its tensors."""
import argparse
from pathlib import Path
import torch
import yaml
from utils.push_only import migrate_checkpoint

parser=argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('destination')
args=parser.parse_args()
destination=Path(args.destination)
if destination.exists():raise FileExistsError(destination)
env=yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors.yaml').read_text())['env']
env['motion_file']='tokenhsi/data/dataset_push_only_push180.yaml'
weights=torch.load(args.source,map_location='cpu',weights_only=False)
converted=migrate_checkpoint(weights,env,'tokenhsi/data/cfg/train/rlg/open_push_unified_amp.yaml')
destination.parent.mkdir(parents=True,exist_ok=True)
torch.save(converted,destination)
print('Saved',destination,'epoch',converted.get('epoch'),'model/optimizer/normalization tensors preserved')

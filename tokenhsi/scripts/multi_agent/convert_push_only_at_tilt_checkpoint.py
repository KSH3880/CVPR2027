"""Create a 570D physical PUSH checkpoint with only the AT tilt gate added."""
import argparse
from pathlib import Path
import torch
from utils.at_tilt_checkpoint import migrate_push_only_checkpoint

parser=argparse.ArgumentParser()
parser.add_argument('source');parser.add_argument('destination')
args=parser.parse_args();destination=Path(args.destination)
if destination.exists():raise FileExistsError(destination)
weights=torch.load(args.source,map_location='cpu',weights_only=False)
converted=migrate_push_only_checkpoint(weights)
converted['at_tilt_migration']['source_checkpoint']=str(Path(args.source).resolve())
destination.parent.mkdir(parents=True,exist_ok=True)
torch.save(converted,destination)
print('Saved',destination,'epoch',converted['epoch'],'observation570D, object30D, all learned state preserved')

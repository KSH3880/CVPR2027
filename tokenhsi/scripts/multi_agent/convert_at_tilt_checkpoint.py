"""Create a separate AT tilt-limited resume checkpoint, retaining learned state."""
import argparse
from pathlib import Path
import torch
from utils.at_tilt_checkpoint import migrate_checkpoint

parser=argparse.ArgumentParser()
parser.add_argument('source');parser.add_argument('destination')
args=parser.parse_args();destination=Path(args.destination)
if destination.exists():raise FileExistsError(destination)
weights=torch.load(args.source,map_location='cpu',weights_only=False)
converted=migrate_checkpoint(weights)
destination.parent.mkdir(parents=True,exist_ok=True)
torch.save(converted,destination)
print('Saved',destination,'epoch',converted['epoch'],'all model/optimizer/RMS/counters preserved')

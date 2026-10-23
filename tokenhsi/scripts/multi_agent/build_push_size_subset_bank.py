"""CPU-only derivation of the 1.20..1.40m RSI bank from the original superset."""
from isaacgym import gymapi  # before modules which import torch
from pathlib import Path
import json
import yaml
from utils.push_size_subset import project_size_subset

BASE=Path('tokenhsi/data/cfg/multi_agent')
source=yaml.safe_load((BASE/'open_push_contact_at_friction06_random_size_goal1to25_at01_no_doors.yaml').read_text())['env']
target=yaml.safe_load((BASE/'push_size120to140_at_tiltfade.yaml').read_text())['env']
for env in (source,target):env['motion_file']='tokenhsi/data/dataset_push_only_push180.yaml'
data=project_size_subset(source,target)
out=Path(target['openPushRSI']['bank_file'])
if out.exists():
    if json.loads(out.read_text())!=data:raise ValueError('Refusing to replace an existing bank with different content')
else:
    with out.open('x') as f:f.write(json.dumps(data,indent=2)+'\n')
print(out, 'candidates',len(data['candidates']['push']),'sizes',len(data['size_subset_provenance']['retained_sizes_m']))

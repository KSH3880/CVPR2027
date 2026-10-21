"""Separate zero-shot diagnostic configs; never modify source/production configs."""
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

TOKENHSI = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TOKENHSI))
from utils.partition_push_paths import resolve_stage1_checkpoint

out = Path(sys.argv[1]).resolve()
source = resolve_stage1_checkpoint(sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None)
saved = torch.load(str(source), map_location='cpu', weights_only=False)
env_path = TOKENHSI / 'data/cfg/multi_agent/partition_push_surround_cpa05.yaml'
env = yaml.safe_load(env_path.read_text())
out.mkdir(parents=True, exist_ok=True)
(out / 'env.yaml').write_text(yaml.safe_dump(env, sort_keys=False))
(out / 'train.yaml').write_text(yaml.safe_dump(saved['relation_experiment_config']['train'], sort_keys=False))
manifest = dict(source_checkpoint=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_epoch=int(saved['epoch']), environment=str(env_path), agents=4,
                physical_doors=0, task_region_tokens=4, ordinary_box_tokens=4,
                learning=False, coordination_head=False, checkpoint_strict=True,
                intervention='Door-free variable-group Partition environment with original Stage1, source RMS frozen')
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(source)

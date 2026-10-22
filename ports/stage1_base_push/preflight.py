"""Reject incompatible checkpoints on CPU before constructing a simulator."""
import sys
import torch
import yaml
from utils.stage1_push_spec import check_checkpoint

env=yaml.safe_load(open('tokenhsi/data/cfg/multi_agent/stage1_base_push.yaml'))['env']
env['motion_file']='tokenhsi/data/dataset_stage1_base_push.yaml'
check_checkpoint(torch.load(sys.argv[1],map_location='cpu'),env,
                 for_training=len(sys.argv)>2 and sys.argv[2]=='train')
print('[Stage1BasePush] checkpoint contract OK')

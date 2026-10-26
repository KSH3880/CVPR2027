"""Reject incompatible checkpoints on CPU before constructing a simulator."""
import sys
import torch
import yaml
from utils.stage1_push_spec import check_checkpoint

variant=sys.argv[3] if len(sys.argv)>3 else 'stage1_base_push'
if variant not in ('stage1_base_push','stage1_push_finetune'):
    raise ValueError('Unknown Stage1 PUSH variant')
env=yaml.safe_load(open('tokenhsi/data/cfg/multi_agent/'+variant+'.yaml'))['env']
env['motion_file']='tokenhsi/data/'+('dataset_stage1_push_finetune.yaml' if variant=='stage1_push_finetune' else 'dataset_stage1_base_push.yaml')
check_checkpoint(torch.load(sys.argv[1],map_location='cpu'),env,
                 for_training=len(sys.argv)>2 and sys.argv[2]=='train')
print('[Stage1BasePush] checkpoint contract OK')

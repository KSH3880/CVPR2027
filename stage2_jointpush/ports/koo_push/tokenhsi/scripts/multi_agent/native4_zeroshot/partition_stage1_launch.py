"""Strict original Stage1 weights in the door-free Partition Push environment."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import isaacgym
from env.tasks.multi_agent.humanoid_ma_partition_push import HumanoidMAPartitionPush
from learning.multi_agent.ma_players import MAPlayerContinuous

sys.path.insert(0, str(Path(__file__).parent))
from attention_probe import evaluate_attention


def validate_stage1(self, checkpoint):
    import yaml
    saved = checkpoint['relation_experiment_config']
    target = yaml.safe_load(Path(self.cfg['args'].cfg_train).read_text())
    assert saved['train']['params']['network'] == target['params']['network']
    assert checkpoint['relation_metadata']['schema_version'] == 12
    assert (saved['num_agents'], saved['num_objects']) == (2, 4)
    assert 'joint_push' not in checkpoint.get('open_push_contract', {})
    assert checkpoint.get('open_push_contract', {}).get('policy') != 'partition_push_ca4'
    assert (self.num_agents, self.num_objects) == (4, 8)
    print('[Partition Stage1 probe] Explicit environment intervention; original Stage1 strict restore, no CA/head expansion', flush=True)


HumanoidMAPartitionPush.validate_inference_checkpoint = validate_stage1
MAPlayerContinuous.run = evaluate_attention
runpy.run_path('tokenhsi/scripts/multi_agent/partition_push_launch.py', run_name='__main__')

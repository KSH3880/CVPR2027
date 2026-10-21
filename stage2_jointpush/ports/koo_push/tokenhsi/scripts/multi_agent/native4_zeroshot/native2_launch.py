"""Original two-person PUSH evaluation, with optional actor door-token removal."""
import runpy
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import isaacgym  # Before torch/Isaac imports.
from env.tasks.multi_agent.humanoid_ma_open_push import HumanoidMAOpenPush
from learning.multi_agent.ma_players import MAPlayerContinuous
from utils.relation_task_spec import check_checkpoint_metadata, checkpoint_metadata
import utils.open_push_rsi as rsi

sys.path.insert(0, str(Path(__file__).parent))
from attention_probe import evaluate_attention


def validate_source(self, checkpoint):
    check_checkpoint_metadata(checkpoint, checkpoint_metadata(self._relation_cfg))
    saved = dict(checkpoint['open_push_contract'])
    actual = self.get_open_push_contract()
    # Match the native4 diagnostic's explicit installed-motion override.
    saved.pop('manifest_sha256', None)
    actual.pop('manifest_sha256', None)
    if saved != actual:
        raise ValueError('Source contract differs outside installed-motion override')
    assert (self.num_agents, self.num_objects) == (2, 4)


HumanoidMAOpenPush.validate_inference_checkpoint = validate_source
# Evaluation never uses interaction RSI. Preserve the original loco/Start path.
rsi.bank_contract = lambda env: {}
rsi.OpenPushRSIBank = lambda *args, **kwargs: None
if os.environ.get('NATIVE2_ATTENTION_CAPTURE') == '1':
    from attention_capture import capture_run
    MAPlayerContinuous.run = capture_run
else:
    MAPlayerContinuous.run = evaluate_attention
runpy.run_path('tokenhsi/scripts/multi_agent/partition_push_launch.py', run_name='__main__')

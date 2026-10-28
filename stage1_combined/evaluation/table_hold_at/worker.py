"""Evaluation-only process; no training code or live run is patched on disk."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tokenhsi'), str(ROOT)]
import isaacgym  # Must precede torch.
import torch
from learning.multi_agent.ma_players import MAPlayerContinuous
from evaluation.table_hold_at.backend import run_eval
original_get_action = MAPlayerContinuous.get_action


def shared_eval(player):
    cfg = player.env.task.cfg['env']['holdAtFT'].get('public_policy')
    if cfg:
        # PyTorch2.0 fused SDP kernels do not support this RTX5090.
        # Use the same attention equations through the math implementation.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        from evaluation.hold_at.public_policy import PublicChairPolicy
        devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            player.public_table_policy = PublicChairPolicy(player.env.task, player, cfg)
    return run_eval(player)


def action(player, obs, is_determenistic=False):
    if hasattr(player, 'public_table_policy'):
        return player.public_table_policy.action()
    return original_get_action(player, obs, is_determenistic)


MAPlayerContinuous.run_eval = shared_eval
MAPlayerContinuous.get_action = action
import run as app
try:
    app.main()
finally:
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()

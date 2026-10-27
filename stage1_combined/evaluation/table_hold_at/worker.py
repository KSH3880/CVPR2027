"""Evaluation-only process; no training code or live run is patched on disk."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tokenhsi'), str(ROOT)]
import isaacgym  # Must precede torch.
import torch
from learning.multi_agent.ma_players import MAPlayerContinuous
from evaluation.table_hold_at.backend import run_eval
MAPlayerContinuous.run_eval = run_eval
import run as app
try:
    app.main()
finally:
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()

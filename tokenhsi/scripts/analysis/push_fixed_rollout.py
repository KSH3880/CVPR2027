"""Opt-in fixed PUSH experiment; original natural entry points stay available."""
import os
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tokenhsi'))
import isaacgym
from push_fixed_layout import install
install()
script='search_push_wait.py' if os.environ.get('PUSH_PREFILTER_ONLY')=='1' else 'push_comparison_rollout.py'
runpy.run_path(str(Path(__file__).parent/script),run_name='__main__')

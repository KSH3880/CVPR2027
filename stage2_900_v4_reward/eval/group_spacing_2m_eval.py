import os
from pathlib import Path
import runpy
import subprocess

ROOT=Path(__file__).resolve().parents[1]
gpu=os.environ.get('TOKENHSI_GPU','')
index=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=index',
    '--format=csv,noheader,nounits'],text=True).strip() if gpu else ''
if index not in ('5','6','7') or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu:
    raise ValueError('The2m evaluation requires physical GPU5/6/7')
entry='group_spacing_1m_eval.py' if index=='5' and os.environ['SPACING_MODEL']=='coopft' else 'group_spacing_eval.py'
runpy.run_path(str(ROOT/'eval'/entry),run_name='__main__')

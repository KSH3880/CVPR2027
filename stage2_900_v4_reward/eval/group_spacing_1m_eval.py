import inspect
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if os.environ['SPACING_MODEL']=='coopft':
    gpu=os.environ.get('TOKENHSI_GPU','')
    index=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=index',
                                   '--format=csv,noheader,nounits'],text=True).strip() if gpu else ''
    if index!='5' or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu:
        raise ValueError('The1m Coop. FT adapter requires physical GPU5')
    sys.path.insert(0,str(ROOT.parent/'sct_coopft/tokenhsi'))
    import isaacgym
    from utils import config
    source=inspect.getsource(config.load_cfg)
    anchor="index not in ('6', '7')"
    if source.count(anchor)!=1:
        raise ValueError('Coop. FT GPU validation source changed')
    source=source.replace(anchor,"index not in ('5',)")
    source=source.replace('sct_coopft requires GPU6/7','The1m Coop. FT adapter requires GPU5')
    namespace=dict(config.load_cfg.__globals__)
    exec(compile(source,str(ROOT/'eval/group_spacing_1m_eval.py'),'exec'),namespace)
    config.load_cfg=namespace['load_cfg']

runpy.run_path(str(ROOT/'eval/group_spacing_eval.py'),run_name='__main__')

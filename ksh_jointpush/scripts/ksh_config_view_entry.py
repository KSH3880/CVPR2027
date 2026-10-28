"""Bridge the documented KSH test wrapper to a frozen current-training config."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def resolved_args(argv, cfg_dir, physical_gpu=None):
    result=list(argv)
    for flag,name in [('--cfg_env','env.yaml'),('--cfg_train','train.yaml'),('--motion_file','motions.yaml')]:
        result[result.index(flag)+1]=str(Path(cfg_dir)/name)
    if physical_gpu is not None:
        for flag in ['--sim_device','--rl_device']:
            value=f'cuda:{physical_gpu}'
            if flag in result:result[result.index(flag)+1]=value
            else:result += [flag,value]
    return result


def main():
    cfg=Path(os.environ['KSH_DOCUMENTED_VIEW_CONFIG'])
    gpu=None
    if os.environ.get('HEADLESS')=='0':
        gpu=int(os.environ['TOKENHSI_GUI_GPU_INDEX'])
        if gpu!=5:raise ValueError('This documented viewer is prepared for GPU5')
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(subprocess.check_output(
            ['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).split())
        os.environ.pop('CUDA_MPS_PIPE_DIRECTORY',None)
        os.environ.pop('CUDA_MPS_LOG_DIRECTORY',None)
    sys.argv=resolved_args(sys.argv,cfg,gpu)
    Path(os.environ['OUTPUT_PATH']).mkdir(parents=True,exist_ok=True)
    (Path(os.environ['OUTPUT_PATH'])/'documented_view_args.json').write_text(json.dumps(sys.argv,indent=2))
    bootstrap=Path(__file__).with_name('ksh_jointpush_run.py')
    sys.argv[0]=str(bootstrap)
    runpy.run_path(str(bootstrap),run_name='__main__')

if __name__=='__main__':main()

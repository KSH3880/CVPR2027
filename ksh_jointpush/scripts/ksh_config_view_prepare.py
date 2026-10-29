"""Prepare the config.md VNC/test route without editing the frozen KSH snapshot."""
from pathlib import Path
import argparse,datetime,json,shutil,hashlib,shlex
ROOT=Path(__file__).resolve().parents[2]

def prepare(checkpoint=None):
    active=json.loads((ROOT/'runs/jointpush/latest_partial_training.json').read_text())
    ckpt=Path(checkpoint).resolve() if checkpoint else sorted((Path(active['experiment_dir'])/'nn').glob('Stage2JointPushRsi_[0-9]*.pth'))[-1]
    if not ckpt.is_file():raise FileNotFoundError(ckpt)
    run=ROOT/'ksh_jointpush/output_etc/check'/('config_md_view_gpu5_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    run.mkdir(parents=True)
    source=run/'source';baseline=ROOT/'ksh_jointpush'
    for name in ['tokenhsi','lpanlib','mps','body_models']:
        shutil.copytree(baseline/name,source/name,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(Path(active['launch'])/'configs',run/'configs');shutil.copy2(ckpt,run/'checkpoint.pth')
    helpers=source/'tokenhsi/scripts/multi_agent'
    # Keep the documented VNC/test scripts byte-identical. Only the GUI service
    # wrapper/helper use the already verified GPU5 native rendering path.
    gui=(ROOT/'jointpush/scripts/run_gui.sh').read_text().replace('ROOT=$(cd "$(dirname "$0")/../ports/koo_native_y2m" && pwd)','ROOT='+shlex.quote(str(source)))
    (helpers/'run-gui.sh').write_text(gui)
    shutil.copy2(ROOT/'jointpush/ports/koo_native_y2m/tokenhsi/scripts/multi_agent/gui_gpu_env.sh',helpers/'gui_gpu_env.sh')
    names=['stage2_jointpush_rsi_vnc.sh','stage2_jointpush_rsi_test.sh']
    hashes={}
    for name in names:
        assert (helpers/name).read_bytes()==(baseline/'tokenhsi/scripts/multi_agent'/name).read_bytes()
        hashes[name]=hashlib.sha256((helpers/name).read_bytes()).hexdigest()
    env='''#!/usr/bin/env bash
set -eu
export TOKENHSI_GPU=5 TOKENHSI_CONDA_ENV=${TOKENHSI_CONDA_ENV:-tokenhsi118} TOKENHSI_GUI_GPU_MODE=native PYTHONDONTWRITEBYTECODE=1
export MPS_GPU_ROOT=/tmp/ksh-jointpush-native-no-mps
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY PUSH_RSI_MIX KSH_RSI_MODE_CHECK
export VNC_DIR="${VNC_DIR:-$HOME/opt/vnc}"
export TORCH_EXTENSIONS_DIR='''+shlex.quote(str(ROOT/'runs/jointpush/torch_extensions'))+''' CXX=/usr/bin/g++-11
export KSH_DOCUMENTED_VIEW_CONFIG='''+shlex.quote(str(run/'configs'))+'''
export JOINTPUSH_ENTRYPOINT='''+shlex.quote(str(ROOT/'ksh_jointpush/scripts/ksh_config_view_entry.py'))+'''
export OUTPUT_PATH='''+shlex.quote(str(source/'output_etc/check/viewer'))+'''
export PORT=${PORT:-6095}
cd '''+shlex.quote(str(source))+'''
'''
    cmd='exec bash tokenhsi/scripts/multi_agent/stage2_jointpush_rsi_vnc.sh '+shlex.quote(str(run/'checkpoint.pth'))+' 4 1 8 "${REPEATS:-100}"\n'
    (run/'run.sh').write_text(env+cmd)
    (run/'check_cpu.sh').write_text(env+'export CUDA_VISIBLE_DEVICES=\'\' HEADLESS=1 SIM_DEVICE=cpu KSH_RSI_MODE_CHECK=1\nexport OUTPUT_PATH='+shlex.quote(str(source/'output_etc/check/cpu_check'))+'\nexec bash tokenhsi/scripts/multi_agent/stage2_jointpush_rsi_test.sh '+shlex.quote(str(run/'checkpoint.pth'))+' 4 1 8 1\n')
    (run/'preparation.json').write_text(json.dumps(dict(checkpoint_source=str(ckpt),checkpoint_sha256=hashlib.sha256((run/'checkpoint.pth').read_bytes()).hexdigest(),documented_scripts_sha256=hashes,original_snapshot_unchanged=True,default_initialization='loco100 per config.md; PUSH_RSI=1 enables push100',gpu=5,port=6095,source=str(source)),indent=2))
    return run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');a=p.parse_args();print(prepare(a.checkpoint))

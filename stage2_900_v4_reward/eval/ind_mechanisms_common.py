import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output_etc/check/ind_mechanisms_v4_2000'
OURS = ROOT/'output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16/nn/Stage2Distill900V4_00002000.pth'
SCT = ROOT/'pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth'
REFERENCE = ROOT/'output_etc/check/v3_2000_tables12_seeds167_20261110/reference/table1/independent_2h'
SUMMARY = ROOT/'output_etc/check/v4_2000_tables12_radius_seeds167_20261112/summary.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def command(entry, output, seed, people, gpu, extra=None):
    args = [sys.executable, str(ROOT/'eval'/entry), '--task', 'HumanoidMACoupledPush',
        '--cfg_train', 'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v4.yaml', '--cfg_env', '',
        '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml', '--checkpoint', str(OURS),
        '--seed', str(seed), '--num_envs', '512', '--num_agents', str(people),
        '--num_objects', str(people*2), '--test', '--eval', '--headless', '--no_video',
        '--eval_protocol', 'general', '--eval_repeats', '1', '--task_graph', 'random_scenario',
        '--enable_size_mode', 'per_action_intersection', '--coupled_assignment', 'ordered',
        '--output_path', str(output)]
    env = dict(os.environ, TOKENHSI_GPU=str(gpu), STAGE2_SCENE_MODE='independent',
               PYTHONHASHSEED=str(seed), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for key in ('CHECK', 'MAX_ITERATIONS', 'REFERENCE_ONLY', 'STAGE1_POLICY_CHECKPOINT',
                'TABLE2_SETTING', 'FSM_STAGE1_CHECKPOINT', 'BIAS_ABLATION', 'IND_SCENE', 'IND_STEPS'):
        env.pop(key, None)
    env.update(extra or {})
    return ['bash', '-c', 'source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"', '_', *args], env

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output_etc/check/ind_dct_ablation_v4_2000_env64_seed1'
CHECKPOINT = ROOT/'output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16/nn/Stage2Distill900V4_00002000.pth'
REFERENCE = OUT/'reference/general_seed1'


def execute(mode, gpu, reference=False):
    path = REFERENCE if reference else OUT/mode
    if path.exists():
        raise ValueError('Refusing to overwrite '+str(path))
    while True:
        used = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        if used + 6144 <= 70000:
            break
        print(f'GPU{gpu}: waiting, {used} MiB occupied', flush=True)
        time.sleep(20)
    path.mkdir(parents=True)
    env = dict(os.environ)
    for key in ('CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY', 'REFERENCE_ONLY',
                'TABLE2_SETTING', 'STAGE1_POLICY_CHECKPOINT', 'FSM_STAGE1_CHECKPOINT',
                'CHECK', 'MAX_ITERATIONS', 'RESUME_CHECKPOINT', 'GROUP_SPAWN_RADIUS',
                'SPACING_METERS', 'STAGE2_SAMPLING_AUDIT', 'BIAS_ABLATION'):
        env.pop(key, None)
    env.update(TOKENHSI_GPU=str(gpu), STAGE2_SCENE_MODE='independent',
        STAGE2_REFERENCE_ROOT=str(REFERENCE.parent), IND_DCT_ABLATION=mode,
        PYTHONHASHSEED='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if reference:
        env['REFERENCE_ONLY'] = '1'
    command = [sys.executable, str(ROOT/'eval/ind_dct_ablation_eval.py'),
        '--task', 'HumanoidMACoupledPush', '--cfg_train', 'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v4.yaml',
        '--cfg_env', '', '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml',
        '--checkpoint', str(CHECKPOINT), '--seed', '1', '--num_envs', '64',
        '--num_agents', '2', '--num_objects', '4', '--test', '--eval', '--headless', '--no_video',
        '--eval_protocol', 'general', '--eval_repeats', '1', '--task_graph', 'random_scenario',
        '--enable_size_mode', 'per_action_intersection', '--coupled_assignment', 'ordered',
        '--output_path', str(path)]
    name = 'reference' if reference else mode
    print(f'Start {name} GPU{gpu}', flush=True)
    with (OUT/'logs'/f'{name}.log').open('w') as log:
        subprocess.run(['bash', '-c', 'source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"',
                        '_', *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'Finished {name} GPU{gpu}', flush=True)


def main():
    (OUT/'logs').mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(dict(envs=64, agents=2, seed=1, horizon=900,
        checkpoint=str(CHECKPOINT), checkpoint_sha256=digest, modes=['none', 'self_only', 'bypass'],
        gpus=[6, 7], inference_only=True), indent=2))
    execute('none', 6, reference=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(execute, 'none', 6)
        second = pool.submit(execute, 'self_only', 7)
        first.result()
        execute('bypass', 6)
        second.result()
    if digest != hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest():
        raise ValueError('Checkpoint changed')
    print('Three evaluations complete; checkpoint preserved', flush=True)


if __name__ == '__main__':
    main()

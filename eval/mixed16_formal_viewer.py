import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TEACHER = ROOT.parent / 'stage1_2_base_teacher'
CHECKPOINT = TEACHER / 'output/stage2_distill_900_v4/Stage2Distill900V4_11-21-37-16/nn/Stage2Distill900V4_00002000.pth'
CHECKPOINT_SHA = 'a1be127ad7118ca137f9de6079a9d4bb123a9d34a3c2c9c9aef24b15290f055a'


def specification(args):
    checks = ROOT / 'output_etc/check'
    run = (args.output or checks / 'mixed16_formal_vnc_NEW_RUN').resolve()
    if checks.resolve() not in run.parents:
        raise ValueError('--output must be below ' + str(checks))
    reference = run / 'reference' / ('general_seed' + str(args.seed))
    output = reference if args.capture_reference else run / 'rollout'
    for path in (CHECKPOINT, TEACHER / 'eval/group_spacing_eval.py'):
        if not path.is_file():
            raise FileNotFoundError(path)
    command = ['--task', 'HumanoidMACoupledPush', '--cfg_train',
        'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v4.yaml', '--cfg_env', '',
        '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml',
        '--checkpoint', str(CHECKPOINT), '--seed', str(args.seed),
        '--num_envs', '1', '--num_agents', '16', '--num_objects', '32',
        '--test', '--eval', '--headless', '--no_video', '--eval_protocol', 'general',
        '--eval_repeats', '1', '--task_graph', 'random_scenario',
        '--enable_size_mode', 'per_action_intersection', '--coupled_assignment', 'ordered',
        '--group_spawn_over4', '--output_path', str(output)]
    return dict(checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA,
        reference=str(reference), run=str(run), output=str(output), seed=args.seed,
        envs=1, agents=16, objects=32, horizon=900, radius=7.5, command=command,
        controls='Space: start/pause; F: camera follow; Esc: exit',
        note='New one-env sample, not a scene from the 512-env evaluation. Not yet GPU validated.')


def main():
    parser = argparse.ArgumentParser(description='Ours V4 2k: one 16H Mixed scene using the formal evaluation sampler.')
    parser.add_argument('--seed', type=int, default=1, help='Any nonnegative integer.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--capture-reference', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--dry-run', action='store_true', help='Print settings without GPU imports, files or VNC services.')
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    spec = specification(args)
    if args.dry_run:
        print(json.dumps(spec, indent=2))
        return
    if os.environ.get('PYTHONHASHSEED') != str(args.seed):
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        os.execv(sys.executable, [sys.executable, *sys.argv])

    digest = hashlib.sha256()
    with CHECKPOINT.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != CHECKPOINT_SHA:
        raise ValueError('Ours V4 2k checkpoint SHA differs')
    if args.output is None:
        checks = ROOT / 'output_etc/check'
        checks.mkdir(parents=True, exist_ok=True)
        args.output = Path(tempfile.mkdtemp(prefix='mixed16_formal_vnc_', dir=str(checks)))
        spec = specification(args)
    elif not args.capture_reference:
        Path(spec['run']).mkdir(parents=True, exist_ok=False)
    destination = Path(spec['output'])
    destination.mkdir(parents=True, exist_ok=False)
    if not args.capture_reference:
        (Path(spec['run']) / 'viewer_manifest.json').write_text(json.dumps(spec, indent=2))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--capture-reference',
            '--seed', str(args.seed), '--output', spec['run']], check=True)
    for name in ('REFERENCE_ONLY', 'TABLE2_SETTING', 'STAGE1_POLICY_CHECKPOINT',
                 'FSM_STAGE1_CHECKPOINT', 'CHECK', 'MAX_ITERATIONS', 'RESUME_CHECKPOINT',
                 'STAGE2_SAMPLING_AUDIT', 'BIAS_ABLATION', 'DESIGN'):
        os.environ.pop(name, None)
    os.environ.update(SPACING_MODEL='ours', SPACING_METERS='native', GROUP_SPAWN_RADIUS='7.5',
        STAGE2_SCENE_MODE='mixed', STAGE2_REFERENCE_ROOT=str(Path(spec['reference']).parent))
    if args.capture_reference:
        os.environ['REFERENCE_ONLY'] = '1'
    sys.path[:0] = [str(TEACHER / 'tokenhsi'), str(TEACHER / 'eval')]
    import isaacgym
    from isaacgym import gymapi
    import torch
    from env.tasks.base_task import BaseTask
    from learning.multi_agent import formal_eval
    from utils import output_paths
    from formal_viewer_display import install

    resolve = output_paths.resolve_output_path

    def local_output(*values, **kwargs):
        kwargs['root'] = ROOT
        return str((ROOT / resolve(*values, **kwargs)).resolve())

    output_paths.resolve_output_path = local_output
    if not args.capture_reference:
        hold = install(torch, gymapi, BaseTask, formal_eval, 0)
    os.chdir(TEACHER)
    sys.argv = [str(TEACHER / 'eval/group_spacing_eval.py'), *spec['command']]
    print('[formal VNC] output=' + str(destination), flush=True)
    print('[formal VNC] ' + spec['controls'], flush=True)
    runpy.run_path(sys.argv[0], run_name='__main__')
    if not args.capture_reference:
        hold()


if __name__ == '__main__':
    main()

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from run_stage2_scalability import ROOT, verify
from table2_sampling import SETTINGS
from table2_metrics import replay


FULL = ROOT/'output/stage2_distill_900_v2/Stage2Distill900V2_08-19-13-51/nn/Stage2Distill900V2_00002000.pth'
STAGE1 = ROOT/'pth/Stage1JointPushDensity3Cone90Finetune_00026000.pth'


def run(args):
    from stage2_xy100 import install
    install()
    output = args.output.resolve()
    if not str(output).startswith(str(ROOT/'output_etc/check')+'/'):
        raise ValueError('Table2 output must be under output_etc/check')
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = (STAGE1 if args.model == 'stage1' else FULL).resolve()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for setting in args.settings:
        people = SETTINGS[setting]
        folder = output/args.model/setting
        reference = output/'reference'/setting
        folder.mkdir(parents=True, exist_ok=True)
        manifest = dict(model=args.model, checkpoint=str(checkpoint), checkpoint_sha256=digest,
            carrier_checkpoint=str(FULL.resolve()), setting=setting, agents=people, objects=2*people,
            seeds=args.seeds, gpu=args.gpu, envs=args.envs, horizon=900, reference=str(reference),
            xy_extent_percent=100, contact_threshold_cm=6, completed=False)
        (folder/'manifest.json').write_text(json.dumps(manifest, indent=2))
        checks = []
        for seed in args.seeds:
            name = 'general_seed'+str(seed)
            command = [sys.executable, str(ROOT/'eval/table2_eval.py'), '--task', 'HumanoidMACoupledPush',
                '--cfg_train', 'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v2.yaml', '--cfg_env', '',
                '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml', '--checkpoint', str(FULL),
                '--seed', str(seed), '--num_envs', str(args.envs), '--num_agents', str(people),
                '--num_objects', str(2*people), '--test', '--eval', '--headless', '--no_video',
                '--eval_protocol', 'general', '--eval_repeats', '1', '--task_graph', 'random_scenario',
                '--enable_size_mode', 'per_action_intersection', '--coupled_assignment', 'ordered']
            for initial_only, destination in ((True, reference/name), (False, folder/name)):
                marker = 'reference_info.json' if initial_only else 'table2.json'
                if (destination/marker).exists():
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                environment = dict(os.environ, TOKENHSI_GPU=str(args.gpu), TABLE2_SETTING=setting,
                    REFERENCE_ONLY=str(int(initial_only)), STAGE2_REFERENCE_ROOT=str(reference), PYTHONHASHSEED=str(seed))
                if args.model == 'stage1' and not initial_only:
                    environment['STAGE1_POLICY_CHECKPOINT'] = str(checkpoint)
                else:
                    environment.pop('STAGE1_POLICY_CHECKPOINT', None)
                log = folder/(name+('_reference.log' if initial_only else '.log'))
                print(f'GPU{args.gpu} {args.model} {setting} seed{seed} reference={initial_only}', flush=True)
                with log.open('w') as stream:
                    subprocess.run(['bash', '-c', 'source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"',
                        '_', *command, '--output_path', str(destination)], cwd=ROOT, env=environment,
                        stdout=stream, stderr=subprocess.STDOUT, check=True)
            path = folder/name
            checks.append(verify(path, reference/name, people, args.envs))
            recomputed = replay(path)
            (path/'table2_recomputed.json').write_text(json.dumps(recomputed, indent=2))
            result = json.loads((path/'result.json').read_text())
            expected = dict(actor=901 if args.model == 'stage1' else 900, critic=900)
            if result['encoder_attention_checks'] != expected:
                raise ValueError('Unexpected encoder call counts')
            if args.model == 'stage1':
                proof = result['stage1_policy']
                if proof['coordination_calls'] or not all(proof[key] for key in ('native_actor_exact','weights_unchanged','normalization_exact')):
                    raise ValueError('Stage1 native policy check failed')
            checks[-1]['cpu_gpu_table2_exact'] = True
        (folder/'verification.json').write_text(json.dumps(checks, indent=2))
        manifest['completed'] = True
        (folder/'manifest.json').write_text(json.dumps(manifest, indent=2))
        print('Completed '+args.model+' '+setting, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=('stage1','stage2'), required=True)
    parser.add_argument('--gpu', type=int, choices=(6,7), required=True)
    parser.add_argument('--settings', nargs='+', choices=SETTINGS, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1,3,6])
    parser.add_argument('--envs', type=int, default=512)
    run(parser.parse_args())

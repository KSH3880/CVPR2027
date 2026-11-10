import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tokenhsi'))
import torch
from evaluate import summarize
from rescore import rescore
import stage2_xy100


def verify(path, reference, agents, envs):
    data = json.loads((path/'result.json').read_text())
    if (data['num_agents'], data['num_objects'], data['num_envs'], data['horizon'],
            data['simulated_steps'], data['xy_extent_percent']) != (agents, 2*agents, envs, 900, 900, 100):
        raise ValueError('Unexpected configuration')
    before = torch.load(reference/'initial.pt', map_location='cpu')
    after = torch.load(path/'initial.pt', map_location='cpu')
    for key in before:
        if key != 'boxes' and not torch.equal(before[key], after[key]):
            raise ValueError('Initial task state mismatch: '+key)
    parked = torch.load(path/'physics_only_reset.pt', map_location='cpu')
    mask = parked['mask']
    if not torch.equal(before['boxes'], parked['observed']):
        raise ValueError('Observed box state mismatch')
    if not torch.equal(before['boxes'][~mask], after['boxes'][~mask]):
        raise ValueError('Required physical box state mismatch')
    observation = torch.load(reference/'observed_scene.pt', map_location='cpu')
    if not torch.equal(observation, parked['observed_scene']):
        raise ValueError('Initial full observation mismatch')
    physics = json.loads((path/'physics_only_verification.json').read_text())
    conditions = json.loads((path/'new_conditions.json').read_text())
    if physics['checked_frames'] != 901 or not conditions['conditions_verified'] or conditions['near50']:
        raise ValueError('Runtime condition check failed')
    score = rescore(path)
    if not all(score[key] for key in ('finite', 'cpu_gpu_exact', 'json_counts_exact', 'errors_match_recorded')):
        raise ValueError('CPU/GPU score validation failed')
    return dict(seed=data['seed'], initial_states_exact=True, initial_observation_exact=True,
                physical_removal_checks=901, cpu_gpu_sr_exact=True, cpu_gpu_errors_match=True)


def run(args):
    stage2_xy100.install()
    torch.set_num_threads(1)
    output, checkpoint = args.output.resolve(), args.checkpoint.resolve()
    if not str(output).startswith(str(ROOT/'output_etc/check')+'/'):
        raise ValueError('Output must be under output_etc/check')
    output.mkdir(parents=True, exist_ok=True)
    policy_checkpoint = args.stage1_checkpoint.resolve() if args.stage1_checkpoint else checkpoint
    digest = hashlib.sha256(policy_checkpoint.read_bytes()).hexdigest()
    for job in args.jobs:
        mode, people = job.split(':')
        people = int(people)
        if mode not in ('independent', 'mixed') or people not in (2, 4, 8, 16):
            raise ValueError('Unknown job: '+job)
        folder = output/(mode+'_'+str(people)+'h')
        reference = output/(mode+'_'+str(people)+'h_reference'+('_unique' if people > 4 else ''))
        folder.mkdir(parents=True, exist_ok=True)
        if args.reference_evaluations:
            source_root = args.reference_evaluations.resolve()
            source_manifest = source_root/folder.name/'manifest.json'
            reference.mkdir(parents=True, exist_ok=True)
            for seed in args.seeds:
                if source_manifest.exists():
                    saved = json.loads(source_manifest.read_text())
                else:
                    sources = json.loads((source_root/'manifest.json').read_text())['mixed4_source_runs']
                    saved = json.loads((Path(sources[str(seed)]).parent/'manifest.json').read_text())
                origin = Path(saved['reference'])
                origin = origin if origin.is_absolute() else ROOT/origin
                target = reference/('general_seed'+str(seed))
                if not target.exists():
                    target.symlink_to((origin/target.name).resolve(), target_is_directory=True)
        manifest = dict(checkpoint=str(policy_checkpoint), checkpoint_sha256=digest, gpu=args.gpu,
            agents=people, objects=2*people, envs=args.envs, horizon=900, seeds=args.seeds,
            scene_mode=mode, xy_extent_percent=100, reference=str(reference), completed=False,
            scene_carrier_checkpoint=str(checkpoint), distractor_attention='all_observed_tokens')
        (folder/'manifest.json').write_text(json.dumps(manifest, indent=2))
        checks = []
        for seed in args.seeds:
            name = 'general_seed'+str(seed)
            entry = 'stage1_protocol_eval.py' if args.stage1_checkpoint else 'stage2_scalability_eval.py'
            command = [sys.executable, str(ROOT/'eval'/entry),
                '--task', 'HumanoidMACoupledPush', '--cfg_train', 'tokenhsi/data/cfg/train/rlg/stage2_distill_900_v2.yaml',
                '--cfg_env', '', '--motion_file', 'tokenhsi/data/dataset_stage1_jointpush.yaml',
                '--checkpoint', str(checkpoint), '--seed', str(seed), '--num_envs', str(args.envs),
                '--num_agents', str(people), '--num_objects', str(2*people), '--test', '--eval',
                '--headless', '--no_video', '--eval_protocol', 'general', '--eval_repeats', '1',
                '--task_graph', 'random_scenario', '--enable_size_mode', 'per_action_intersection',
                '--coupled_assignment', 'ordered']
            if people > 4:
                command.append('--group_spawn_over4')
            for initial_only, destination in ((True, reference/name), (False, folder/name)):
                marker = 'reference_info.json' if initial_only else 'result.json'
                if (destination/marker).exists():
                    continue
                environment = dict(os.environ, TOKENHSI_GPU=str(args.gpu), STAGE2_SCENE_MODE=mode,
                    REFERENCE_ONLY=str(int(initial_only)), STAGE2_REFERENCE_ROOT=str(reference),
                    PYTHONHASHSEED=str(seed))
                if args.stage1_checkpoint:
                    environment['STAGE1_POLICY_CHECKPOINT'] = str(policy_checkpoint)
                log = folder/(name+('_reference.log' if initial_only else '.log'))
                print(f'GPU{args.gpu} {job} seed{seed} reference={initial_only}', flush=True)
                with log.open('w') as stream:
                    subprocess.run(['bash', '-c', 'source tokenhsi/scripts/multi_agent/runtime_env.sh\nexec "$@"',
                        '_', *command, '--output_path', str(destination)], cwd=ROOT, env=environment,
                        stdout=stream, stderr=subprocess.STDOUT, check=True)
            checks.append(verify(folder/name, reference/name, people, args.envs))
            data = json.loads((folder/name/'result.json').read_text())
            if data.get('distractor_attention') != 'all_observed_tokens':
                raise ValueError('Distractor attention policy differs')
            if args.stage1_checkpoint:
                policy = data['stage1_policy']
                if not all(policy[key] for key in ('native_actor_exact', 'normalization_exact', 'weights_unchanged')) or policy['coordination_calls']:
                    raise ValueError('Stage1 native policy verification failed')
        summary = summarize(folder)
        (folder/'verification.json').write_text(json.dumps(checks, indent=2))
        manifest['completed'] = True
        (folder/'manifest.json').write_text(json.dumps(manifest, indent=2))
        print(f'Completed {job}: '+json.dumps(summary['interaction_group_scores']['region']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(6, 7), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--jobs', nargs='+', required=True)
    parser.add_argument('--envs', type=int, default=512)
    parser.add_argument('--stage1-checkpoint', type=Path)
    parser.add_argument('--reference-evaluations', type=Path)
    run(parser.parse_args())

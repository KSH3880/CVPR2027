"""Prepare/run native Chair HOLDING_AT 512x3 evaluation in an isolated runs folder."""
import argparse, csv, datetime, json, os, subprocess, sys
from pathlib import Path
from metrics import PROTOCOL, CARRY_PROTOCOL, aggregate, aggregate_tokenhsi, sha256, validate
from cli_helpers import format_summary

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--template-checkpoint', type=Path, help='78k zero-shot: native FT checkpoint provides env/sim/train only')
    p.add_argument('--object-root', type=Path, default=ROOT / 'tokenhsi/data/dataset_dynamic_objects')
    p.add_argument('--python', default='/home/hwanhee/anaconda3/envs/tokenhsi_jhh/bin/python')
    p.add_argument('--gpu', type=int, default=5)
    p.add_argument('--public-chair-checkpoint', type=Path, help='Original TokenHSI Chair adaptation weights; --checkpoint constructs the scene only')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--concise', action='store_true', help='Print seed rates and mean/std; keep full traces and JSON in the output directory')
    p.add_argument('--smoke', action='store_true', help='4 environments, seed0, full900step; not performance evaluation')
    a = p.parse_args()
    if not a.checkpoint.is_file():
        p.error('Checkpoint must exist')
    count, seeds = (4, [0]) if a.smoke else (512, [0, 1, 2])
    output = (a.output or ROOT / 'output_etc/check/hold_at_eval' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    if not any(parent in output.parents for parent in (PROJECT / 'runs', ROOT / 'output_etc/check')):
        p.error('Output must be a new directory under stage1_combined/output_etc/check/ or project runs/')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    command = [a.python, str(Path(__file__).with_name('prepare.py')),
        '--checkpoint', str(a.checkpoint.resolve()), '--output', str(output),
        '--object-root', str(a.object_root.resolve()), '--num-envs', str(count)]
    if a.template_checkpoint:
        command += ['--template-checkpoint', str(a.template_checkpoint.resolve())]
    if a.public_chair_checkpoint:
        command += ['--public-chair-checkpoint', str(a.public_chair_checkpoint.resolve())]
    subprocess.run(command, env=env, check=True)
    prep = json.loads((output / 'preparation.json').read_text())
    source = Path(prep['source'])
    manifest = dict(protocol=PROTOCOL, secondary_protocol=CARRY_PROTOCOL,
                    secondary_is_same_rollout=True, secondary_is_native_environment=False,
                    checkpoint=str(output / 'checkpoint.pth'),
                    checkpoint_sha256=prep['checkpoint_sha256'], num_envs=count,
                    seeds=seeds, gpu=a.gpu, smoke=a.smoke, prepared_only=a.prepare_only,
                    config_source=prep['config_source'], config_only_template=prep['config_only_template'],
                    checkpoint_byte_copy_unchanged=prep['checkpoint_byte_copy_unchanged'])
    if a.public_chair_checkpoint:
        manifest['public_policy'] = json.loads((output / 'public_policy_provenance.json').read_text())
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    if a.prepare_only:
        print('PREPARED', output)
        return
    records = []
    for seed in seeds:
        child = os.environ.copy()
        for key in ('CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY', 'CUDA_VISIBLE_DEVICES'):
            child.pop(key, None)
        child.update(TOKENHSI_GPU=str(a.gpu), PYTHONDONTWRITEBYTECODE='1', TOKENHSI_EVAL_PYTHON=a.python)
        print('START', seed, output, flush=True)
        with (output / f'seed{seed}.log').open('w') as log:
            rc = subprocess.run(['bash', str(source / 'evaluation/hold_at/seed.sh'), str(seed), str(count)],
                                env=child, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc:
            (output / 'failed.json').write_text(json.dumps(dict(seed=seed, returncode=rc), indent=2))
            raise SystemExit(f'Seed{seed} failed with {rc}; inspect {output}')
        folder = source / f'output_etc/check/hold_at_seed{seed}'
        validate(folder)
        record = json.loads((folder / 'results.json').read_text())
        record['run'] = str(folder)
        records.append(record)
        (output / 'records.json').write_text(json.dumps(records, indent=2))
        if a.concise:
            line = 'seed{}: ours {}/{} = {:.2f}%'.format(seed, record['point_ever_count'],
                  count, 100 * record['point_ever_count'] / count)
            if 'tokenhsi_criteria' in record:
                secondary = record['tokenhsi_criteria']
                line += ' | TokenHSI criteria {}/{} = {:.2f}%'.format(
                    secondary['success_count'], count, 100 * secondary['success_rate'])
            print(line, flush=True)
    summary = dict(protocol=PROTOCOL, records=records, smoke=a.smoke)
    if not a.smoke:
        summary['statistics'] = aggregate(records)
        summary['tokenhsi_criteria_statistics'] = aggregate_tokenhsi(records)
    with (output / 'episodes.csv').open('w', newline='') as stream:
        writer = None
        for record in records:
            with (Path(record['run']) / 'episodes.csv').open(newline='') as source_csv:
                reader = csv.DictReader(source_csv)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=reader.fieldnames)
                    writer.writeheader()
                writer.writerows(reader)
    for name, digest in json.loads((output / 'asset_sha256.json').read_text()).items():
        assert sha256(output / name) == digest, name
    lines = ['# Native Chair HOLDING+AT evaluation', '', 'Protocol: ' + PROTOCOL, '',
             'AT point and region are identical: object center 3D distance <=0.20m, first-episode ever.']
    if a.smoke:
        lines.append('Smoke validation only: 4 environments x seed0; not a performance result.')
    else:
        stats = summary['statistics']
        lines.append('512 environments x independent seeds0/1/2, max30s; mean +/- sample std: '
                     + '{:.2f} +/- {:.2f}%'.format(stats['mean_percent'], stats['sample_std_percentage_points']))
        secondary = summary['tokenhsi_criteria_statistics']
        lines.append('TokenHSI carry criteria on the SAME episodes (first599actions,20cm+up-dot>=0.86615): '
                     + '{:.2f} +/- {:.2f}%'.format(secondary['mean_percent'], secondary['sample_std_percentage_points']))
    lines += ['This secondary metric is a re-score of the common scene, not an original TokenHSI environment/continuous-repeat evaluation.',
              'Per-environment records for both metrics: episodes.csv.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'completed.json').write_text(json.dumps(dict(rc=0, all_rescored=True,
        smoke=a.smoke, sha256={name:sha256(output / name) for name in ('run_manifest.json', 'summary.json', 'records.json', 'asset_sha256.json', 'episodes.csv')}), indent=2))
    if a.concise:
        print(format_summary(summary, output), flush=True)
    else:
        print('RESULT', json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()

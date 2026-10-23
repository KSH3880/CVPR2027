#!/usr/bin/env python3
"""Native in-training-range CLIMB vs boxes matched on bbox and mass."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CASES = ('climb_box', 'climb_cabinet', 'climb_table')


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--envs', type=int, default=64)
    parser.add_argument('--episode-length', type=int, default=600)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT not in output.parents or output.exists():
        parser.error('Use a new directory inside this checkout')
    output.mkdir(parents=True)
    report = dict(physical_gpu=os.environ.get('TOKENHSI_GPU'), results={}, paired_checks={},
                  limitations=['Native subset, no scaling; limited object identities.',
                               'BBox and mass matched, collider-derived inertia and COM may differ.',
                               'Single seed; size intervals are not exact training grid points.'])
    for geometry in ('native', 'matched-box'):
        cmd = [sys.executable, str(HERE / 'evaluate_object_zero_shot.py'),
               '--output', str(output / geometry), '--cases', *CASES,
               '--climb-ascent', '--climb-training-range', '--object-geometry', geometry,
               '--envs', str(args.envs), '--episode-length', str(args.episode_length),
               '--seed', str(args.seed), '--assets-per-category', '0', '--headless']
        print('Evaluating in-range:', geometry, flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        report['results'][geometry] = read(output / geometry / 'summary.json')
        (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    for case in CASES:
        a, b = [output / g / case for g in ('native', 'matched-box')]
        report['paired_checks'][case + '/initial_state'] = read(a / 'initial_state.json') == read(b / 'initial_state.json')
        av, bv = [read(p / 'runtime_audit.json') for p in (a, b)]
        pa, pb = av['physical_sources_by_asset'], bv['physical_sources_by_asset']
        report['paired_checks'][case + '/bbox_mass_friction'] = (
            av['bbox'] == bv['bbox'] and pa.keys() == pb.keys() and all(
                abs(pa[k]['mass'] - pb[k]['mass']) < 1e-4 and pa[k]['friction'] == pb[k]['friction']
                for k in pa))
    report['paired_checks']['source_and_checkpoint_unchanged'] = all(
        all(read(output / g / 'integrity.json').values()) for g in ('native', 'matched-box'))
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if not all(report['paired_checks'].values()):
        raise RuntimeError('Paired comparison checks failed; inspect comparison.json')


if __name__ == '__main__':
    main()

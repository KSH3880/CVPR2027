#!/usr/bin/env python3
"""Paired inference-only SIT/CLIMB seat geometry ablation for KSH39000."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--envs', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--assets-per-category', type=int, default=0)
    parser.add_argument('--episode-length', type=int, default=600)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT not in output.parents or output.exists():
        parser.error('Use a new output directory inside this checkout')
    output.mkdir(parents=True)
    modes = ('full', 'seat-height', 'seat-inset')
    cases = ('sit_chair', 'climb_chair')
    report = dict(protocol=vars(args).copy(), results={}, paired_checks={})
    report['protocol']['output'] = str(output)
    report['protocol']['physical_gpu'] = os.environ.get('TOKENHSI_GPU')
    for mode in modes:
        cmd = [sys.executable, str(HERE / 'evaluate_object_zero_shot.py'),
               '--output', str(output / mode), '--cases', *cases,
               '--policy-bbox', mode, '--climb-surface', 'seat', '--headless',
               '--envs', str(args.envs), '--repeats', str(args.repeats),
               '--assets-per-category', str(args.assets_per_category),
               '--episode-length', str(args.episode_length), '--seed', str(args.seed)]
        print('Running paired seat ablation:', mode, flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        report['results'][mode] = {}
        for case in cases:
            trials = json.loads((output / mode / case / 'trials.json').read_text())
            def stats(rows):
                return dict(num_agent_trials=len(rows), **{
                    key: sum(r[key] for r in rows) / len(rows)
                    for key in ('final_success', 'ever_success', 'terminated',
                                'foot_above_seat_15cm_fraction')})
            report['results'][mode][case] = dict(overall=stats(trials), by_asset={
                asset: stats([r for r in trials if r['asset'] == asset])
                for asset in sorted({r['asset'] for r in trials})})
        (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    for case in cases:
        for filename in ('initial_state.json', 'runtime_audit.json'):
            values = [json.loads((output / mode / case / filename).read_text()) for mode in modes]
            report['paired_checks'][case + '/' + filename] = all(v == values[0] for v in values[1:])
    report['paired_checks']['upstream_and_checkpoint_unchanged'] = all(
        all(json.loads((output / mode / 'integrity.json').read_text()).values()) for mode in modes)
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    if not all(report['paired_checks'].values()):
        raise RuntimeError('Paired geometry/initialization checks failed; inspect comparison.json')
    print(json.dumps({mode: {case: row['overall'] for case, row in cases_.items()}
                      for mode, cases_ in report['results'].items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()

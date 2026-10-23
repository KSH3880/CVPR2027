"""Run retention comparison and PUSH suites in separate frozen subprocesses."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from metrics import aggregate

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PORT = REPO/'ports/jhh_stage1_push_raw_cuboid'
STAGE1 = PORT/'output/stage1/Stage1V2Fixed_00073000.pth'
PUSH = PORT/'output/stage1_v2_fixed_push_360_raw_cuboid/Stage1V2FixedPush360RawCuboid_21-16-00-23/nn/Stage1V2FixedPush360RawCuboid_00022500.pth'


def compare_initial(a, b):
    with np.load(a, allow_pickle=False) as first, np.load(b, allow_pickle=False) as second:
        for key in ('root', 'dof', 'boxes', 'sizes', 'goal', 'packet', 'origins', 'mass'):
            if not np.array_equal(first[key], second[key]):
                raise RuntimeError('Policy initial scenes differ: '+key+'; '+str(a)+'; '+str(b))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage1-checkpoint', type=Path, default=STAGE1)
    p.add_argument('--push-checkpoint', type=Path, default=PUSH)
    p.add_argument('--suite', choices=('retention', 'push', 'all'), default='all')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        p.error('Output exists; choose a new directory')
    if PORT/'output_etc/check' not in args.output.parents:
        p.error('Output must be below '+str(PORT/'output_etc/check'))
    if len(set(args.seeds)) != len(args.seeds):
        p.error('Duplicate seeds')
    args.output.mkdir(parents=True)
    plan = []
    if args.suite in ('retention', 'all'):
        for label, checkpoint in (('stage1_73000', args.stage1_checkpoint), ('push_22500', args.push_checkpoint)):
            for task in ('at', 'ontop', 'sit', 'climb'):
                for seed in args.seeds:
                    plan.append((label, checkpoint.resolve(), task, 'loco', 2., 5., seed))
    if args.suite in ('push', 'all'):
        for start in ('loco', 'rsi'):
            for lo, hi in ((2., 5.), (5., 10.)):
                for seed in args.seeds:
                    plan.append(('push_22500', args.push_checkpoint.resolve(), 'push', start, lo, hi, seed))
    records, runs, initials = [], [], {}
    for label, checkpoint, task, start, lo, hi, seed in plan:
        name = f'{label}_{task}_{start}_{lo:g}to{hi:g}_seed{seed}'
        dest = args.output/name
        command = ['bash', str(HERE/'run.sh'), '--checkpoint', str(checkpoint), '--policy', label,
                   '--task', task, '--start', start, '--distance-min', str(lo), '--distance-max', str(hi),
                   '--seed', str(seed), '--output', str(dest)]
        if args.smoke:
            command += ['--smoke', '--num-envs', '4', '--steps', '12']
        print('Evaluating', name, flush=True)
        with (args.output/(name+'.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        runs.append(dict(command=command, exit_code=result.returncode, directory=str(dest)))
        (args.output/'run_manifest.json').write_text(json.dumps(runs, indent=2)+'\n')
        if result.returncode:
            raise SystemExit('Evaluation failed; see '+str(args.output/(name+'.log')))
        records.append(json.loads((dest/'results.json').read_text()))
        if task != 'push':
            key = (task, seed)
            if key in initials:
                compare_initial(initials[key], dest/'initial_0.npz')
            else:
                initials[key] = dest/'initial_0.npz'
    summary = aggregate(records)
    # Four-task average: average each seed first, then sample std of those averages.
    for label in ('stage1_73000', 'push_22500'):
        selected = [r for r in records if r['policy'] == label and r['task'] != 'push']
        if not selected:
            continue
        row = dict(policy=label, task='four_task_average', start='loco', distance_range=[], seeds=args.seeds)
        for metric in ('point', 'region'):
            values = [np.mean([r['scores'][metric]['rate_percent'] for r in selected if r['seed'] == seed]) for seed in args.seeds]
            row[metric] = dict(mean_percent=float(np.mean(values)), sample_std_percent=float(np.std(values, ddof=1)) if len(values)>1 else None)
        summary.append(row)
    (args.output/'summary.json').write_text(json.dumps(dict(smoke=args.smoke, records=records, summary=summary,
        initial_scene_comparison='exact, including DOF' if initials and len(records)>len(initials) else 'not applicable'), indent=2)+'\n')
    lines = ['# Single-agent evaluation'+(' (SMOKE: not a benchmark)' if args.smoke else ''), '',
             '| Policy | Task | Start | Distance | Point % | Region % |', '| --- | --- | --- | --- | --- | --- |']
    def cell(value):
        sd = value['sample_std_percent']
        return f"{value['mean_percent']:.2f}"+(f' ± {sd:.2f}' if sd is not None else ' (one seed)')
    for row in summary:
        lines.append(f"| {row['policy']} | {row['task']} | {row['start']} | {row['distance_range']} | {cell(row['point'])} | {cell(row['region'])} |")
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Saved', args.output/'REPORT.md')


if __name__ == '__main__':
    main()

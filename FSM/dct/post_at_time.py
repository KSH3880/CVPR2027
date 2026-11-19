import argparse
import csv
from pathlib import Path

import numpy as np

from common import ROOT, read, write, sha
from suite import LABELS


def complete_mean(values):
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    suite, out = args.suite.resolve(), args.output.resolve()
    if not any(root in out.parents for root in (ROOT/'output_etc/check', ROOT/'runs/table89_delay')):
        raise ValueError('Output must be under runs/table89_delay or output_etc/check')
    summary = read(suite/'summary.json')
    if not summary['completed']:
        raise ValueError('Suite is not complete')
    out.mkdir(parents=True, exist_ok=False)
    files = list((suite/'analysis').glob('*/*.json'))
    files += [Path(x['analysis']) for x in read(suite/'manifest.json').get('baselines', [])]
    index = {}
    fields = ('setting', 'model', 'seed', 'delay')
    for p in files:
        row = read(p)
        index[tuple(row[k] for k in fields)] = p
    hashes = {str(suite/'summary.json'): sha(suite/'summary.json'),
              str(Path(__file__).resolve()): sha(Path(__file__))}
    runs, trials = [], []
    for row in summary['runs']:
        if not row['setting'].startswith('enable'):
            continue
        key = tuple(row[k] for k in fields)
        p = index[key]
        source = Path(row['source'])
        for f in (p, p.with_suffix('.npz'), source/'table2_trials.npz', source/'table2.json'):
            hashes[str(f)] = sha(f)
        for name in ('table2_trials.npz', 'table2.json'):
            f = source/name
            if hashes[str(f)] != row['source_sha256'][str(f)]:
                raise ValueError('Verified source changed')
        with np.load(p.with_suffix('.npz')) as z:
            at, group = z['first_at'], z['first_group']
        with np.load(source/'table2_trials.npz') as z:
            original = z['first_success']
        expected = np.where((original > 0) & (original <= row['analyzed_steps']), original, -1)
        np.testing.assert_array_equal(group, expected)
        valid = group > 0
        if valid.sum() != row['successes'] or (at > 0).sum() != row['at_successes']:
            raise ValueError('Success population mismatch')
        if not ((at[valid] > 0) & (at[valid] <= group[valid])).all():
            raise ValueError('Invalid AT/group ordering')
        dt = read(source/'table2.json')['control_dt']
        if abs(dt-1/30) > 1e-8:
            raise ValueError('Unexpected control timestep')
        values = (group[valid]-at[valid])*dt
        np.testing.assert_allclose(values, group[valid]*dt-at[valid]*dt, atol=1e-12)
        result = {k: row[k] for k in (*fields, 'trials', 'successes', 'group_sr_percent')}
        result.update(post_at_s=float(values.mean()) if len(values) else None,
                      median_s=float(np.median(values)) if len(values) else None,
                      sum_seconds=float(values.sum()))
        runs.append(result)
        for env in range(row['trials']):
            trials.append(dict(zip(fields, key), env_id=env, at_step=int(at[env]), group_step=int(group[env]),
                included=bool(valid[env]), post_at_s=float((group[env]-at[env])*dt) if valid[env] else None))
    seeds = sorted({r['seed'] for r in runs})
    delays = sorted({r['delay'] for r in runs}, key=str)
    settings = sorted({r['setting'] for r in runs})
    macro = []
    for model in LABELS:
        for seed in seeds:
            for delay in delays:
                selected = [r for r in runs if r['model'] == model and r['seed'] == seed and r['delay'] == delay]
                if len(selected) != 3:
                    raise ValueError('Incomplete type coverage')
                count = sum(r['successes'] for r in selected)
                macro.append(dict(setting='enable_average', model=model, seed=seed, delay=delay,
                    post_at_s=complete_mean([r['post_at_s'] for r in selected]), successes=count,
                    trials=sum(r['trials'] for r in selected),
                    pooled_post_at_s=sum(r['sum_seconds'] for r in selected)/count if count else None))
    stats, lines = [], ['# Post-AT Completion Time: Group success only', '',
        '(first simultaneous Group success step − first AT success step) / 30, in seconds.',
        'AT-only and failed trials are excluded; no failure-time substitution. Zero successful trials = N/A.',
        'Within each seed: successful-episode mean per type → equal-weight mean of three types. Then seed mean ± sample std.',
        'If a type has zero successes, its equal-weight family mean is N/A; pooled successful-trial means are also saved.',
        'Condition order: '+' → '.join(map(str, delays))+f'; seeds={seeds}.', '']
    for setting in [*settings, 'enable_average']:
        lines += ['## '+setting, '', '| Model | Post-AT (s) | Group success/total |', '|---|---:|---:|']
        for model, label in LABELS.items():
            cells, counts = [], []
            for delay in delays:
                rows = [r for r in runs+macro if r['setting'] == setting and r['model'] == model and r['delay'] == delay]
                values = [r['post_at_s'] for r in rows]
                value = complete_mean(values)
                std = float(np.std(values, ddof=1)) if value is not None and len(values) > 1 else None
                stats.append(dict(setting=setting, model=model, delay=delay, mean_s=value, std_s=std,
                    successes=sum(r['successes'] for r in rows), trials=sum(r['trials'] for r in rows)))
                cells.append('N/A' if value is None else f'{value:.3f}'+(f' ± {std:.3f}' if std is not None else ''))
                counts.append(f"{sum(r['successes'] for r in rows)}/{sum(r['trials'] for r in rows)}")
            lines.append('| '+label+' | '+' → '.join(cells)+' | '+' → '.join(counts)+' |')
        lines.append('')
    write(out/'summary.json', dict(source_suite=str(suite), runs=runs, macro_per_seed=macro, statistics=stats))
    with (out/'trials.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    if any(sha(Path(p)) != h for p,h in hashes.items()):
        raise ValueError('Input changed during analysis')
    write(out/'verification.json', dict(passed=True, runs=len(runs), trial_rows=len(trials),
        group_successes=sum(r['successes'] for r in runs), success_only=True,
        online_first_success_exact=True, at_ordering_valid=True, control_dt_verified=True, source_sha256=hashes))


if __name__ == '__main__':
    main()

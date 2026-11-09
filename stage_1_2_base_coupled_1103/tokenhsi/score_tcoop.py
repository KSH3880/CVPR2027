"""Reaggregate t_coop without Isaac Gym, torch or GPU."""
import argparse
import json
from pathlib import Path
import numpy as np
from utils.cooperative_push_metric import sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', nargs='+', type=Path, help='tcoop_trace.npz or formal-eval seed folder')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--start-mode', choices=['both','displacement','speed'], default='both')
    parser.add_argument('--contact', nargs='+', type=float, default=[.06,.08,.12])
    parser.add_argument('--speeds', nargs='+', type=float, default=[i/10 for i in range(2,11)])
    parser.add_argument('--sustain-seconds', type=float, default=1.)
    parser.add_argument('--displacement', type=float, default=.2)
    args = parser.parse_args()
    modes = ('displacement','speed') if args.start_mode == 'both' else (args.start_mode,)
    sources = []
    for path in args.traces:
        path = path/'tcoop_trace.npz' if path.is_dir() else path
        with np.load(path, allow_pickle=False) as archive:
            results = sweep(dict(archive), args.contact, args.speeds, modes,
                            args.sustain_seconds, args.displacement)
        sources.append(dict(path=str(path.resolve()), results=results))
    # Equal trial weights include no-start failures as 0%; do not pool frames.
    combined = []
    for i, result in enumerate(sources[0]['results']):
        groups = {}
        for key in ('2','3','4','all'):
            rows = [s['results'][i]['summary'][key] for s in sources]
            count = sum(r['trials'] for r in rows)
            rates = [r['t_coop_mean_percent'] for r in rows if r['trials']]
            groups[key] = dict(trials=count, start_failures=sum(r['start_failures'] for r in rows),
                t_coop_mean_percent=sum((r['t_coop_mean_percent'] or 0)*r['trials'] for r in rows)/count if count else None,
                source_mean_min=min(rates) if rates else None, source_mean_max=max(rates) if rates else None)
        combined.append(dict(config=result['config'], summary=groups))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(aggregation='equal trial mean including no-start zeros',
                                         combined=combined, sources=sources), indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()

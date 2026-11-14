import argparse
from pathlib import Path

from utils.before_stop_evaluation import write_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]/'output_etc/check'
    if root.resolve() not in args.output.resolve().parents:
        parser.error('Output must be below output_etc/check/')
    report = write_report(args.folders, args.output)
    if report is None:
        parser.error('No waiting telemetry; no simulation was started')
    print(f"Stopping diagnostics: {len(report['seeds'])} seeds -> {args.output/'stops.json'}")

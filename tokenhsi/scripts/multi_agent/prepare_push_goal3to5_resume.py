"""Prepare a separate bank and full-state checkpoint for PUSH goals 3--5m."""
from isaacgym import gymapi  # before torch
import argparse
import hashlib
import json
from pathlib import Path
import torch
from utils.push_goal_resume import environments, project_bank, migrate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, nargs='?')
    parser.add_argument('output', type=Path, nargs='?')
    parser.add_argument('--bank-only', action='store_true')
    args = parser.parse_args()
    if not args.bank_only and (args.checkpoint is None or args.output is None):
        parser.error('checkpoint and new output required')
    if not args.bank_only and args.output.exists():
        raise ValueError('Refusing to overwrite an existing checkpoint')
    source, target = environments()
    bank = project_bank(source, target)
    path = Path(target['openPushRSI']['bank_file'])
    if path.exists():
        if json.loads(path.read_text()) != bank:
            raise ValueError('Existing bank differs; preserve it')
    else:
        with path.open('x') as f:
            f.write(json.dumps(bank, indent=2) + '\n')
    if args.bank_only:
        print(path, len(bank['candidates']['push']))
        return
    raw = args.checkpoint.read_bytes()
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    result = migrate(weights, source, target)
    report = dict(source=str(args.checkpoint.resolve()), source_sha256=hashlib.sha256(raw).hexdigest(),
        output=str(args.output.resolve()), epoch=int(weights['epoch']), frame=int(weights['frame']),
        goal_distance_range_m=[3., 5.], preserved='model, optimizer, AMP/RMS, epoch/frame and all other learned state',
        changed='goal/workspace/RSI-bank metadata only')
    result['push_goal_range_migration'] = report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f:
        torch.save(result, f)
    args.output.with_suffix('.migration.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

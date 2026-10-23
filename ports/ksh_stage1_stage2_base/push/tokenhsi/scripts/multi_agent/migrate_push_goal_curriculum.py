"""Create a separate full-state resume checkpoint for the PUSH goal curriculum."""
import argparse
from pathlib import Path
import torch
import yaml
from utils.push_goal_curriculum import migrate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cfg_env', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.suffix != '.pth' or target.exists():
        raise ValueError('Output must be a new .pth file')
    cfg = yaml.safe_load(Path(args.cfg_env).read_text())
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    result = migrate_checkpoint(weights, cfg['env'])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        torch.save(result, stream)
    print(f"Created {target}; preserved epoch={result['epoch']}, frame={result.get('frame')}")


if __name__ == '__main__':
    main()

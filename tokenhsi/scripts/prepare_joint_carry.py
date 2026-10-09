import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
MOTION_FILE = 'tokenhsi/data/dataset_loco_sit_carry_climb.yaml'


def motion_paths(root):
    path = root / MOTION_FILE
    groups = yaml.safe_load(path.read_text())['motions']
    return sorted({path.parent / entry[key]
                   for entries in groups.values() for entry in entries
                   for key in ('file', 'obj_file') if entry.get(key)})


def connect_data(root, data_root):
    source = Path(data_root).expanduser().resolve()
    data = root / 'tokenhsi/data'
    branches = sorted({Path(*path.relative_to(data).parts[:2])
                       for path in motion_paths(root)})
    pending = []
    for branch in branches:
        target, destination = (source / branch).resolve(), data / branch
        if not target.is_dir():
            raise ValueError(f'Missing {target}; TOKENHSI_DATA_ROOT must point to tokenhsi/data')
        if destination.resolve() == target:
            continue
        if destination.exists():
            raise ValueError(f'Existing data preserved: {destination}. Unset TOKENHSI_DATA_ROOT or use its current source.')
        pending.append((destination, target))
    for destination, target in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            destination.unlink()
        destination.symlink_to(target, target_is_directory=True)


def check_data(root):
    missing = [str(path.relative_to(root)) for path in motion_paths(root) if not path.is_file()]
    if missing:
        raise ValueError('Missing original TokenHSI data. Set TOKENHSI_DATA_ROOT=/path/to/TokenHSI/tokenhsi/data.\n'
                         + '\n'.join(missing[:5]) + f'\n({len(missing)} missing files)')
    manifest = json.loads((root / 'joint_carry/runtime_manifest.json').read_text())
    for filename, digest in manifest['sha256'].items():
        path = root / filename
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Missing or changed bundled AMP/RSI file: {filename}. Restore it from this checkout.')
    asset = root / 'tokenhsi/data/assets/mjcf/phys_humanoid_v3.xml'
    if not asset.is_file():
        raise ValueError(f'Missing humanoid asset: {asset}')
    return len(motion_paths(root)), len(manifest['sha256'])


def main():
    parser = argparse.ArgumentParser(description='Connect existing TokenHSI data and check bundled joint-carry AMP/RSI.')
    parser.add_argument('--data-root', default=os.environ.get('TOKENHSI_DATA_ROOT'))
    args = parser.parse_args()
    try:
        if args.data_root:
            connect_data(ROOT, args.data_root)
        motions, bundled = check_data(ROOT)
    except (OSError, ValueError) as error:
        parser.exit(1, f'{error}\n')
    print(f'Joint carry ready: {motions} original files, {bundled} bundled AMP/RSI hashes verified.')


if __name__ == '__main__':
    main()

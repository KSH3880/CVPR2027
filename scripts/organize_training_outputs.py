#!/usr/bin/env python3
"""Move local training artifacts into project output/, preserving old paths.

Dry run by default. --apply records every move and active-path alias in a JSON
manifest. Running processes, source snapshots, and verification runs stay put.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def active_references():
    references = []
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            # Shell command bodies are not file arguments. In particular, do not
            # let this tool's invoking shell make every planned path look active.
            if not args or Path(args[0]).name in ('bash', 'sh', 'timeout'):
                continue
            cwd = Path(os.readlink(str(proc / 'cwd')))
            refs = [str(cwd)]
            for arg in args[1:]:
                refs.append(arg)
                if arg and not arg.startswith('-') and not Path(arg).is_absolute():
                    refs.append(str(cwd / arg))
            for fd in (proc / 'fd').iterdir():
                try:
                    refs.append(os.readlink(str(fd)))
                except OSError:
                    pass
            references.append((int(proc.name), refs))
        except (OSError, PermissionError):
            continue
    return references


def plan():
    pairs = []
    for group in ('carry_planner', 'stack_planner', 'task_allocation'):
        base = ROOT / 'runs' / group
        if base.is_dir() and not base.is_symlink():
            for child in sorted(base.iterdir()):
                pairs.append((child, ROOT / 'TokenHSI-coord/output' / group / child.name))
    base = ROOT / 'runs/jointpush'
    ksh = base / 'ksh_jointpush_rsi/output'
    if ksh.is_dir():
        pairs.append((ksh, ROOT / 'ksh_jointpush/output'))
    seen = set()
    for checkpoint in sorted(base.rglob('*.pth')):
        if checkpoint.parent.name != 'nn' or 'output_etc' in checkpoint.parts:
            continue
        source = checkpoint.parent.parent.parent
        if source in seen or ksh == source or ksh in source.parents:
            continue
        seen.add(source)
        relative = source.relative_to(base)
        if 'output' in relative.parts:
            relative = Path(*relative.parts[relative.parts.index('output') + 1:])
        pairs.append((source, ROOT / 'jointpush/output' / relative))
    explicit = [
        ('runs/table_carry/finetune_20261019_115145', 'stage2_jointcarry/output/table_carry/finetune_20261019_115145'),
        ('runs/push_door/split_goal_gpu0_20261015_152651/training', 'stage1_fix/output/push_door/split_goal_gpu0_20261015_152651'),
        ('runs/chair_hold_at_78k_20261026_135444/train', 'stage1_combined/output/chair_hold_at_78k_20261026_135444'),
        ('runs/chair_bbox_ft_20261022/source/output_etc/check/chair_bbox35_20k_20261022_142206', 'stage1_combined/output/chair_bbox35_20k_20261022_142206'),
    ]
    pairs.extend((ROOT / old, ROOT / new) for old, new in explicit if (ROOT / old).exists())
    return pairs


def checkpoints(path):
    return {str(p.relative_to(path)): (p.stat().st_ino, p.stat().st_size)
            for p in path.rglob('*.pth') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'runs/training_output_migration_20261028.json')
    args = parser.parse_args()
    references = active_references()
    records = []
    for source, destination in plan():
        if source.is_symlink() or (destination.is_symlink() and destination.resolve() == source.resolve()):
            continue
        if destination.exists() or destination.is_symlink():
            raise RuntimeError('Destination already exists: ' + str(destination))
        active = [pid for pid, refs in references if any(str(source) in ref for ref in refs)]
        records.append(dict(source=str(source.relative_to(ROOT)), destination=str(destination.relative_to(ROOT)),
                            action='active_alias' if active else 'move', active_pids=active))
    # Preflight the whole plan before making any changes.
    destinations = [r['destination'] for r in records]
    if len(destinations) != len(set(destinations)):
        raise RuntimeError('Duplicate destination')
    for record in records:
        source, destination = ROOT / record['source'], ROOT / record['destination']
        print(record['action'], record['source'], '->', record['destination'], record['active_pids'])
        if not args.apply:
            continue
        # Recheck immediately before each mutation. Never relocate live writers.
        active = [pid for pid, refs in active_references() if any(str(source) in ref for ref in refs)]
        if active:
            record.update(action='active_alias', active_pids=active)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if record['action'] == 'active_alias':
            destination.symlink_to(os.path.relpath(str(source), str(destination.parent)), target_is_directory=True)
        else:
            before = checkpoints(source) if source.is_dir() else {}
            # Preserve the referents of relative links when directory depth changes.
            links = [(p.relative_to(source), p.resolve()) for p in source.rglob('*')
                     if p.is_symlink() and not os.path.isabs(os.readlink(str(p)))] if source.is_dir() else []
            source.rename(destination)
            try:
                source.symlink_to(os.path.relpath(str(destination), str(source.parent)), target_is_directory=destination.is_dir())
            except Exception:
                destination.rename(source)
                raise
            for relative, target in links:
                link = destination / relative
                if target == source or source in target.parents:
                    target = destination / target.relative_to(source)
                link.unlink()
                link.symlink_to(target, target_is_directory=target.is_dir())
            assert checkpoints(destination) == before, 'Checkpoint inode/size changed'
            assert source.resolve() == destination.resolve(), 'Compatibility path broken'
            record['checkpoint_count'] = len(before)
        # Persist progress after every operation, so interrupted runs are auditable.
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(records[:records.index(record) + 1], ensure_ascii=False, indent=2) + '\n')
    print('Planned operations:', len(records))


if __name__ == '__main__':
    main()

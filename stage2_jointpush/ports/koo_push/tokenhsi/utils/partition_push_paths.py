"""Machine-local Stage1 paths, shared by Partition Push launch and CPU tests."""
import os
from pathlib import Path
import re
import yaml

NAME='OpenPushPush180ContactAtWarmstart_00029000.pth'


def checkpoint_path(value):
    path=Path(value).expanduser()
    if path.is_file():return path.resolve()
    candidates=[]
    if path.is_dir():
        for candidate in path.rglob('OpenPushPush180ContactAtWarmstart_*.pth'):
            match=re.search(r'_(\d+)\.pth$',candidate.name)
            if match:candidates.append((int(match[1]),candidate))
    if not candidates:
        raise ValueError('No Stage1 checkpoint found at %s; set STAGE1_CHECKPOINT or paths.local.yaml stage1_checkpoint' % path)
    latest=max(step for step,_ in candidates)
    found=[p for step,p in candidates if step==latest]
    if len(found)!=1:raise ValueError('Multiple Stage1 runs; specify an exact .pth file')
    return found[0].resolve()


def resolve_stage1_checkpoint(value=None, *, project_root=None, environ=None):
    root=Path(project_root) if project_root is not None else Path(__file__).resolve().parents[5]
    env=os.environ if environ is None else environ
    explicit=value or env.get('STAGE1_CHECKPOINT')
    if explicit:return checkpoint_path(explicit)
    configured=env.get('PARTITION_PUSH_PATHS')
    local=Path(configured).expanduser() if configured else root/'stage2_jointpush/paths.local.yaml'
    if configured and not local.is_file():raise ValueError('PARTITION_PUSH_PATHS does not exist: %s' % local)
    if local.is_file():
        data=yaml.safe_load(local.read_text()) or {}
        if not isinstance(data,dict) or set(data)-{'stage1_checkpoint'}:
            raise ValueError('paths.local.yaml supports only stage1_checkpoint')
        if data.get('stage1_checkpoint'):
            path=Path(data['stage1_checkpoint']).expanduser()
            return checkpoint_path(path if path.is_absolute() else local.resolve().parent/path)
    for candidate in (root/'stage2_jointpush/output'/NAME,
                      root/'runs/jointpush/stage1_sources'/NAME,
                      root/'runs/jointpush/push_stage1_repro_20261012/checkpoints'/NAME):
        if candidate.is_file():return candidate.resolve()
    raise ValueError('No Stage1 checkpoint found; set STAGE1_CHECKPOINT=/path/model.pth or stage2_jointpush/paths.local.yaml stage1_checkpoint')

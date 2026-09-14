import os
import tempfile
from pathlib import Path

import torch


FIELDS = (
    "roots", "dof_pos", "dof_vel", "boxes", "base_goal", "stage_goal",
    "top_goal", "top_goal_committed", "latched_base", "retreat_goal",
    "retreat_dir",
)
POSITION_FIELDS = (
    "roots", "boxes", "base_goal", "stage_goal", "top_goal",
    "latched_base", "retreat_goal",
)


def export_bank(task):
    valid = getattr(task, "_ss_bootstrap_valid", None)
    if valid is None:
        return None
    ids = valid.nonzero(as_tuple=False).flatten()
    if len(ids) == 0:
        return None
    return {
        "version": 1,
        "env_ids": ids.cpu(),
        "box_sizes": task._box_lib._box_size.view(task.num_envs, 2, 3)[ids].cpu(),
        "base_agent": task._ss_base_agent[ids].cpu(),
        "origin_xy": task._initial_humanoid_root_states[ids, :, :2].mean(1).cpu(),
        "buffers": {
            name: getattr(task, "_ss_bootstrap_" + name)[ids].cpu()
            for name in FIELDS
        },
    }


def read_bank(path):
    state = torch.load(path, map_location="cpu")
    bank = state.get("stack_bootstrap", state)
    if bank is None or bank.get("version") != 1:
        raise ValueError("저장된 stack 시작 상태가 없습니다. 먼저 collect_stack_bootstrap.sh를 실행하세요.")
    if len(bank["env_ids"]) == 0:
        raise ValueError("저장된 stack 시작 상태가 비어 있습니다.")
    return bank


def select_bank(bank, index, num_envs):
    count = len(bank["env_ids"])
    if index < 0 or index >= count:
        raise ValueError(f"STACK_BOOTSTRAP_INDEX는 0~{count - 1}이어야 합니다.")
    ids = (torch.arange(num_envs) + index) % count
    return {
        "version": bank["version"],
        **{name: bank[name][ids] for name in (
            "env_ids", "box_sizes", "base_agent", "origin_xy"
        )},
        "buffers": {name: bank["buffers"][name][ids] for name in FIELDS},
    }


def import_bank(task, bank):
    task._init_stack_bootstrap_buffers()
    origin = task._initial_humanoid_root_states[:, :, :2].mean(1)
    offset = origin - bank["origin_xy"].to(task.device)
    for name in FIELDS:
        value = bank["buffers"][name].to(task.device).clone()
        if name in POSITION_FIELDS:
            shift = offset[:, None, :] if value.ndim == 3 else offset
            value[..., :2] += shift
        target = getattr(task, "_ss_bootstrap_" + name)
        if target.shape != value.shape:
            raise ValueError(f"stack 시작 상태의 {name} 크기가 현재 태스크와 다릅니다.")
        target.copy_(value)
    task._ss_bootstrap_valid[:] = True


def save_bank(path, bank):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(destination.parent), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(bank, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

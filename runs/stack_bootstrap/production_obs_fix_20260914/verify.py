import os
if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("6", "7"):
    raise RuntimeError("GPU must be 6 or 7")
import atexit
from pathlib import Path
lock = Path("/home/hwanhee/koo_cvpr/runs/queue/gpu_locks") / ("gpu" + os.environ["CUDA_VISIBLE_DEVICES"] + ".obsfix" + str(os.getpid()))
lock.parent.mkdir(exist_ok=True)
lock.write_text(str(os.getpid()) + "\n")
atexit.register(lock.unlink)
import isaacgym
import json
import runpy
import sys
from types import SimpleNamespace
import torch
sys.path.insert(0, "tokenhsi")
from learning.transformer.trans_players import TransPlayerContinuous
from env.tasks.humanoid import compute_humanoid_observations_max
from env.tasks.humanoid_amp import build_amp_observations
from tokenhsi.utils import stack_bootstrap
OUT = Path(os.environ["STACK_PROBE_OUT"])


def verify(self):
    task = self.env.task
    checkpoint = "/home/hwanhee/koo_cvpr/TokenHSI-masteer/output/masteer/ms57_ms18e9000_sharedgoal_w2s_boot15_6000_s0/Humanoid_14-13-21-06/nn/Humanoid.pth"
    self.restore(checkpoint)
    obs = self.env_reset()
    self.get_batch_size(obs["obs"], 1)
    if task._ss_bootstrap_import is not None:
        raise RuntimeError("Natural test must not import a bank")
    task._ss_bootstrap_frac = 1.0
    task._ss_bootstrap_eval = True
    task._init_stack_bootstrap_buffers()
    alive = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
    entries = set(); endings = {}; max_error = 0.0
    for step in range(task.max_episode_length):
        prev = task._ss_phase.clone()
        action = self.get_action(obs, True)
        obs, reward, done, info = self.env_step(self.env, action)
        if not isinstance(obs, dict):
            obs = {"obs": obs}
        live = task._compute_humanoid_obs()
        delivered = task.obs_buf[:, :live.shape[-1]]
        err = float((delivered - live).abs().max())
        max_error = max(max_error, err)
        torch.testing.assert_close(delivered, live, atol=1e-5, rtol=1e-5)
        new = alive & (prev < task.STACK) & (task._ss_phase == task.STACK)
        ids = new.nonzero(as_tuple=False).flatten()
        if len(ids):
            task._capture_stack_bootstrap(ids)
            entries.update(ids.tolist())
        ended = alive & done.view(task.num_envs, 2).any(1)
        for i in ended.nonzero(as_tuple=False).flatten().tolist():
            endings[i] = {"reason": int(task._ss_dbg_end_reason[i]), "success": bool(task._ss_success[i])}
        alive &= ~ended
        if (step + 1) % 150 == 0:
            print("FIX_PROGRESS", step + 1, "entries", len(entries), "max_body_obs_error", max_error, flush=True)
        if not bool(alive.any()):
            break
    if not entries:
        raise RuntimeError("No natural STACK entry for integration verification")
    bank = stack_bootstrap.export_bank(task)
    bank_path = OUT / "stack_bank_v2.pth"
    stack_bootstrap.save_bank(bank_path, bank)
    loaded = stack_bootstrap.read_bank(bank_path)
    torch.testing.assert_close(loaded["buffers"]["body_states"], bank["buffers"]["body_states"])
    selected = stack_bootstrap.select_bank(loaded, 0, 2)
    offset = torch.tensor([[1.25, -2.5], [-3.0, 4.0]])
    mock = SimpleNamespace(device="cpu", _initial_humanoid_root_states=(selected["origin_xy"] + offset)[:, None, :].repeat(1, 2, 1), _ss_bootstrap_valid=torch.zeros(2, dtype=torch.bool), _init_stack_bootstrap_buffers=lambda: None)
    for name in stack_bootstrap.FIELDS:
        setattr(mock, "_ss_bootstrap_" + name, torch.zeros_like(selected["buffers"][name]))
    stack_bootstrap.import_bank(mock, selected)
    torch.testing.assert_close(mock._ss_bootstrap_body_states[..., :2], selected["buffers"]["body_states"][..., :2] + offset[:, None, None, :])
    torch.testing.assert_close(mock._ss_bootstrap_body_states[..., 2:], selected["buffers"]["body_states"][..., 2:])
    legacy = "/home/hwanhee/koo_cvpr/runs/stack_bootstrap/ms57_ms18e9000_sharedgoal_w2s_boot15_6000_s0.pth"
    try:
        stack_bootstrap.read_bank(legacy)
    except ValueError:
        legacy_rejected = True
    else:
        raise RuntimeError("Legacy bank must not restore stale observations")
    reset_ids = torch.tensor(sorted(entries)[:3], device=task.device)
    rows = task.agent_rows(reset_ids)
    expected = task._ss_bootstrap_body_states[reset_ids].clone().reshape(-1, task.num_bodies, 13)
    expected_roots = task._ss_bootstrap_roots[reset_ids].clone()
    expected_dof = task._ss_bootstrap_dof_pos[reset_ids].clone().reshape(len(rows), -1)
    expected_dof_vel = task._ss_bootstrap_dof_vel[reset_ids].clone().reshape(len(rows), -1)
    task._reset_envs(reset_ids)
    torch.testing.assert_close(task._humanoid_root_states[reset_ids], expected_roots)
    torch.testing.assert_close(task._kinematic_humanoid_rigid_body_states[rows], expected)
    expected_obs = compute_humanoid_observations_max(expected[..., :3], expected[..., 3:7], expected[..., 7:10], expected[..., 10:13], task._local_root_obs_policy, task._root_height_obs_policy)
    torch.testing.assert_close(task.obs_buf[rows, :expected_obs.shape[-1]], expected_obs)
    expected_amp = build_amp_observations(expected[:, 0, :3], expected[:, 0, 3:7], expected[:, 0, 7:10], expected[:, 0, 10:13], expected_dof, expected_dof_vel, expected[:, task._key_body_ids, :3], task._local_root_obs, task._root_height_obs, task._dof_obs_size, task._dof_offsets)
    if task._enable_task_specific_disc:
        expected_amp = torch.cat((expected_amp, task._task_mask[rows].float()), -1)
    torch.testing.assert_close(task._curr_amp_obs_buf[rows], expected_amp)
    torch.testing.assert_close(task._hist_amp_obs_buf[rows], expected_amp[:, None, :].expand(-1, task._num_amp_obs_steps - 1, -1))
    result = {"natural_episodes": task.num_envs, "natural_stack_entries": len(entries), "natural_stack_fall_terminals": sum(endings[i]["reason"] == 1 for i in entries if i in endings), "natural_stack_successes": sum(endings[i]["success"] for i in entries if i in endings), "running_body_obs_max_abs_error": max_error, "bank_version": loaded["version"], "bank_count": len(bank["env_ids"]), "bootstrap_reset_ids": reset_ids.tolist(), "bootstrap_body_observation_check": "passed", "bootstrap_amp_history_check": "passed", "bank_remap_check": "passed", "legacy_bank_rejected": legacy_rejected}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2))
    print("FIX_VERIFIED", json.dumps(result), flush=True)


TransPlayerContinuous.run = verify
runpy.run_path("tokenhsi/run.py", run_name="__main__")

import os
if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("6", "7"):
    raise RuntimeError("Only physical GPU 6 or 7 is permitted")
import atexit
from pathlib import Path
lock = Path("/home/hwanhee/koo_cvpr/runs/queue/gpu_locks") / ("gpu" + os.environ["CUDA_VISIBLE_DEVICES"] + ".naturalprobe" + str(os.getpid()))
lock.parent.mkdir(exist_ok=True)
lock.write_text(str(os.getpid()) + "\n")
atexit.register(lock.unlink)
import isaacgym
import json
import random
import runpy
import sys
from types import MethodType
import numpy as np
import torch
sys.path.insert(0, "tokenhsi")
from learning.transformer.trans_players import TransPlayerContinuous
from env.tasks.humanoid import Humanoid
OUT = Path(os.environ["STACK_PROBE_OUT"])
POLICIES = {
    "ms57": "/home/hwanhee/koo_cvpr/TokenHSI-masteer/output/masteer/ms57_ms18e9000_sharedgoal_w2s_boot15_6000_s0/Humanoid_14-13-21-06/nn/Humanoid.pth",
    "ms18": "/home/hwanhee/koo_cvpr/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth",
}


def run(self):
    task = self.env.task
    if task._ss_bootstrap_import is not None or task._ss_bootstrap_frac != 0 or self.is_rnn:
        raise RuntimeError("Natural probe requires bootstrap disabled and a non-recurrent policy")
    original = Humanoid._compute_humanoid_obs
    mode = "cached"

    def body_obs(task, env_ids=None):
        obs = original(task, env_ids)
        if env_ids is not None and mode != "cached":
            rows = task.agent_rows(env_ids)
            phases = task._ss_phase[torch.div(rows, 2, rounding_mode="floor")]
            use = task.progress_rows()[rows] > 0
            if mode == "stack_only":
                use &= phases >= task.STACK
            elif mode == "live_until_stack":
                use &= phases < task.STACK
            live = original(task, None)
            obs[use] = live[rows[use]]
        return obs

    task._compute_humanoid_obs = MethodType(body_obs, task)
    self.get_batch_size(task.obs_buf, 1)
    seed_state = (torch.get_rng_state(), torch.cuda.get_rng_state(), np.random.get_state(), random.getstate())
    ticks = {k: v for k, v in vars(task).items() if "tick" in k and isinstance(v, int)}
    results = []
    initial = None
    for policy, checkpoint in POLICIES.items():
        self.restore(checkpoint)
        for mode in ("cached", "stack_only", "all_live", "live_until_stack"):
            torch.set_rng_state(seed_state[0]); torch.cuda.set_rng_state(seed_state[1])
            np.random.set_state(seed_state[2]); random.setstate(seed_state[3])
            for name, value in ticks.items():
                setattr(task, name, value)
            obs = self.env_reset()
            current_initial = {k: getattr(task, k).clone() for k in ("_humanoid_root_states", "_dof_pos", "_dof_vel", "_box_states", "_box_tar_pos", "_gt_path", "_mscale", "obs_buf")}
            if initial is None:
                initial = current_initial
            else:
                for name, value in current_initial.items():
                    torch.testing.assert_close(value, initial[name], atol=2e-4, rtol=2e-4)
            n = task.num_envs
            alive = torch.ones(n, dtype=torch.bool, device=task.device)
            entered = torch.zeros(n, dtype=torch.bool, device=task.device)
            max_phase = torch.zeros(n, dtype=torch.long, device=task.device)
            entries = {}; endings = {}; traces = {}
            all_ids = torch.arange(n, device=task.device)
            base_rows, top_rows = task._role_rows(all_ids)
            first_tilt = torch.full((n,), -1, dtype=torch.long, device=task.device)
            for step in range(task.max_episode_length):
                prev_phase = task._ss_phase.clone()
                action = self.get_action(obs, True)
                obs, reward, done, info = self.env_step(self.env, action)
                if not isinstance(obs, dict):
                    obs = {"obs": obs}
                phases = task._ss_phase
                max_phase = torch.where(alive, torch.maximum(max_phase, phases), max_phase)
                roots = task.humanoid_rows(task._humanoid_root_states)
                q = roots[:, 3:7]
                upright = 1 - 2 * (q[:, 0] ** 2 + q[:, 1] ** 2)
                new = alive & (prev_phase < task.STACK) & (phases == task.STACK) & ~entered
                entered |= new
                tilted = alive & entered & (first_tilt < 0) & (upright[top_rows] < 0.70710678)
                first_tilt[tilted] = step + 1
                record_ids = new.nonzero(as_tuple=False).flatten().tolist()
                trace_ids = [i for i in traces if bool(alive[i]) and step + 1 - entries[i]["step"] in (1, 10, 30, 60, 150)]
                if record_ids or trace_ids:
                    live = original(task, None)
                    raw_error = (task.obs_buf[:, :live.shape[-1]] - live).square().mean(-1).sqrt()
                    for i in record_ids + trace_ids:
                        br = int(base_rows[i]); tr = int(top_rows[i])
                        state = {"step": step + 1, "phase": int(phases[i]),
                                 "top_root": roots[tr].tolist(), "base_root": roots[br].tolist(),
                                 "top_upright": float(upright[tr]), "base_upright": float(upright[br]),
                                 "top_box": task.humanoid_rows(task._box_states)[tr].tolist(),
                                 "top_goal": task._box_tar_pos[tr].tolist(),
                                 "top_box_dist": float((task.humanoid_rows(task._box_states)[tr, :3] - task._box_tar_pos[tr]).norm()),
                                 "top_body_obs_rmse": float(raw_error[tr]), "base_body_obs_rmse": float(raw_error[br])}
                        if i in record_ids:
                            entries[i] = state; traces[i] = []
                            print("NATURAL_ENTRY", policy, mode, i, json.dumps(state), flush=True)
                        traces[i].append(state)
                ended = alive & done.view(n, 2).any(1)
                for i in ended.nonzero(as_tuple=False).flatten().tolist():
                    endings[i] = {"step": step + 1, "phase": int(phases[i]),
                                  "reason": int(task._ss_dbg_end_reason[i]), "terminated": bool(task._terminate_buf[i]),
                                  "success": bool(task._ss_success[i]), "top_upright": float(upright[top_rows[i]]),
                                  "base_upright": float(upright[base_rows[i]]), "top_tilt45_step": int(first_tilt[i])}
                alive &= ~ended
                if (step + 1) % 150 == 0:
                    print("NATURAL_PROGRESS", policy, mode, step + 1, "alive", int(alive.sum()), "entries", len(entries), flush=True)
                if not bool(alive.any()):
                    break
            counts = {str(p): int(((max_phase == p)).sum()) for p in range(6)}
            record = {"policy": policy, "checkpoint": checkpoint, "mode": mode, "environments": n,
                      "steps": step + 1, "dt": task.dt, "natural_stack_entries": len(entries),
                      "max_phase_counts": counts, "entries": entries, "endings": endings, "traces": traces,
                      "top_tilt45_before_done": int(((first_tilt >= 0) & entered).sum()),
                      "stack_success": sum(x["success"] for x in endings.values()),
                      "stack_survivors_at_end": int((entered & alive).sum()),
                      "bootstrap_active_count": int(task._ss_bootstrap_active.sum())}
            results.append(record)
            (OUT / "results.json").write_text(json.dumps(results, indent=2))
            print("NATURAL_RESULT", json.dumps({k: v for k, v in record.items() if k not in ("entries", "endings", "traces")}), flush=True)
    print("NATURAL_COMPLETE", flush=True)


TransPlayerContinuous.run = run
runpy.run_path("tokenhsi/run.py", run_name="__main__")

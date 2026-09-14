import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("6", "7"):
    raise RuntimeError("Only physical GPU 6 or 7 is permitted")
import atexit
from pathlib import Path
lock = Path("/home/hwanhee/koo_cvpr/runs/queue/gpu_locks") / ("gpu" + os.environ["CUDA_VISIBLE_DEVICES"] + ".obsprobe" + str(os.getpid()))
lock.parent.mkdir(exist_ok=True)
lock.write_text(str(os.getpid()) + "\n")
atexit.register(lock.unlink)
import isaacgym
import json
import runpy
import sys
from pathlib import Path
from types import MethodType
import torch
sys.path.insert(0, "tokenhsi")
from learning.transformer.trans_players import TransPlayerContinuous
from env.tasks.humanoid import Humanoid

OUT = Path(os.environ["STACK_PROBE_OUT"])
POLICIES = {
    "ms18": "/home/hwanhee/koo_cvpr/TokenHSI-masteer/output/masteer/ms18_maskteam_origscale_c06_s0/imported_00009000/nn/Humanoid_00009000.pth",
    "juan": "/home/hwanhee/juan/CVPR2027/TokenHSI-masteer/output/sequential_stack/anti_feat_top_s2/Humanoid_08-11-35-39/nn/Humanoid_00012000.pth",
}


def run(self):
    task = self.env.task
    if task.num_envs != 6 or self.is_rnn:
        raise RuntimeError("Probe requires six environments and a non-recurrent policy")
    task._ss_bootstrap_frac = 1.0
    task._ss_bootstrap_eval = True
    task._ss_rehearsal_frac = 0.0
    original = Humanoid._compute_humanoid_obs
    fixed_rows = torch.arange(6, 12, device=task.device)

    def body_obs(task, env_ids=None):
        obs = original(task, env_ids)
        if env_ids is not None and bool((task.progress_buf > 0).any()):
            rows = task.agent_rows(env_ids)
            use = (rows >= 6) & (task.progress_rows()[rows] > 0)
            live = original(task, None)
            obs[use] = live[rows[use]]
        return obs

    task._compute_humanoid_obs = MethodType(body_obs, task)
    results = []
    self.get_batch_size(task.obs_buf, 1)
    for policy, checkpoint in POLICIES.items():
        self.restore(checkpoint)
        for repeat in range(2):
            obs = self.env_reset()
            origin = task._initial_humanoid_root_states[:, :, :2].mean(1)
            shift = origin[3:] - origin[:3]
            cache = task._kinematic_humanoid_rigid_body_states
            cache[6:] = cache[:6].clone()
            cache[6:, :, :2] += shift.repeat_interleave(2, 0)[:, None, :]
            for name in ("_gt_path", "_s_end", "_arc_root", "_arc_box", "_mscale", "_prev_arc"):
                value = getattr(task, name)
                value[6:] = value[:6].clone()
                if name == "_gt_path":
                    value[6:] += shift.repeat_interleave(2, 0)[:, None, :]
            task._compute_observations(torch.arange(6, device=task.device))
            obs = self.env_reset([])
            root = task._humanoid_root_states.clone()
            root[:, :, :2] -= origin[:, None, :]
            torch.testing.assert_close(root[:3], root[3:])
            torch.testing.assert_close(task._dof_pos[:3], task._dof_pos[3:])
            torch.testing.assert_close(task._dof_vel[:3], task._dof_vel[3:])
            torch.testing.assert_close(obs["obs"][:6], obs["obs"][6:], atol=2e-4, rtol=2e-4)
            first_action = self.get_action(obs, True)
            torch.testing.assert_close(first_action[:6], first_action[6:], atol=2e-4, rtol=2e-4)
            first_done = [-1] * 6
            first_tilt = [-1] * 6
            trajectory = []
            for step in range(150):
                action = first_action if step == 0 else self.get_action(obs, True)
                obs, reward, done, info = self.env_step(self.env, action)
                if not isinstance(obs, dict):
                    obs = {"obs": obs}
                live = original(task, None)
                cached = original(task, torch.arange(6, device=task.device))
                nself = live.shape[-1]
                delivered = task.obs_buf[:, :nself]
                rows, top = task._role_rows(torch.arange(6, device=task.device))
                roots = task.humanoid_rows(task._humanoid_root_states)
                q = roots[top, 3:7]
                upright = 1 - 2 * (q[:, 0] ** 2 + q[:, 1] ** 2)
                done_env = done.view(6, 2).any(1)
                for idx in range(6):
                    if first_done[idx] < 0 and bool(done_env[idx]):
                        first_done[idx] = step + 1
                    if first_tilt[idx] < 0 and float(upright[idx]) < 0.70710678:
                        first_tilt[idx] = step + 1
                if step < 3 or step % 10 == 9:
                    rmse = lambda x: x.square().mean(-1).sqrt().tolist()
                    trajectory.append({
                        "step": step + 1,
                        "phase": task._ss_phase.tolist(),
                        "top_upright": upright.tolist(),
                        "top_root_z": roots[top, 2].tolist(),
                        "top_ang_speed": roots[top, 10:13].norm(dim=-1).tolist(),
                        "top_box_dist": (task.humanoid_rows(task._box_states)[top, :3] - task._box_tar_pos[top]).norm(dim=-1).tolist(),
                        "cache_vs_live_rmse": rmse(cached - live),
                        "delivered_vs_live_rmse": rmse(delivered - live),
                    })
                if step == 0:
                    torch.testing.assert_close(root[:3], root[3:])
                    print("PROBE_FIRST", policy, repeat, json.dumps(trajectory[-1]), flush=True)
            record = {
                "policy": policy, "checkpoint": checkpoint, "repeat": repeat,
                "source_env_ids": task._ss_bootstrap_import["env_ids"].tolist(),
                "modes": ["cached"] * 3 + ["live_after_first_step"] * 3,
                "first_done_step": first_done, "first_top_tilt45_step": first_tilt,
                "dt": task.dt, "trajectory": trajectory,
                "first_action_max_pair_diff": float((first_action[:6] - first_action[6:]).abs().max()),
            }
            results.append(record)
            (OUT / "results.json").write_text(json.dumps(results, indent=2))
            print("PROBE_RESULT", json.dumps({k: v for k, v in record.items() if k != "trajectory"}), flush=True)
    print("PROBE_COMPLETE", flush=True)


TransPlayerContinuous.run = run
runpy.run_path("tokenhsi/run.py", run_name="__main__")

"""Register the planner-free mid-path release diagnostic task."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from isaacgym import gymapi as _gymapi  # noqa: F401
import torch

torch.cuda.set_device(int(os.environ.get("STACK_VIEW_GPU", "1")))

COORD_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = COORD_ROOT.parent
MASTEER_ROOT = WORKSPACE / "TokenHSI-masteer"
sys.path.insert(0, str(MASTEER_ROOT / "tokenhsi"))
sys.path.insert(0, str(COORD_ROOT))

import utils.parse_task as task_registry  # noqa: E402
from stack_planner.midpath_release_env import HumanoidMAMidpathReleaseView  # noqa: E402

task_registry.HumanoidMAMidpathReleaseView = HumanoidMAMidpathReleaseView

import run as tokenhsi_run  # noqa: E402
from utils.config import get_args, load_cfg, set_np_formatting, set_seed  # noqa: E402


def main():
    """Repeat scenarios with explicit done-env resets, never reset(None)."""
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, _ = load_cfg(args)
    seed = set_seed(
        cfg_train["params"].get("seed", 0),
        cfg_train["params"].get("torch_deterministic", False),
    )
    cfg_train["params"]["seed"] = seed
    cfg_train["params"]["config"]["seed"] = seed
    cfg_train["params"]["config"]["train_dir"] = args.output_path
    if args.motion_file:
        cfg["env"]["motion_file"] = args.motion_file

    tokenhsi_run.args = args
    tokenhsi_run.cfg = cfg
    tokenhsi_run.cfg_train = cfg_train
    runner = tokenhsi_run.build_alg_runner(tokenhsi_run.RLGPUAlgoObserver())
    runner.load(cfg_train)
    runner.reset()
    player = runner.create_player()
    player.restore(args.checkpoint)
    player.model.eval()
    task = player.env.task
    if player.is_rnn:
        raise RuntimeError("midpath release repeat loop only supports stateless policies")
    # Do not depend on whether the framework's construction reset happened
    # before or after the subclass finished initializing its diagnostic state.
    task.install_midpath_paths()
    task._compute_observations()
    sync_debug = int(os.environ.get("STACK_CUDA_SYNC_DEBUG", "0")) != 0

    def sync_cuda(stage):
        if not sync_debug:
            return
        try:
            torch.cuda.synchronize(task.device)
        except RuntimeError as error:
            raise RuntimeError(
                "CUDA/PhysX failed before or during {} (this is the first "
                "synchronized failure point)".format(stage)
            ) from error

    # Construction already did a complete reset and produced the initial obs.
    # rl_games' standard one-env player instead calls reset(None) once here and
    # again at every game boundary; that full reset breaks this GPU pipeline.
    obs = torch.clamp(
        task.obs_buf, -player.env.clip_obs, player.env.clip_obs
    ).to(player.device)
    player.get_batch_size(obs, 1)
    print("[midpath-release] repeat loop active; explicit done-env resets only",
          flush=True)
    episodes = torch.zeros(task.num_envs, dtype=torch.long, device=task.device)
    while True:
        action = player.get_action({"obs": obs}, is_determenistic=True)
        obs, _, done_rows, info = player.env.step(action)
        sync_cuda("env.step")
        player._post_step(info)
        done_env = done_rows.reshape(task.num_envs, task.num_agents).any(dim=1)
        if done_env.any():
            episodes[done_env] += 1
            completed = torch.nonzero(done_env, as_tuple=False).squeeze(-1)
            print("[midpath-release] resetting envs={} completed={}".format(
                completed.tolist(), episodes[completed].tolist()), flush=True)
            row_ids = torch.nonzero(
                done_env.repeat_interleave(task.num_agents), as_tuple=False
            ).squeeze(-1)
            player.env.reset(row_ids)
            task.install_midpath_paths(completed)
            task._compute_observations(completed)
            obs = torch.clamp(
                task.obs_buf, -player.env.clip_obs, player.env.clip_obs
            ).to(player.device)
            sync_cuda("done-env reset")


if __name__ == "__main__":
    main()

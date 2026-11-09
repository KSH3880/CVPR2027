"""Inference entry point for Isaac Gym / CUDA 11 on newer GPUs."""
import runpy
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tokenhsi'), str(ROOT)]
import isaacgym  # Must precede torch.
import torch

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10 and int(torch.version.cuda.split('.')[0]) < 12:
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_nvfuser_enabled(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    # CUDA 11 NVRTC cannot compile prod reductions for Blackwell. This
    # three-axis integer product has an equivalent precompiled-kernel form.
    import inspect
    from push_reference.utils import size_rsi
    original = size_rsi.task_probabilities
    source = inspect.getsource(original)
    before = '(bounds[:, 1] - bounds[:, 0] + 1).prod()'
    assert source.count(before) == 1
    after = '((bounds[0, 1] - bounds[0, 0] + 1) * (bounds[1, 1] - bounds[1, 0] + 1) * (bounds[2, 1] - bounds[2, 0] + 1))'
    namespace = dict(original.__globals__)
    exec(compile(source.replace(before, after), original.__code__.co_filename, 'exec'), namespace)
    original.__code__ = namespace['task_probabilities'].__code__

if os.environ.get('VIEW_LOOP') == '1':
    from learning.push_kd_player import PushKDPlayer

    def continuous_view(self):
        """Run deterministic policy continuously; reset completed scenes only."""
        obs = self.env_reset()
        self.get_batch_size(obs['obs'], 1)
        done_ids = []
        episodes = 0
        steps = 0
        print('PUSH_VIEW_LOOP started (close viewer or press Escape to exit)', flush=True)
        while True:
            obs = self.env_reset(done_ids)
            action = self.get_action(obs, True)
            obs, _, done, _ = self.env_step(self.env, action)
            steps += 1
            if steps == 30 and os.environ.get('VIEW_CAPTURE'):
                ref = self.env.reference
                ref.gym.write_viewer_image_to_file(ref.viewer, os.environ['VIEW_CAPTURE'])
                print('PUSH_VIEW_CAPTURE saved', flush=True)
            done_ids = done.nonzero().flatten()
            if len(done_ids):
                episodes += len(done_ids) // self.env.reference.num_agents
                print('PUSH_VIEW_LOOP completed scenes={}'.format(episodes), flush=True)

    PushKDPlayer.run_eval = continuous_view
runpy.run_path(str(ROOT / 'tokenhsi/run.py'), run_name='__main__')

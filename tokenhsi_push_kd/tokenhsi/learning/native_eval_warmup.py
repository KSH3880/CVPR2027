"""Warm-up before the native SIT/CARRY/CLIMB eval, shared by the student (PushKDPlayer) and the
original-code teacher eval (scripts/eval_teacher_original.py) so both follow the same protocol.

The original TokenHSI eval counts the first episode after env creation, where CLIMB collapses
(about 3% vs 99.8% in later repeats, for the teacher in the original code too). Running the policy
for longer than one episode (600 steps) first removes that artifact. Warm-up steps are not counted.
"""
import os

DEFAULT_STEPS = 700  # > episodeLength 600, so every env finishes at least one full episode


def warmup_steps():
    return int(os.environ.get('NATIVE_EVAL_WARMUP', DEFAULT_STEPS))


def run_warmup(player, steps):
    """Step the evaluated policy (deterministic) and reset finished envs, as run_eval does."""
    print('native eval warm-up: {} steps (not counted)'.format(steps), flush=True)
    obs = player.env_reset()
    player.get_batch_size(obs['obs'], 1)
    done_ids = []
    for _ in range(steps):
        obs = player.env_reset(done_ids)
        _, _, done, _ = player.env_step(player.env, player.get_action(obs, True))
        done_ids = done.nonzero(as_tuple=False)[:, 0]

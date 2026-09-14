import isaacgym
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "tokenhsi")
from learning.transformer.trans_players import TransPlayerContinuous
from learning.common_agent import CommonAgent
from learning.amp_agent import AMPAgent


def verify(self):
    obs = self.env_reset()
    task = self.env.task
    bank = task._ss_bootstrap_import
    if bank is None:
        raise RuntimeError("No imported stack bank")
    expected_roots = task._ss_bootstrap_roots.clone()
    expected_boxes = task._ss_bootstrap_boxes.clone()
    first_phase = task._ss_phase.tolist()
    torch.testing.assert_close(task._humanoid_root_states, expected_roots)
    torch.testing.assert_close(task.agent_axis(task._box_states), expected_boxes)
    torch.testing.assert_close(task._box_lib._box_size.view(task.num_envs, 2, 3), bank["box_sizes"].to(task.device))
    torch.testing.assert_close(task._ss_base_agent, bank["base_agent"].to(task.device))
    if not bool((task._ss_phase == task.STACK).all()):
        raise RuntimeError("First reset must start in STACK")
    for _ in range(3):
        action = self.get_action(obs, True)
        obs, reward, done, info = self.env_step(self.env, action)
    self.env_reset(torch.arange(task.num_envs, device=task.device))
    if not bool((task._ss_phase == task.STACK).all()):
        raise RuntimeError("Second reset must start in STACK")
    torch.testing.assert_close(task._humanoid_root_states, expected_roots)
    torch.testing.assert_close(task.agent_axis(task._box_states), expected_boxes)
    agent = AMPAgent.__new__(AMPAgent)
    agent.vec_env = SimpleNamespace(env=SimpleNamespace(task=task))
    original = CommonAgent.get_full_state_weights
    try:
        CommonAgent.get_full_state_weights = lambda self: {"model": {}}
        weights = agent.get_full_state_weights()
    finally:
        CommonAgent.get_full_state_weights = original
    if len(weights["stack_bootstrap"]["env_ids"]) != task.num_envs:
        raise RuntimeError("Training checkpoint hook must include bank")
    report = {
        "first_phase": first_phase,
        "second_phase": task._ss_phase.tolist(),
        "stack_phase": task.STACK,
        "source_env_ids": bank["env_ids"].tolist(),
        "box_sizes": bank["box_sizes"].tolist(),
        "base_agent": task._ss_base_agent.tolist(),
        "checkpoint_bank_count": len(weights["stack_bootstrap"]["env_ids"]),
        "simulation_steps": 3,
    }
    Path("../runs/stack_bootstrap/ms57_reset_check.json").write_text(json.dumps(report, indent=2))
    print("STACK_RESET_CHECK", json.dumps(report))


TransPlayerContinuous.run = verify
runpy.run_path("tokenhsi/run.py", run_name="__main__")

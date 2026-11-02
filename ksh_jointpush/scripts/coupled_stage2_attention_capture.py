"""Read-only actor Stage2 inputs from the standard player on fixed PUSH group layouts."""
from pathlib import Path
import hashlib
import json
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ksh_jointpush.scripts.coupled_base_run import install
install()
import numpy as np
import torch
from learning.multi_agent.ma_players import MAPlayerContinuous
from utils import coupled_push_spec as spec


PATTERNS = {'1plus3': [0, 1, 1, 1], '2plus2': [0, 0, 2, 2], '4': [0, 0, 0, 0]}
SELECTED = os.environ.get('STAGE2_ATTN_PATTERNS', '1plus3').split(',')
if not SELECTED or len(set(SELECTED)) != len(SELECTED) or any(key not in PATTERNS for key in SELECTED):
    raise ValueError('STAGE2_ATTN_PATTERNS accepts unique comma-separated 1plus3,2plus2,4')
CURRENT_GROUPS = PATTERNS[SELECTED[0]]


def fixed_groups(count, people, probability, device, preset='random_scenario'):
    assert people == 4
    return torch.tensor(CURRENT_GROUPS, device=device).expand(count, -1).clone()


spec.sample_groups = fixed_groups


def digest(player):
    result = hashlib.sha256()
    for state in (player.model.state_dict(), player.running_mean_std.state_dict()):
        for key, value in sorted(state.items()):
            result.update(key.encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


@torch.no_grad()
def capture_case(self, out, pattern):
    task = self.env.task
    assert task.num_envs == 1 and task.num_agents == 4 and not self.is_rnn
    self.model.eval()
    self.running_mean_std.eval()
    self._eval = False
    self._reward_debug = False
    out.mkdir(parents=True, exist_ok=True)
    if any((out/name).exists() for name in ('capture.json', 'states.json', 'actor_coordination_inputs.pt')):
        raise ValueError('Use a new STAGE2_ATTN_OUTPUT directory; existing captures are preserved')
    branch = self.model.a2c_network.coordination
    before = digest(self)
    inputs, outputs, packets, groups, observations, states = [], [], [], [], [], []
    pending = {}
    wanted = False

    def pre(module, args, kwargs):
        if wanted:
            assert not pending
            pending.update(humans=args[0].detach().cpu().clone(),
                           packet=args[2].detach().cpu().clone(),
                           groups=kwargs['groups'].detach().cpu().clone())

    def post(module, args, result):
        if wanted:
            assert pending
            pending['output'] = result.detach().cpu().clone()

    hooks = [branch.register_forward_pre_hook(pre, with_kwargs=True),
             branch.register_forward_hook(post)]
    action_error = None
    episodes = int(os.environ.get('STAGE2_ATTN_EPISODES', '3'))
    horizon = int(os.environ.get('STAGE2_ATTN_STEPS', '600'))
    interval = int(os.environ.get('STAGE2_ATTN_INTERVAL', '10'))
    endings = []
    try:
        for episode in range(episodes):
            obs = self.env_reset()
            self.get_batch_size(obs['obs'], 1)
            assert task._groups[0].tolist() == PATTERNS[pattern]
            for step in range(horizon):
                wanted = step % interval == 0
                if wanted:
                    assignment = task._agent_box_assignment[0]
                    slots = assignment.unique()
                    states.append(dict(episode=episode, step=step, control_dt=float(task.dt),
                        groups=task._groups[0].tolist(), members=task._members[0].tolist(),
                        humans=task._humanoid_root_states[0, :, :3].tolist(),
                        physical_slots=slots.tolist(), assignment=assignment.tolist(),
                        boxes=task._box_states[0, slots, :7].tolist(),
                        box_sizes=task._box_size[0, slots].tolist(),
                        goals=task._task_goal_positions()[0].tolist(),
                        center_goals=task._tar_pos[0].tolist(),
                        contact=task.relation_runtime.contact[0].tolist(),
                        group_gate=task.relation_runtime.gate[0].tolist(),
                        success=task.relation_runtime.done[0].tolist()))
                    observations.append(obs['obs'].detach().cpu().clone())
                if action_error is None:
                    cpu_before = torch.get_rng_state()
                    cuda_before = torch.cuda.get_rng_state()
                action = self.get_action(obs, self.is_determenistic)
                assert bool(torch.isfinite(action).all())
                if wanted:
                    inputs.append(pending['humans'])
                    packets.append(pending['packet'])
                    groups.append(pending['groups'])
                    outputs.append(pending['output'])
                    pending.clear()
                if action_error is None:
                    cpu_after = torch.get_rng_state()
                    cuda_after = torch.cuda.get_rng_state()
                    wanted = False
                    torch.set_rng_state(cpu_before)
                    torch.cuda.set_rng_state(cuda_before)
                    baseline = self.get_action(obs, self.is_determenistic)
                    torch.set_rng_state(cpu_after)
                    torch.cuda.set_rng_state(cuda_after)
                    action_error = float((action-baseline).abs().max())
                    assert action_error == 0., action_error
                wanted = False
                raw, reward, done, info = self.env_step(self.env, action)
                obs = self.obs_to_torch(raw)
                assert bool(torch.isfinite(obs['obs']).all() and torch.isfinite(reward).all())
                if bool(done.any()):
                    endings.append(dict(episode=episode, last_step=step,
                                        reason='environment_done'))
                    break
            else:
                endings.append(dict(episode=episode, last_step=horizon-1,
                                    reason='capture_horizon'))
            print('STAGE2_ATTN_EPISODE', episode, 'samples', len(states), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    assert len(states) == len(inputs) == len(outputs)
    assert digest(self) == before
    torch.save(dict(humans=torch.cat(inputs), packet=torch.cat(packets),
                    groups=torch.cat(groups), output=torch.cat(outputs),
                    observations=torch.cat(observations)), out/'actor_coordination_inputs.pt')
    (out/'states.json').write_text(json.dumps(states, indent=2)+'\n')
    checkpoint = Path(task.cfg['args'].checkpoint).resolve()
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        actor_branch='a2c_network.coordination', groups=PATTERNS[pattern], pattern=pattern, frames=len(states),
        horizon=horizon, sample_interval=interval, seed=int(task.cfg['args'].seed),
        episodes=episodes, endings=endings, layer_count=len(branch.blocks),
        head_count=branch.blocks[0].attention.num_heads, deterministic=bool(self.is_determenistic),
        unchanged_action_max_error=action_error, model_and_rms_unchanged=True,
        optimizer_steps=0, shuffle_in_original_policy_unchanged=True,
        description='Actual normalized actor-encoder outputs captured before human shuffle; original canonical outputs retained as replay reference.')
    (out/'capture.json').write_text(json.dumps(report, indent=2)+'\n')
    print('STAGE2_ATTN_CAPTURE_COMPLETE', json.dumps(report), flush=True)
    return report


def capture(self):
    global CURRENT_GROUPS
    out = Path(os.environ['STAGE2_ATTN_OUTPUT'])
    reports = []
    for pattern in SELECTED:
        CURRENT_GROUPS = PATTERNS[pattern]
        target = out if len(SELECTED) == 1 else out/pattern
        reports.append(capture_case(self, target, pattern))
    return reports


MAPlayerContinuous.run_eval = capture
runpy.run_path(str(ROOT/'ksh_jointpush/coupled_base/tokenhsi/run.py'), run_name='__main__')

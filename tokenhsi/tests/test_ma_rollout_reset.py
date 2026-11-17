from types import SimpleNamespace

import isaacgym
import pytest
import torch

from env.tasks.multi_agent.vec_task_wrapper_ma import VecTaskPythonWrapperMA
from learning.multi_agent.ma_agent import MAAgent


class Buffer:
    def __init__(self, steps, people):
        self.steps = steps
        self.tensor_dict = {
            'values': torch.zeros(steps, people, 1),
            'next_values': torch.zeros(steps, people, 1),
            'rewards': torch.zeros(steps, people, 1),
            'dones': torch.zeros(steps, people, dtype=torch.bool),
            'amp_obs': torch.zeros(steps, people, 2),
        }

    def update_data(self, name, step, value):
        if name not in self.tensor_dict:
            self.tensor_dict[name] = value.new_zeros(self.steps, *value.shape)
        self.tensor_dict[name][step] = value


def learner(agents, terminal_envs, initially_done=False):
    steps, envs = 32, 3
    people = agents*envs
    ages = torch.zeros(envs, dtype=torch.long)
    resets, transitions, rewards = [], [], []

    def reset(ids):
        resets.append(torch.as_tensor(ids).long().tolist())
        ages[ids] = 0

    task = SimpleNamespace(reset=reset, viewer=None)
    wrapper = SimpleNamespace(task=task, num_agents=agents,
        _policy_observation=lambda: ages[:, None].float().clone())

    def env_reset(rows):
        return {'obs': VecTaskPythonWrapperMA.reset(wrapper, rows)}

    def env_step(actions):
        transitions.append(ages.clone())
        ages.add_(1)
        ended = torch.zeros(envs, dtype=torch.bool)
        ended[list(terminal_envs)] = ages[list(terminal_envs)] >= steps
        # One agent falls; the other agents share the reset and retain bootstrap.
        terminated = torch.zeros(people, dtype=torch.bool)
        terminated[ended.nonzero().flatten()*agents] = True
        return {'obs': ages[:, None].float().clone()}, torch.ones(people, 1), ended.repeat_interleave(agents), {
            'terminate': terminated, 'amp_obs': torch.zeros(people, 2)}

    agent = SimpleNamespace(
        horizon_length=steps, num_agents=agents, batch_size=steps*people,
        dones=torch.full((people,), initially_done, dtype=torch.bool),
        set_eval=lambda: None, update_list=['values'], use_action_masks=False,
        env_reset=env_reset, env_step=env_step, _rand_action_probs=None,
        get_action_values=lambda obs, _: {'values': torch.full((people, 1), 5.),
            'actions': torch.zeros(people, 2), 'rand_action_mask': torch.ones(people)},
        rewards_shaper=lambda x: x, _eval_critic=lambda obs: torch.full((people, 1), 5.),
        experience_buffer=Buffer(steps, people),
        current_rewards=torch.zeros(people, 1), current_lengths=torch.zeros(people),
        game_rewards=SimpleNamespace(update=lambda x: rewards.append(x.clone())),
        game_lengths=SimpleNamespace(update=lambda x: None),
        algo_observer=SimpleNamespace(process_infos=lambda *args: None),
        vec_env=SimpleNamespace(env=SimpleNamespace(task=task)),
        _calc_amp_rewards=lambda x: {}, _combine_rewards=lambda x, _: x,
        discount_values=lambda d, v, r, n: torch.zeros_like(r), tensor_list=[],
        _group_agent_tensor=lambda x: x,
    )
    return agent, resets, transitions, rewards


@pytest.mark.parametrize('agents', [1, 4])
@pytest.mark.parametrize('terminal_envs', [(), (1,), (0, 2)])
def test_last_step_done_resets_before_next_rollout_action(agents, terminal_envs):
    agent, resets, transitions, rewards = learner(agents, terminal_envs)
    MAAgent.play_steps(agent)
    MAAgent.play_steps(agent)

    assert resets[32] == list(terminal_envs)
    expected = torch.full((3,), 32)
    expected[list(terminal_envs)] = 0
    torch.testing.assert_close(transitions[32], expected)
    completed = torch.cat([x.flatten() for x in rewards])
    torch.testing.assert_close(completed, torch.full((2*len(terminal_envs),), 32.))
    bootstrap = agent.experience_buffer.tensor_dict['next_values'][-1]
    expected = torch.full((3*agents, 1), 5.)
    expected[torch.tensor(terminal_envs, dtype=torch.long)*agents] = 0
    torch.testing.assert_close(bootstrap, expected)


@pytest.mark.parametrize('agents', [1, 4])
def test_initial_done_flags_reset_each_scene_once(agents):
    agent, resets, transitions, _ = learner(agents, (), initially_done=True)
    MAAgent.play_steps(agent)
    assert resets[0] == [0, 1, 2]
    torch.testing.assert_close(transitions[0], torch.zeros(3, dtype=torch.long))

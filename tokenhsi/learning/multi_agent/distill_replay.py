import torch


class DistillReplay:
    def __init__(self, capacity, observation_size, action_size, device):
        self.obs = torch.empty(capacity, observation_size, device=device)
        self.actions = torch.empty(capacity, action_size, device=device)
        self.size = self.cursor = 0

    def add(self, obs, actions):
        capacity = len(self.obs)
        obs, actions = obs[-capacity:], actions[-capacity:]
        index = (torch.arange(len(obs), device=obs.device)+self.cursor)%capacity
        self.obs[index], self.actions[index] = obs, actions
        self.cursor = (self.cursor+len(obs))%capacity
        self.size = min(capacity, self.size+len(obs))

    def sample(self, count):
        index = torch.randint(self.size, (count,), device=self.obs.device)
        return self.obs[index], self.actions[index]

    def state_dict(self):
        return dict(obs=self.obs[:self.size].clone(), actions=self.actions[:self.size].clone(),
                    size=self.size, cursor=self.cursor)

    def load_state_dict(self, state):
        if state['size'] > len(self.obs) or state['obs'].shape[1:] != self.obs.shape[1:]:
            raise ValueError('Distillation replay dimensions differ')
        if state['actions'].shape[1:] != self.actions.shape[1:]:
            raise ValueError('Distillation replay action dimensions differ')
        self.size, self.cursor = state['size'], state['cursor']
        self.obs[:self.size].copy_(state['obs'])
        self.actions[:self.size].copy_(state['actions'])

"""Group-weighted training success statistics, independent of the simulator."""

import torch


class Stage2TaskSuccessMetrics:
    names = ('push_solo', 'push_2', 'push_3', 'push_4',
             'sit', 'climb', 'holding_at', 'holding_ontop')

    def __init__(self, num_envs, num_agents, device):
        self.ever = torch.zeros(num_envs, num_agents, dtype=torch.bool, device=device)
        # Current successes, group-steps, final successes, ever successes, episodes.
        self.sums = torch.zeros(len(self.names), 5, device=device)

    def reset(self, ids):
        self.ever[ids] = False

    @torch.no_grad()
    def observe(self, done, groups, templates, ending):
        same = groups[:, :, None] == groups[:, None, :]
        members = same.sum(-1)
        current = (done[:, None, :] | ~same).all(-1)
        self.ever |= current
        # Each physical PUSH group contributes once, regardless of member count.
        owners = torch.arange(groups.shape[1], device=groups.device)[None]
        valid = (groups == owners) & (templates >= 1) & (templates <= 5)
        bucket = torch.where(templates == 5, members - 1, templates + 3)
        ends = ending[:, None].expand_as(current)
        values = torch.stack((current, torch.ones_like(current), current & ends,
                              self.ever & ends, ends), -1).to(self.sums.dtype)
        values *= valid[..., None]
        self.sums.index_add_(0, bucket.flatten().clamp(0, 7), values.flatten(0, 1))

    def consume(self):
        result = {}
        for name, row in zip(self.names, self.sums):
            prefix = 'task_success/' + name + '/'
            result[prefix + 'group_steps'] = row[1].clone()
            result[prefix + 'completed'] = row[4].clone()
            if row[1] > 0:
                result[prefix + 'current_rate'] = row[0] / row[1]
            if row[4] > 0:
                result[prefix + 'final_rate'] = row[2] / row[4]
                result[prefix + 'ever_rate'] = row[3] / row[4]
        self.sums.zero_()
        return result

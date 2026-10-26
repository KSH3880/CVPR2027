"""Completed-agent episode metrics, independent of simulator and reward shaping."""
import torch


class Stage1EpisodeMetrics:
    NAMES = ('holding', 'sit', 'climb', 'holding_at', 'holding_ontop',
             'push_at', 'push_at_loco', 'push_at_rsi')
    FIELDS = ('completed_agents', 'final_success', 'ever_success',
              'start_distance_m', 'remaining_distance_m', 'distance_reduction_m',
              'progress_fraction', 'best_progress_fraction')

    def __init__(self, num_envs, num_agents, device):
        shape = (num_envs, num_agents)
        self.template = torch.zeros(shape, dtype=torch.long, device=device)
        self.rsi = torch.zeros(shape, dtype=torch.bool, device=device)
        self.active = torch.zeros_like(self.rsi)
        self.ever = torch.zeros_like(self.rsi)
        self.start = torch.zeros(shape, device=device)
        self.best = torch.zeros_like(self.start)
        self.sums = torch.zeros(len(self.NAMES), len(self.FIELDS), device=device)

    def reset(self, ids, template, rsi, distance, success):
        self.template[ids] = template
        self.rsi[ids] = rsi
        self.active[ids] = True
        self.ever[ids] = success
        self.start[ids] = distance
        self.best[ids] = 0

    def progress(self, distance):
        # Post-RSI reset distance is the denominator; RSI placement gets no credit.
        return ((self.start - distance) / self.start.clamp_min(1e-6)).clamp(0, 1)

    def observe(self, distance, success):
        self.ever |= success & self.active
        self.best = torch.maximum(self.best, self.progress(distance))

    def finish(self, ending, distance, success):
        mask = self.active & ending[:, None]
        values = torch.stack((torch.ones_like(distance), success.float(),
                              self.ever.float(), self.start, distance,
                              self.start - distance, self.progress(distance), self.best), -1)
        # Every agent contributes once to its task, PUSH also to its reset-mode bin.
        for group in range(len(self.NAMES)):
            selected = mask & ((self.template == group) if group < 6 else
                               ((self.template == 5) & (self.rsi == (group == 7))))
            self.sums[group] += (values * selected[..., None]).sum((0, 1))
        self.active &= ~mask

    def consume(self):
        result = {}
        for group, name in enumerate(self.NAMES):
            sums = self.sums[group]
            prefix = 'episode/' + name + '/'
            result[prefix + self.FIELDS[0]] = sums[0].clone()
            if sums[0] > 0:
                for index, field in enumerate(self.FIELDS[1:], 1):
                    if index <= 2 or name.startswith('push_at'):
                        result[prefix + field] = (sums[index] / sums[0]).clone()
        self.sums.zero_()
        return result

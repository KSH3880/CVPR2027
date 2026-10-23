"""Read-only PUSH motion accounting, sampled at each physical tick (XY metres)."""
import torch


def tick_motion(before, after, goal, contact):
    delta = after[..., :2] - before[..., :2]
    distance = delta.norm(dim=-1)
    # Signed radial improvement telescopes over a fixed-goal episode. Retreat is negative.
    progress = (goal[..., :2] - before[..., :2]).norm(dim=-1) - (
        goal[..., :2] - after[..., :2]).norm(dim=-1)
    return torch.stack((distance, distance * contact, progress,
                        progress * contact, progress.clamp_min(0) * contact), -1)


class PushMotionMetrics:
    names = ('travel', 'contact_travel', 'goal_progress', 'contact_goal_progress',
             'contact_goal_forward')

    def __init__(self, shape, device):
        self.step = torch.zeros(*shape, len(self.names), device=device)
        self.episode = torch.zeros_like(self.step)
        self.sums = {}
        self.counts = {}

    def add(self, name, values, mask):
        self.sums[name] = self.sums.get(name, 0) + (values * mask).sum().detach()
        self.counts[name] = self.counts.get(name, 0) + mask.sum().detach()

    def begin_step(self):
        self.step.zero_()

    def tick(self, before, after, goal, contact, active):
        self.step += tick_motion(before, after, goal, contact) * active[..., None]
        self.add('contact_tick_fraction', contact.float(), active)

    def finish_step(self, active, contact, at, goal_distance, dt):
        self.episode += self.step
        self.add('contact', contact.float(), active)
        self.add('at', at.float(), active)
        self.add('contact_and_at', (contact & at).float(), active)
        self.add('goal_distance_m', goal_distance, active)
        for i, name in enumerate(self.names):
            self.add(name + '_m_per_step', self.step[..., i], active)
            self.add(name + '_m_per_s', self.step[..., i] / dt, active)

    def finish_episodes(self, ending, active):
        mask = ending[:, None] & active
        for i, name in enumerate(self.names):
            self.add('episode_' + name + '_m', self.episode[..., i], mask)

    def reset(self, ids):
        self.episode[ids] = 0
        self.step[ids] = 0

    def consume(self):
        result = {'push/' + key: value / self.counts[key].clamp_min(1)
                  for key, value in self.sums.items()}
        self.sums, self.counts = {}, {}
        return result

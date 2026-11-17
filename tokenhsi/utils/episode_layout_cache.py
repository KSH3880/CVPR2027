import torch


class EpisodeLayoutCache:
    def __init__(self, num_envs, entries, bucket_count=27):
        self.num_envs = num_envs
        self.entries = entries
        self.bucket_count = bucket_count
        self.data = None

    def store(self, bucket, ids, alpha, tensors):
        if self.data is None:
            device = ids.device
            self.data = dict(
                alpha=torch.full((self.num_envs, self.bucket_count, self.entries), float('nan'), device=device),
                next=torch.zeros(self.num_envs, self.bucket_count, dtype=torch.long, device=device),
                tensors={name: value.new_empty((self.num_envs, self.bucket_count, self.entries, *value.shape[1:]))
                         for name, value in tensors.items()})
        data = self.data
        slots = data['next'][ids, bucket]
        for name, value in tensors.items():
            data['tensors'][name][ids, bucket, slots] = value[ids]
        data['alpha'][ids, bucket, slots] = alpha
        data['next'][ids, bucket] = (slots+1) % self.entries

    def lookup(self, bucket, ids, alpha):
        if self.data is None:
            return torch.full_like(ids, -1)
        valid = self.data['alpha'][ids, bucket] == alpha[:, None]
        scores = torch.rand(valid.shape, device=ids.device).masked_fill(~valid, -1.)
        return torch.where(valid.any(-1), scores.argmax(-1), -1)

    def restore(self, bucket, ids, slots, tensors):
        stored = self.data['tensors']
        for name, target in tensors.items():
            target[ids] = stored[name][ids, bucket, slots]


def buckets(mode, family, alpha, same_platform=None, final_only=False):
    task = torch.where(family == 4, 2, family-1)
    result = mode*3+task
    stride = 9
    if not final_only:
        distance = torch.where(alpha == 0, 0, torch.where(alpha == 1, 2, 1))
        result = result*3+distance
        stride = 27
    return result if same_platform is None else result+stride*same_platform.long()


def validate(env):
    spec = env['sensorScene']['bridge']['layout_cache']
    if (env['sensorScene']['variant'] != 'bridge_before_fixed_v1'
            or 'carry_boundary_rsi' in env['sensorScene']['bridge']):
        raise ValueError('Layout caching requires the original fixed BEFORE Bridge')
    if spec.get('entries_per_bucket') != 2 or not 0 <= spec.get('refresh_probability', -1) <= 1:
        raise ValueError('Layout cache requires two entries per bucket and a valid refresh probability')
    return spec

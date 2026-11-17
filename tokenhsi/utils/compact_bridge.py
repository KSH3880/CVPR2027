import torch


NAME = 'stage2_sensor_900_cliff_bridge_compact'
KINDS = ('full', 'entry_before', 'entry_after', 'exit_before')


def progress(update, spec):
    return min(1., max(0., (update-spec['warmup_updates'])/spec['anneal_updates']))


def sample_distance(count, update, spec, device, evaluation=False):
    if evaluation:
        return torch.ones(count, device=device)
    choice = torch.multinomial(torch.tensor(spec['mixture'], device=device), count, True)
    return torch.where(choice == 0, 0., torch.where(choice == 2, 1., progress(update, spec)))


def visitors(start, destination, pushers):
    return torch.stack([(((start == side) | (destination == side)) & ~pushers).sum(-1)
                        for side in (1, 2)], -1)


def source_depth(half, phases, alpha, boundary=None):
    normal = phases == 0
    if boundary is None:
        boundary = torch.zeros_like(phases)
    scores = torch.arange(phases.shape[1], device=phases.device).expand_as(phases).clone()
    scores -= (boundary > 0).long()*phases.shape[1]
    scores += (~normal).long()*100
    order = scores.argsort(-1)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(phases.shape[1], device=phases.device).expand_as(phases))
    pitch = 1.2+.3*alpha
    maximum = 2*half[:, 0]-.8-pitch*(normal.sum(-1)-1).clamp_min(0)
    high = torch.minimum(.8+.4+.8*alpha, maximum).clamp_min(.8)
    base = .8+(high-.8)*torch.rand_like(alpha)
    return base[:, None]+pitch[:, None]*rank


def goal_band(alpha, rank, radius, extent, required=0.):
    low = torch.maximum(radius+.8+.7*alpha+1.2*rank, radius+required)
    high = torch.maximum(radius+1.5+1.5*alpha+1.2*rank, low+.5)
    high = torch.minimum(high, extent-radius-.5)
    return low, high


def choose_boundary(eligible, spec, evaluation=False):
    n, m = eligible.shape
    kind = torch.zeros(n, m, dtype=torch.long, device=eligible.device)
    if evaluation or not n:
        return kind
    scores = torch.rand(n, m, device=eligible.device).masked_fill(~eligible, -1)
    person = scores.argmax(-1)
    modes = torch.multinomial(torch.tensor(spec['probabilities'], device=eligible.device), n, True)
    modes = torch.where(eligible.any(-1), modes, 0)
    kind[torch.arange(n, device=eligible.device), person] = modes
    return kind


def validate(scene):
    bridge = scene['bridge']
    if bridge['stage_size_by_people'] != [[5.,6.], [5.5,6.5], [6.5,7.5], [7.,8.]]:
        raise ValueError('Compact Bridge requires the agreed per-person platform sizes')
    if bridge['clearance'] != .05 or scene['cliff']['size_range'] != [7.,8.]:
        raise ValueError('Compact geometry requires 5cm end seams and 7-8m Cliff')
    spec = bridge['distance_curriculum']
    expected = dict(warmup_updates=500, anneal_updates=2000, mixture=[.2,.6,.2])
    if bridge.get('layout_cache') is not None and spec.get('mixture') == [0.,0.,1.]:
        expected['mixture'] = [0.,0.,1.]
    if spec != expected:
        raise ValueError('Unsupported compact distance schedule')
    if bridge['boundary_rsi'] != dict(probabilities=[.2,.3,.25,.25], edge_distance=[.35,.65], attempts=32):
        raise ValueError('Unsupported boundary RSI settings')

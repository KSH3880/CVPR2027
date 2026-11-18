import torch

from utils import stage2_spec
from utils.before_spec import sample_roles
from utils.sensor_stage2_spec import paired_people


NAME = 'stage2_sensor_900_warehouse_safety'
BASE = 'stage2_sensor_900_warehouse_bridge_init_rsi'


def cliff_roles(sizes, probabilities, all_edge_probability):
    n, people = sizes.shape[:2]
    groups = torch.arange(people, device=sizes.device)[None].repeat(n, 1)
    pre = torch.full_like(groups, -1)
    weights = sizes.new_tensor(list(probabilities.values())[1:5])
    tasks = torch.multinomial(weights, n*people, True).reshape(n, people)+1
    rows = (torch.rand(n, device=sizes.device) >= all_edge_probability).nonzero().flatten()
    pairs = torch.rand(len(rows), people, device=sizes.device).argsort(-1)[:, :2]
    groups[rows[:, None], pairs] = pairs.min(-1).values[:, None]
    tasks[rows[:, None], pairs] = 5
    return groups, pre, tasks


def room_roles(sizes, ranges, probabilities, spec, preset):
    domain, groups, pre, tasks = stage2_spec.sample_episode(sizes, ranges, probabilities,
        preset, spec['scene_weights'], num_objects=8)
    weights = list(probabilities.values())[1:5]
    rows = (domain == 0).nonzero().flatten()
    if len(rows):
        selected = preset if preset.startswith('before_') else 'random_scenario'
        p, t = sample_roles(sizes[rows], ranges, selected,
            solo_probabilities=weights, pairing_probability=spec['before_pairing_probability'])
        groups[rows] = torch.where(p >= 0, p, torch.arange(tasks.shape[1], device=tasks.device))
        pre[rows], tasks[rows] = p, t
    if not spec['independent_push']:
        members = (groups[:, :, None] == groups[:, None, :]).sum(-1)
        solo = (tasks == 5) & (members == 1) & ~paired_people(pre)
        if solo.any():
            tasks[solo] = torch.multinomial(sizes.new_tensor(weights), int(solo.sum()), True)+1
    return domain, groups, pre, tasks


def resume_compatible(saved, current):
    from utils.coupled_push_spec import checkpoint_contract
    from utils.experiments import native_config
    return saved == checkpoint_contract(native_config(BASE)['env']) and current == checkpoint_contract(native_config(NAME)['env'])

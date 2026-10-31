import torch

from push_reference.utils.unified_training import EXPERT_PROBS, append_family


def sample_demo(task, count):
    packet = task.relation_runtime.suffix().reshape(-1, 4, 5)
    families = torch.where(packet[..., 1] == 0, 1, torch.where(packet[..., 1] == 1, 2, 0))
    probabilities = torch.bincount(families.flatten(), minlength=3).float().clamp_min(1)
    family = torch.multinomial(probabilities, count, replacement=True)
    skills = torch.multinomial(torch.tensor(EXPERT_PROBS, device=task.device)[family], 1).flatten()
    sources = torch.multinomial(torch.tensor([.8, .1, .1], device=task.device), count, replacement=True)
    sources = torch.where(family == 0, sources, 0)
    task._last_amp_source_counts = torch.bincount(sources, minlength=3)
    result = torch.empty(count, task.get_num_amp_obs(), device=task.device)
    for source in range(3):
        names = enumerate(task.cfg["env"]["skill"]) if source == 0 else [(-1, ('backward', 'sideways')[source-1])]
        for skill_id, name in names:
            mask = (sources == source) & ((skills == skill_id) if source == 0 else True)
            rows = mask.nonzero().flatten()
            if not len(rows):
                continue
            lib = task._motion_lib[name] if source == 0 else task._stage2_amp_libs[name]
            clips = lib.sample_motions(len(rows))
            truncate = task.dt*(task._num_amp_obs_steps-1)
            times = lib.sample_time(clips, truncate_time=truncate)+truncate
            obs = task.build_amp_obs_demo(clips, times, lib).reshape(len(rows), task._num_amp_obs_steps, -1)
            result[rows] = append_family(obs, family[rows, None].expand(-1, task._num_amp_obs_steps)).flatten(1)
    return result

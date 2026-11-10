import torch

from utils import stage2_spec


def prepare_intersection_bank(sizes, masses, env, preset):
    people = env['numAgents']
    ranges = env['box']['build']['taskSizeRanges']
    episode = stage2_spec.sample_episode(sizes[:, :people], ranges,
        env['relationGraph']['template_probabilities'], preset,
        env['stage2']['scene_weights'], num_objects=env['numObjects'])
    _, _, predecessor, templates = episode
    rows, follower = (predecessor >= 0).nonzero(as_tuple=True)
    leader = predecessor[rows, follower]
    for template, family in ((1, 'SIT'), (2, 'CLIMB'), (4, 'SUPPORT')):
        selected = templates[rows, follower] == template
        es, owners = rows[selected], leader[selected]
        density = masses[es, owners]/sizes[es, owners].prod(-1)
        for axis, ((low, high), (a, b)) in enumerate(zip(ranges['HOLDING_AT'], ranges[family])):
            sizes[es, owners, axis] = torch.randint(round(max(low, a)*20),
                round(min(high, b)*20)+1, (len(es),), device=sizes.device)/20
        masses[es, owners] = density*sizes[es, owners].prod(-1)
    return episode

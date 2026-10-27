from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib
import torch

from utils.size_rsi import STAGE1_V2_RANGES, TASK_PROBS, task_probabilities
from utils.edge_scenario_spec import CLIMB_TEMPLATES, compose_canonical_graph
from utils.edge_ontop_spec import select_graph

SAMPLER = 'stage1_jointpush'
PUSH = 12
TEMPLATES = CLIMB_TEMPLATES + ('PUSH_AT',)


def box_profiles():
    return [[x/20, y/20, z/20] for x in range(32, 45, 2)
            for y in range(32, 45, 2) for z in range(29, 35)
            if 10*min(x, y) >= 11*z]


def validate_env(env):
    from utils.unified_training import validate_unified_env, SKILLS
    from utils.task_role_spec import TASK_PROBS as legacy_probs
    if (env['numAgents'], env['numObjects']) != (2, 4):
        raise ValueError('stage1_jointpush requires two people and four boxes')
    probs = env['relationGraph']['template_probabilities']
    if tuple(probs) != TEMPLATES or list(probs.values()) != [0., .1, .2, .25, .2, .25]:
        raise ValueError('stage1_jointpush task proportions must be 10/20/25/20/25')
    if tuple(env['skill']) != SKILLS + ('push',):
        raise ValueError('stage1_jointpush needs the original skills plus push')
    if env['jointPush']['box'] != dict(mass=20., friction=.4, profiles=box_profiles()):
        raise ValueError('stage1_jointpush requires the KOO stable cuboid physics')
    scene = env['jointPush']['scene']
    if (scene['source_angle_degrees'] != 360. or scene['goal_angle_degrees'] != 360.
            or scene['source_distance'] != [2., 10.] or scene['goal_distance'] != [2., 10.]
            or env['eval'].get('sampling') != 'training_loco_only'
            or env['eval']['skill'] != ['loco'] or env['eval']['skillInitProb'] != [1.]):
        raise ValueError('stage1_jointpush requires 2-10m/360-degree sampling and loco-only evaluation')
    original = deepcopy(env)
    del original['jointPush']
    original['relationGraph']['sampler'] = 'two_agent_four_object_independent_unified'
    original['relationGraph']['template_probabilities'] = dict(zip(CLIMB_TEMPLATES, legacy_probs))
    original['skill'] = original['skill'][:-1]
    original['skillInitProb'] = original['skillInitProb'][:-1]
    original['skillDiscProb'] = original['skillDiscProb'][:-1]
    original['templateRsi'] = {k: v[:-1] for k, v in original['templateRsi'].items() if k != 'PUSH_AT'}
    validate_unified_env(original)


def compose_graph(templates, shuffle=False):
    graph = compose_canonical_graph(torch.where(templates == 5, 3, templates), shuffle, 4)
    push = (templates == 5).gather(1, graph.edge_owner)
    return replace(graph, edge_relation=torch.where(push & (graph.edge_relation == 6), PUSH, graph.edge_relation))


def compile_graph(spec, m, o, device=None):
    return select_graph(compose_graph(torch.full((1, m), 3, device=device, dtype=torch.long)), 0)


def sample_assets(n, env, device, preset='random_scenario'):
    probs = torch.tensor(list(env['relationGraph']['template_probabilities'].values()), device=device)
    if preset != 'random_scenario':
        index = {'sit': 1, 'climb': 2, 'holding_at': 3, 'holding_ontop': 4, 'push': 5}[preset]
        probs.zero_()
        probs[index] = 1
    seed = torch.multinomial(probs, n*2, replacement=True).reshape(n, 2)
    sizes = torch.empty(n, 4, 3, device=device)
    profiles = torch.tensor(env['jointPush']['box']['profiles'], device=device)
    for task, name in enumerate(TEMPLATES):
        rows, owners = (seed == task).nonzero(as_tuple=True)
        if task == 5:
            sizes[rows, owners] = profiles[torch.randint(len(profiles), (len(rows),), device=device)]
        else:
            for axis, (lo, hi) in enumerate(STAGE1_V2_RANGES[name]):
                sizes[rows, owners, axis] = torch.randint(round(lo*20), round(hi*20)+1, (len(rows),), device=device)/20
    for axis, (lo, hi) in enumerate(STAGE1_V2_RANGES['SUPPORT']):
        sizes[:, 2:, axis] = torch.randint(round(lo*20), round(hi*20)+1, (n, 2), device=device)/20
    push = seed == 5
    posterior = torch.zeros(n, 2, 6, device=device)
    posterior[..., 5] = push
    if (~push).any():
        old = task_probabilities(sizes[:, :2][~push], STAGE1_V2_RANGES)
        old *= probs[:5]/old.new_tensor(TASK_PROBS)
        old /= old.sum(-1, keepdim=True)
        posterior[..., :5][~push] = old
    return sizes, posterior


def contract(env):
    return dict(revision=1, graph=env['relationGraph'], push=env['jointPush'],
                rsi=env['templateRsi'], amp=env['ampExpertMapping'],
                box=env['box'], horizon=env['episodeLength'],
                spawn_radius=env['agentSpawnRadius'], spacing=env['envSpacing'],
                bank_sha256=hashlib.sha256(Path(env['jointPush']['rsi']['bank_file']).read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256(Path(env['motion_file']).read_bytes()).hexdigest())

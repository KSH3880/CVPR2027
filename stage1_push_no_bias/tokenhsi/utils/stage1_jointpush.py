from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib
import torch

from utils.size_rsi import STAGE1_RANGES, TASK_PROBS, task_probabilities
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
    if env['numAgents'] < 1 or env['numObjects'] < env['numAgents']:
        raise ValueError('PUSH requires numAgents >= 1 and numObjects >= numAgents')
    probs = env['relationGraph']['template_probabilities']
    if 'adaptiveSampling' in env:
        from utils.stage1_push_adaptive import validate_spec
        validate_spec(env)
    elif 'samplingSchedule' in env:
        from utils.stage1_push_schedule import validate_schedule
        validate_schedule(env)
    elif tuple(probs) != TEMPLATES or list(probs.values()) not in (
            [0., .1, .15, .25, .2, .3], [0., .1, .2, .25, .2, .25], [0., .075, .1875, .24375, .24375, .25], [0., .05, .125, .1625, .1625, .5]):
        raise ValueError('stage1_jointpush requires a supported current or historical task mixture')
    if tuple(env['skill']) != SKILLS + ('push',):
        raise ValueError('stage1_jointpush needs the original skills plus push')
    physical = env['jointPush']['box']
    if physical not in (dict(mass=20., friction=.4, profiles=box_profiles()),
                        dict(density=3., friction=.4, profiles=box_profiles())):
        raise ValueError('stage1_jointpush requires 20kg or density 3, friction 0.4 and the 167 cuboid profiles')
    if 'density' in physical and env['jointPush']['rsi'].get('reference_mass') != 20.:
        raise ValueError('Density variant must declare reuse of the 20kg RSI bank')
    if 'at_progress_min_delta' in env['jointPush']:
        if not 0. < float(env['jointPush']['at_progress_min_delta']) < float('inf'):
            raise ValueError('PUSH AT progress gate requires a finite positive distance threshold')
    scene = env['jointPush']['scene']
    if (scene['source_angle_degrees'] != 360. or scene['goal_angle_degrees'] not in (90., 360.)
            or scene['source_distance'] != [2., 10.] or scene['goal_distance'] != [2., 10.]
            or env['eval'].get('sampling') != 'training_loco_only'
            or env['eval']['skill'] != ['loco'] or env['eval']['skillInitProb'] != [1.]):
        raise ValueError('stage1_jointpush requires 2-10m sampling, 360/90-degree goals and loco-only evaluation')
    original = deepcopy(env)
    del original['jointPush']
    original['relationGraph']['sampler'] = 'two_agent_four_object_independent_unified'
    original['relationGraph']['template_probabilities'] = dict(zip(CLIMB_TEMPLATES, legacy_probs))
    original['skill'] = original['skill'][:-1]
    original['skillInitProb'] = original['skillInitProb'][:-1]
    original['skillDiscProb'] = original['skillDiscProb'][:-1]
    original['templateRsi'] = {k: v[:-1] for k, v in original['templateRsi'].items() if k != 'PUSH_AT'}
    validate_unified_env(original)


def compose_graph(templates, shuffle=False, num_objects=None):
    graph = compose_canonical_graph(torch.where(templates == 5, 3, templates), shuffle, num_objects)
    push = (templates == 5).gather(1, graph.edge_owner)
    return replace(graph, edge_relation=torch.where(push & (graph.edge_relation == 6), PUSH, graph.edge_relation))


def compile_graph(spec, m, o, device=None):
    return select_graph(compose_graph(torch.full((1, m), 3, device=device, dtype=torch.long), num_objects=o), 0)


def sample_templates(weights, num_objects, prefer_ontop=False):
    """Draw tasks, redrawing excess ON_TOP owners to fit distinct spare supports."""
    n, m = weights.shape[:2]
    if prefer_ontop and num_objects < 2*m:
        templates = torch.full((n, m), 4, device=weights.device, dtype=torch.long)
    else:
        templates = torch.multinomial(weights.flatten(0, 1), 1).reshape(n, m)
    if num_objects < 2*m and (templates == 4).any():
        ontop = templates == 4
        priority = torch.rand(n, m, device=weights.device).masked_fill(~ontop, 2.)
        rank = priority.argsort(-1).argsort(-1)
        redraw = ontop & (rank >= num_objects-m)
        if redraw.any():
            alternatives = weights[redraw].clone()
            alternatives[:, 4] = 0
            if (alternatives.sum(-1) <= 0).any():
                raise ValueError('No size-compatible alternative for an excess ON_TOP task')
            templates[redraw] = torch.multinomial(alternatives, 1).flatten()
    return templates


def sample_assets(n, env, device, preset='random_scenario'):
    m, o = env['numAgents'], env['numObjects']
    prior = env['relationGraph']['template_probabilities']
    if 'adaptiveSampling' in env:
        from utils.stage1_push_adaptive import state_for_env, template_probabilities
        prior = template_probabilities(state_for_env(env))
    probs = torch.tensor([prior[t] for t in TEMPLATES], device=device)
    if preset != 'random_scenario':
        index = {'sit': 1, 'climb': 2, 'holding_at': 3, 'holding_ontop': 4, 'push': 5}[preset]
        probs.zero_()
        probs[index] = 1
    if 'adaptiveSampling' in env and preset == 'random_scenario':
        from utils.stage1_push_adaptive import slot_templates
        seed = slot_templates(dict(zip(TEMPLATES, probs.tolist())), n*m, device,
                              env['adaptiveSampling']['minimum_probability'],
                              strict=not env.get('_adaptiveEvaluation', False)).reshape(n, m)
    elif 'samplingSchedule' in env and preset == 'random_scenario' and n*m >= 100:
        from utils.stage1_push_schedule import balanced_templates
        seed = balanced_templates(probs, n*m).reshape(n, m)
    else:
        seed = torch.multinomial(probs, n*m, replacement=True).reshape(n, m)
    sizes = torch.empty(n, o, 3, device=device)
    profiles = torch.tensor(env['jointPush']['box']['profiles'], device=device)
    for task, name in enumerate(TEMPLATES):
        rows, owners = (seed == task).nonzero(as_tuple=True)
        if task == 5:
            sizes[rows, owners] = profiles[torch.randint(len(profiles), (len(rows),), device=device)]
        else:
            for axis, (lo, hi) in enumerate(STAGE1_RANGES[name]):
                sizes[rows, owners, axis] = torch.randint(round(lo*20), round(hi*20)+1, (len(rows),), device=device)/20
    for axis, (lo, hi) in enumerate(STAGE1_RANGES['SUPPORT']):
        sizes[:, m:, axis] = torch.randint(round(lo*20), round(hi*20)+1, (n, o-m), device=device)/20
    push = seed == 5
    if 'adaptiveSampling' in env and not (env.get('_adaptiveEvaluation', False) and o < 2*m):
        # Each physical slot owns its sampled task for this focus block. This
        # preserves every task's class-conditional size and guarantees its frame
        # budget instead of silently restricting PUSH to the original slots.
        return sizes, torch.nn.functional.one_hot(seed, 6).to(sizes.dtype)
    posterior = torch.zeros(n, m, 6, device=device)
    posterior[..., 5] = push
    if (~push).any():
        old = task_probabilities(sizes[:, :m][~push], STAGE1_RANGES)
        # An ON_TOP preset with too few supports needs eligible alternatives.
        posterior_prior = (old.new_tensor([prior[t] for t in TEMPLATES])
                           if preset == 'holding_ontop' and o < 2*m else probs)
        old *= posterior_prior[:5]/old.new_tensor(TASK_PROBS)
        old /= old.sum(-1, keepdim=True)
        posterior[..., :5][~push] = old
    return sizes, posterior


def box_densities(sizes, push, physical):
    m = push.shape[1]
    density = torch.full(sizes.shape[:-1], 100., device=sizes.device)
    rho = physical['density'] if 'density' in physical else physical['mass']/sizes[:, :m].prod(-1)
    density[:, :m] = torch.where(push, rho, density[:, :m])
    return density


def contract(env):
    result = dict(revision=1, graph=env['relationGraph'], push=env['jointPush'],
                rsi=env['templateRsi'], amp=env['ampExpertMapping'],
                box=env['box'], horizon=env['episodeLength'],
                spawn_radius=env['agentSpawnRadius'], spacing=env['envSpacing'],
                bank_sha256=hashlib.sha256(Path(env['jointPush']['rsi']['bank_file']).read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256(Path(env['motion_file']).read_bytes()).hexdigest())
    if 'samplingSchedule' in env:
        result.update(revision=2, sampling_schedule=deepcopy(env['samplingSchedule']))
    if 'adaptiveSampling' in env:
        result.update(revision=3, adaptive_sampling=deepcopy(env['adaptiveSampling']))
    return result

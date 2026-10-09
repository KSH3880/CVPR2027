import torch

from utils.edge_context_spec import AT
from utils.edge_ontop_spec import ON_TOP, permute_graph, select_graph
from utils.edge_interaction_spec import SIT, CLIMB
from utils.edge_scenario_spec import compose_canonical_graph


SAMPLER = 'two_agent_before_task_embedding'
PRESETS = ('random_scenario', 'before_climb', 'before_sit', 'before_stack',
           'independent', 'sit', 'climb', 'holding_at', 'holding_ontop')
SIZE_RANGE = [[.5, .6], [.5, .6], [.35, .45]]
TASK_PROBS = [0., .10, .25, .325, .325]
CONTRACT = dict(gate='current_at_success', progress='always',
                holding='always', saturation='gated_current_success',
                collision={'mode': 'cpa', 'ttc_discount': .99})


def validate_sampler(spec, m=2, o=4):
    expected = dict(mode='edge_composition', sampler=SAMPLER, semantic_only=True,
        policy_task_roles=True, max_edges_per_agent=2, edge_capacity=4,
        before_fraction=.8, family_probabilities={
            'before_climb': 1/3, 'before_sit': 1/3, 'before_stack': 1/3},
        independent_template_probabilities=TASK_PROBS, shuffle_edge_order=True,
        shuffle_token_order=True, dependency='carry_at_payload_to_target')
    if spec != expected or (m, o) != (2, 4):
        raise ValueError('BEFORE requires 2H/4O, 80/20 scenes and three equal families')


def environment_families(n, device, preset='random_scenario'):
    if preset not in PRESETS:
        raise ValueError('Unknown BEFORE preset: ' + preset)
    result = torch.zeros(n, dtype=torch.long, device=device)
    if preset == 'random_scenario':
        count = n - round(.2*n)
        result[:count] = torch.arange(count, device=device) % 3 + 1
        return result[torch.randperm(n, device=device)]
    if preset.startswith('before_'):
        result[:] = PRESETS.index(preset)
    return result


def sample_sizes(n, device):
    return (torch.randint(3, (n, 4, 3), device=device) +
            torch.tensor([10, 10, 7], device=device)) / 20.


def sample_graph(n, spec, device='cpu', preset='random_scenario', role_swap=False,
                 generator=None, families=None):
    validate_sampler(spec)
    if role_swap:
        raise ValueError('BEFORE uses token permutation, not a separate role-swap preset')
    if families is None:
        families = environment_families(n, device, preset)
    templates = torch.multinomial(torch.tensor(TASK_PROBS, device=device),
                                 2*n, replacement=True, generator=generator).reshape(n, 2)
    if preset in PRESETS[5:]:
        templates[:] = {'sit': 1, 'climb': 2, 'holding_at': 3, 'holding_ontop': 4}[preset]
    dependent = families > 0
    templates[dependent, 0] = 3
    templates[:, 1] = torch.where(dependent,
        torch.where(families == 1, 2, torch.where(families == 2, 1, 4)), templates[:, 1])
    graph = compose_canonical_graph(templates, shuffle=False)
    follower = graph.edge_valid & (graph.edge_owner == 1)
    target = follower & ((graph.edge_relation == SIT) | (graph.edge_relation == CLIMB)
                         | (graph.edge_relation == ON_TOP)) & dependent[:, None]
    graph.edge_dst[target] = 2
    predecessor = graph.edge_valid & (graph.edge_owner == 0) & (graph.edge_relation == AT)
    graph.prereq_mask[:] = target[:, :, None] & predecessor[:, None, :]
    if spec['shuffle_edge_order']:
        graph = permute_graph(graph, torch.rand(n, 4, device=device,
                              generator=generator).argsort(-1))
    return graph


def compile_graph(spec, m, o, device=None):
    validate_sampler(spec, m, o)
    return select_graph(sample_graph(1, spec, device or 'cpu', 'before_climb'), 0)


def validate_env(env):
    from utils.unified_training import SKILLS, EXPERT_PROBS
    from utils.size_rsi import SCREEN
    validate_sampler(env['relationGraph'], env['numAgents'], env['numObjects'])
    build = env['box']['build']
    if build.get('commonSizeRange') != SIZE_RANGE or build['sizeInterval'] != .05 or \
            build['randomDensity'] or env['box']['reset']['randomAssignment']:
        raise ValueError('BEFORE requires common 50..60/50..60/35..45cm assets')
    if env['skill'] != list(SKILLS) or not env.get('ampTaskConditioning') or \
            env['sizeAwareRsi'] != dict(cacheDirectory='output/rsi_cache',
                lateProbability=.7, lateFraction=.3, screen=SCREEN):
        raise ValueError('BEFORE requires Stage-1 task-conditioned AMP and size RSI')
    expected = {'SIT': [.5, .5, 0., 0., 0., 0., 0., 0.],
                'CLIMB': [.5, 0., .5, 0., 0., 0., 0., 0.],
                'HOLDING_AT': [.4, 0., 0., 0., 0., .1, .4, .1],
                'HOLDING_ON_TOP': [.4, 0., 0., 0., 0., .1, .4, .1]}
    if any(env['templateRsi'][key] != value for key, value in expected.items()):
        raise ValueError('BEFORE RSI must preserve Stage-1 skill probabilities')
    expected_amp = torch.tensor([.8*2/3 + .2*.65, .8/6 + .2*.1, .8/6 + .2*.25]) @ torch.tensor(EXPERT_PROBS)
    if not torch.allclose(torch.tensor(env['skillDiscProb']), expected_amp, atol=1e-7, rtol=0):
        raise ValueError('BEFORE AMP prior must match the 80/20 task mixture')


def family_probabilities(families, preset):
    independent = torch.tensor([.65, .1, .25], device=families.device)
    if preset in PRESETS[5:]:
        independent = torch.tensor({'sit': [0., 1., 0.], 'climb': [0., 0., 1.],
            'holding_at': [1., 0., 0.], 'holding_ontop': [1., 0., 0.]}[preset], device=families.device)
    ratios = torch.bincount(families, minlength=4).float() / len(families)
    return (ratios[0]*independent + ratios[1]*independent.new_tensor([.5, 0., .5])
            + ratios[2]*independent.new_tensor([.5, .5, 0.])
            + ratios[3]*independent.new_tensor([1., 0., 0.]))

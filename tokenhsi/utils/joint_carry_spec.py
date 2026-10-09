import torch

from utils.edge_context_spec import HOLDING, AT
from utils.edge_ontop_spec import ON_TOP, permute_graph, select_graph
from utils.edge_stage2_spec import _empty_graph


JOINT_SAMPLER = 'two_agent_joint_carry_task_embedding'
JOINT_PRESETS = ('random_scenario', 'joint_carry_at', 'joint_carry_ontop')


def validate_env(env):
    from utils.joint_carry_amp import validate_ablation
    validate_ablation(env)
    from utils.mixed_carry_spec import MIXED_SAMPLER, validate_sampler as validate_mixed
    mixed = env['relationGraph'].get('sampler') == MIXED_SAMPLER
    (validate_mixed if mixed else validate_sampler)(
        env['relationGraph'], env['numAgents'], env['numObjects'])
    if mixed:
        from utils.size_rsi import SCREEN
        if env.get('sizeAwareRsi') != dict(cacheDirectory='output/rsi_cache',
                lateProbability=.7, lateFraction=.3, screen=SCREEN) or \
                env.get('independentCarryRsi') != [.4, 0., 0., 0., 0., .1, .4, .1] or \
                env['relationReward']['joint_carry'].get('independent') != {
                    'environment_fraction': .2, 'placement_gate': 'own_current_holding',
                    'rsi': 'stage1_size_aware'}:
            raise ValueError('Mixed carry requires Stage-1 size RSI and independent holding gate')
    elif 'independent' in env['relationReward']['joint_carry']:
        raise ValueError('Independent carry requires the mixed sampler')
    build, reset = env['box']['build'], env['box']['reset']
    if build['baseSize'] != [.52, .8, .4] or build['jointSupportSize'] != [.52, .8, .3] or \
            build['randomSize'] or build['randomDensity'] or reset['randomAssignment'] or \
            reset['randomHeight'] or (not mixed and 'sizeAwareRsi' in env):
        raise ValueError('Paired RSI requires fixed payload/support assets and canonical binding')
    if env.get('ampTaskConditioning') is not True or \
            env['jointCarryRsi']['ampHistory'] != 'repeat_reset_state':
        raise ValueError('Joint carry requires carry-conditioned AMP and matching reset history')
    expected = [1/3, 0., 0., 0., 1/3, 1/6, 0., 1/6]
    if env['skill'] != ['loco', 'sit', 'climb', 'climbNoRSI', 'omomo', 'pickUp', 'carryWith', 'putDown'] or \
            env['skillDiscProb'] != expected or \
            env['templateRsi']['HOLDING_AT'] != [.4, 0., 0., 0., 0., .1, .4, .1] or \
            env['templateRsi']['HOLDING_ON_TOP'] != [.5, 0., 0., 0., 0., .1, .4, 0.]:
        raise ValueError('Joint carry RSI/AMP skill probabilities do not match the experiment')


def validate_sampler(spec, m=2, o=4):
    expected = dict(mode='edge_composition', sampler=JOINT_SAMPLER,
        semantic_only=True, policy_task_roles=True, max_edges_per_agent=2,
        edge_capacity=4, family_probabilities={'joint_carry_at': .5, 'joint_carry_ontop': .5},
        shuffle_edge_order=True, shuffle_token_order=True,
        coupling='same_carry_task_payload_target')
    if spec != expected or (m, o) != (2, 4):
        raise ValueError('Joint carry requires 2 humans/4 objects, paired AT/ON_TOP 50/50 and explicit coupling')


def sample_graph(n, spec, device='cpu', preset='random_scenario', role_swap=False,
                 generator=None):
    validate_sampler(spec)
    if preset not in JOINT_PRESETS:
        raise ValueError('Unknown joint-carry preset: ' + preset)
    ontop = (torch.rand(n, device=device, generator=generator) >= .5
             if preset == 'random_scenario' else
             torch.full((n,), preset == 'joint_carry_ontop', device=device, dtype=torch.bool))
    graph = _empty_graph(n, 2, 4, 4, device)
    # Payload O0, support O2, and shared G0 preserve the Stage-1 canonical layout.
    graph.edge_src[:] = torch.tensor([0, 2, 1, 2], device=device)
    target = torch.where(ontop, 4, 6)
    graph.edge_dst[:, 0::2] = 2
    graph.edge_dst[:, 1::2] = target[:, None]
    graph.edge_relation[:, 0::2] = HOLDING
    graph.edge_relation[:, 1::2] = torch.where(ontop, ON_TOP, AT)[:, None]
    graph.edge_owner[:] = torch.tensor([0, 0, 1, 1], device=device)
    graph.edge_valid[:] = True
    graph.required_goal[:, 1::2] = True
    if spec['shuffle_edge_order']:
        graph = permute_graph(graph, torch.rand(n, 4, device=device,
                                               generator=generator).argsort(-1))
    return graph


def compile_graph(spec, m, o, device=None):
    validate_sampler(spec, m, o)
    return select_graph(sample_graph(1, spec, device or 'cpu', 'joint_carry_at'), 0)

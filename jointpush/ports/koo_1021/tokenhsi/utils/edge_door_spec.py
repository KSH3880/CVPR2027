"""Independent OPEN bindings; existing Stage-1 semantic packet contract."""
import math
import torch
from utils.edge_context_spec import EdgeContextGraph

OPEN = 11
DOOR_SAMPLER = 'two_agent_two_door_open'
SINGLE_DOOR_SAMPLER = 'single_agent_door_open'
DOOR_VARIANT = 'stage1_door_open'

def compile_graph(spec, m, o, device=None):
    sampler = SINGLE_DOOR_SAMPLER if (m, o) == (1, 1) else DOOR_SAMPLER
    if (m, o) not in ((1, 1), (2, 2)) or spec != {
            'sampler': sampler, 'semantic_only': True,
            'bindings': list(range(m))}:
        raise ValueError('Door OPEN requires independent H0->Door0, H1->Door1')
    h = torch.arange(m, device=device)
    valid = torch.ones(m, dtype=torch.bool, device=device)
    return EdgeContextGraph(tuple('open_'+str(i) for i in range(m)), m, o, h, m+h,
        torch.full_like(h, OPEN), h, valid, valid.clone(),
        torch.zeros(m, m, dtype=torch.bool, device=device), torch.full_like(h, -1))

def validate_reward(c):
    from utils.edge_stage1_spec import STAGE1_CONTEXT_MODE
    if c.get('mode') != STAGE1_CONTEXT_MODE:
        raise ValueError('Door reuses Stage-1 mode')
    expected = {'mode', 'schema_version', 'stage1_variant', 'state_reward_weight',
        'progress_reward_weight', 'success_reward_weight', 'door', 'task_sharing',
        'success', 'observation', 'diagnostics'}
    if set(c) != expected or c['schema_version'] != 11 or c['stage1_variant'] != DOOR_VARIANT:
        raise ValueError('Invalid Door Stage-1 reward schema')
    if c['observation'] != {'graph_packet_fields': ['valid','src','dst','relation','owner']}:
        raise ValueError('Door uses the existing semantic packet')
    if c['task_sharing'] != {'self': .9, 'teammate': .1}:
        raise ValueError('Door preserves Stage-1 task sharing')
    if c['success'] != {'saturation': 'current_success', 'terminate_when_all_subgoals_done': False}:
        raise ValueError('Door uses current success, without early success termination')
    if c['state_reward_weight'] not in (.2, .4):
        raise ValueError('Door state scale must be 0.2 (legacy) or 0.4 (HOLDING-matched)')
    for k in ('progress_reward_weight', 'success_reward_weight'):
        if c[k] != .2:
            raise ValueError('Door preserves 0.2 progress/success scales')
    d = c['door']
    if set(d) != {'target_angle', 'angle_tolerance', 'hand_distance_scale', 'approach_delta', 'approach_sigma'}:
        raise ValueError('Invalid door reward fields')
    for key,value in d.items():
        if isinstance(value, bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0 or (key!='angle_tolerance' and value==0):
            raise ValueError('Door scales and angles must be finite and positive')
    if not d['angle_tolerance'] < d['target_angle'] < 1.7:
        raise ValueError('Door target must lie within hinge limits')

def validate_env(env):
    compile_graph(env['relationGraph'], env['numAgents'], env['numObjects'])
    validate_reward(env['relationReward'])
    if env['policyObsMode'] != 'clean_scene' or env.get('ampTaskConditioning', False):
        raise ValueError('Door uses clean_scene and existing unconditioned AMP')
    if type(env.get('ampEnabled')) is not bool:
        raise ValueError('ampEnabled must be boolean')
    if env.get('templateRsi') or env.get('referenceRsi') or env['box']['reset']['randomHeight']:
        raise ValueError('Door reset does not use box reference RSI/platforms')
    if set(env['ampExpertMapping']) != {'OPEN'}:
        raise ValueError('Door requires an explicit OPEN expert mapping')
    row = env['ampExpertMapping']['OPEN']
    if len(row) != len(env['skill']) or any(not math.isfinite(v) or v < 0 for v in row) or abs(sum(row)-1)>1e-8:
        raise ValueError('Invalid OPEN expert mapping')
    direction = env['door'].get('openingDirection', 'push')
    hinge = env['door'].get('hinge')
    if hinge is not None:
        if not isinstance(hinge, dict) or set(hinge) != {'damping', 'friction'}:
            raise ValueError('Door hinge requires damping and friction')
        for value in hinge.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Door hinge values must be finite and nonnegative')
    if direction not in ('push', 'pull'):
        raise ValueError('Door openingDirection must be push or pull')
    if direction == 'pull' and (env.get('doorReference') or not env.get('doorApproachAMP')):
        raise ValueError('Pull Door currently requires approach-only loco RSI')
    lo,hi = env['door']['resetAngleRange']
    if not 0 <= lo <= hi < env['relationReward']['door']['target_angle']-env['relationReward']['door']['angle_tolerance']:
        raise ValueError('Door reset must be within limits and initially unsolved')
    reference=env.get('doorReference')
    if reference:
        expected={'interaction_skills','handle_height','height_tolerance','min_root_height',
            'max_root_speed','max_foot_height','min_forward_reach','late_fraction',
            'late_probability','stand_off'}
        if set(reference)!=expected or not set(reference['interaction_skills'])<=set(env['skill']):
            raise ValueError('Invalid Door reference fields/skills')
        for key in expected-{'interaction_skills'}:
            value=reference[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError('Invalid Door reference '+key)
        if reference['late_fraction']>1 or reference['late_probability']>1:
            raise ValueError('Invalid Door reference sampling probability')
        probabilities=env['skillInitProb']
        if len(probabilities)!=len(env['skill']) or any(v<0 for v in probabilities) or abs(sum(probabilities)-1)>1e-8:
            raise ValueError('Invalid Door RSI skill mixture')
    elif env['skillInitProb'][env['skill'].index('loco')] != 1:
        raise ValueError('Door starts from existing loco reference, without box RSI')

    approach=env.get('doorApproachAMP')
    if approach:
        from utils.door_approach_amp import validate
        validate(approach)
        if reference or not env['ampEnabled']:
            raise ValueError('Approach-only AMP requires loco-only expert and RSI')
        if approach.get('gate_enabled', True) or direction == 'pull':
            if env['skill']!=['loco'] or env['skillInitProb']!=[1.] or row!=[1.] or env['skillDiscProb']!=[1.]:
                raise ValueError('Approach-only AMP requires loco-only expert and RSI')
        else:
            # Full AMP can use real door experts while keeping the existing
            # locomotion-only reset. Gated/pull configurations keep their contract.
            skills=env['skill']
            expected_init=[1. if name=='loco' else 0. for name in skills]
            if len(set(skills))!=len(skills) or env['skillInitProb']!=expected_init or env['skillDiscProb']!=row:
                raise ValueError('Full Door AMP requires loco-only RSI and consistent expert probabilities')

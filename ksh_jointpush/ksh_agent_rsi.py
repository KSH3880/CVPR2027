"""Optional independent per-agent PUSH/loco RSI; the frozen snapshot stays intact."""
from copy import deepcopy
import torch
from utils.jointpush_geometry import rotate

MODE = 'independent_agents_v1'


def sample_agent_initialization(groups, templates, probability=.5):
    if not 0 <= probability <= 1:
        raise ValueError('PUSH RSI probability must be in [0,1]')
    return (torch.rand(groups.shape, device=groups.device) < probability) & (templates == 5)


def mixed_body_overlap(body_states, geometry, groups, interaction):
    body = body_states[:, :, geometry['ids']]
    points = body[..., :3] + rotate(body[..., 3:7], geometry['offsets'].expand(*body.shape[:-1], 3))
    radii = geometry['radii']
    bad = torch.zeros(len(groups), device=groups.device, dtype=torch.bool)
    for a in range(groups.shape[1]):
        for b in range(a):
            selected = (groups[:, a] == groups[:, b]) & (interaction[:, a] | interaction[:, b])
            rows = selected.nonzero().flatten()
            if not len(rows):
                continue
            distance = (points[rows, a, :, None] - points[rows, b, None]).square().sum(-1)
            bad[rows] |= (distance < (radii[:, None] + radii[None] + .005).square()).any((1, 2))
    return bad


def install(env):
    spec = env.get('stage2JointPush', {})
    mode = spec.get('coupled_rsi_selection', 'group')
    if mode == 'group':
        return  # Keep original group sampler, overlap tests and checkpoint contract.
    if mode != MODE:
        raise ValueError('Unknown PUSH RSI selection: ' + str(mode))
    from utils import stage2_jointpush
    from env.tasks.multi_agent import humanoid_ma_stage2_jointpush_rsi as task_module
    original_validate = stage2_jointpush.validate_env

    def validate(config):
        if config['stage2JointPush']['coupled_rsi_selection'] != MODE:
            raise ValueError('Independent RSI adapter requires its own configuration')
        original = deepcopy(config)
        original['stage2JointPush']['coupled_rsi_selection'] = 'group'
        return original_validate(original)

    stage2_jointpush.validate_env = validate
    task_module.sample_push_initialization = sample_agent_initialization
    task_module.coupled_body_overlap = mixed_body_overlap
    print('KSH_PUSH_RSI_SELECTION', dict(mode=mode,
          push_probability=env['jointPush']['rsi']['interaction_probability']), flush=True)

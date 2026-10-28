"""Group-balanced PUSH RSI: all-push / nontrivial mixture / all-loco, one third each."""
from copy import deepcopy
import torch
from jointpush.ksh_agent_rsi import mixed_body_overlap

MODE = 'balanced_group_ternary_v1'
PROBABILITIES = dict(all_push=1/3, mixed=1/3, all_loco=1/3)


def sample_balanced_initialization(groups, templates, probability=.5):
    if probability != .5:
        raise ValueError('Balanced RSI uses push/loco .5 for single-agent groups')
    count, people = groups.shape
    same = groups[:, :, None] == groups[:, None, :]
    size = same.sum(-1)
    # Group IDs identify leaders; every member reads the same categorical draw.
    category = torch.randint(3, groups.shape, device=groups.device).gather(1, groups)
    draws = torch.rand(groups.shape, device=groups.device).gather(1, groups)
    # Uniformly draw one of all nonempty, non-full subsets of this group.
    pattern = 1 + (draws * (2**size - 2).clamp_min(1)).long()
    earlier = torch.arange(people, device=groups.device)[None, :] < torch.arange(people, device=groups.device)[:, None]
    rank = (same & earlier).sum(-1)
    mixed = ((pattern >> rank) & 1).bool()
    result = (category == 0) | ((category == 1) & mixed)
    solo = torch.rand(groups.shape, device=groups.device) < .5
    return torch.where(size == 1, solo, result) & (templates == 5)


def install(env):
    spec = env['stage2JointPush']
    if spec['coupled_rsi_selection'] != MODE:
        raise ValueError('Balanced RSI requires its own configuration')
    if spec.get('rsi_group_state_probabilities') != PROBABILITIES or spec.get('rsi_solo_push_probability') != .5:
        raise ValueError('Balanced RSI requires equal thirds and solo push/loco .5')
    from utils import stage2_jointpush
    from env.tasks.multi_agent import humanoid_ma_stage2_jointpush_rsi as task_module
    original_validate = stage2_jointpush.validate_env
    def validate(config):
        if config['stage2JointPush']['coupled_rsi_selection'] != MODE:
            raise ValueError('Balanced RSI adapter requires matching configuration')
        original = deepcopy(config)
        original['stage2JointPush']['coupled_rsi_selection'] = 'group'
        return original_validate(original)
    stage2_jointpush.validate_env = validate
    task_module.sample_push_initialization = sample_balanced_initialization
    task_module.coupled_body_overlap = mixed_body_overlap
    print('KSH_PUSH_RSI_SELECTION', dict(mode=MODE, group_probabilities=PROBABILITIES,
          solo_push_probability=.5), flush=True)

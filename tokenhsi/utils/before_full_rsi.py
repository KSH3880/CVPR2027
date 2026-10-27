"""Stage1 RSI contract shared by solo and BEFORE roles."""
SKILLS = ('loco', 'sit', 'climb', 'climbNoRSI', 'omomo', 'pickUp', 'carryWith', 'putDown')
TEMPLATE_RSI = {
    'HOLDING': [.5, 0., 0., 0., 0., .5, 0., 0.],
    'SIT': [.5, .5, 0., 0., 0., 0., 0., 0.],
    'CLIMB': [.5, 0., .5, 0., 0., 0., 0., 0.],
    'HOLDING_AT': [.5, 0., 0., 0., 0., .1, .3, .1],
    'HOLDING_ON_TOP': [.5, 0., 0., 0., 0., .1, .3, .1],
}
SPEC = dict(version=1, frames='stage1', placement_retry_limit=8,
            fallback='stage1_available', evaluation_default='loco')


def full_rsi(env):
    return env.get('before', {}).get('full_rsi') is not None


def evaluation_weights(weights, mode, predecessor, templates, rows, owners):
    """Evaluation-only starts; both_carry targets ON_TOP pairs, others use loco."""
    result = weights.clone()
    if mode == 'mix':
        return result
    result.zero_()
    result[:, 0] = 1.
    if mode == 'both_carry':
        followers = (predecessor[rows] >= 0) & (templates[rows] == 4)
        leader = ((predecessor[rows] == owners[:, None]) & followers).any(-1)
        follower = followers.gather(1, owners[:, None]).flatten()
        selected = leader | follower
        result[selected, 0] = 0.
        result[selected, SKILLS.index('carryWith')] = 1.
    elif mode != 'loco':
        raise ValueError('Unknown full RSI evaluation mode: ' + mode)
    return result

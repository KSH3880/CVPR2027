"""Stage1 fixed-platform physics and saved env/sim restoration."""
from copy import deepcopy
from pathlib import Path


MODE = 'fixed_warmup_v1'
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent/stage1.yaml'


def uses_fixed_platforms(env):
    reset = env['box']['reset']
    mode = reset.get('platformPhysics')
    if mode is None:
        return False
    ground = (env.get('before') is not None and reset.get('platformMode') == 'ground_only'
              and reset.get('randomHeight') is False and reset.get('randomHeightProb') == 0.)
    if (mode != MODE or env.get('relationReward', {}).get('stage1_variant') != 'stage1'
            or (not ground and (reset.get('platformMode') != 'carry_payload_tokenhsi'
                                or reset.get('randomHeight') is not True))):
        raise ValueError('fixed_warmup_v1 platforms require the Stage1 carry platform config')
    return True


def resolve_config(source, native):
    """Keep the saved environment, explicitly opt into fixed shelves, retain sim.

    Old v2 snapshots contain no sim section. Only that missing section is taken
    from this experiment's native YAML; a partially specified sim is rejected.
    """
    from utils.checkpoint_compat import environment
    env = environment(source['env'])
    if env.get('relationReward', {}).get('stage1_variant') != 'stage1':
        raise ValueError('Stage1 requires its saved reward/physics contract')
    previous = env['box']['reset'].get('platformPhysics')
    if previous not in (None, MODE):
        raise ValueError('Unknown saved platform physics: ' + str(previous))
    env['box']['reset']['platformPhysics'] = MODE
    uses_fixed_platforms(env)
    sim = deepcopy(source['sim'] if 'sim' in source else native['sim'])
    # A default PhysX rest offset of 1mm leaves a 2mm box/shelf gap, exceeding
    # the AT success plane tolerance. Preserve complete saved physics settings.
    if (not isinstance(sim, dict) or 'substeps' not in sim
            or not isinstance(sim.get('physx'), dict)
            or set(native['sim']) - set(sim)
            or set(native['sim']['physx']) - set(sim.get('physx', {}))
            or sim.get('physx', {}).get('rest_offset') != 0.0):
        raise ValueError('Stage1 needs complete sim settings with physx.rest_offset=0')
    return {'env': env, 'sim': sim}

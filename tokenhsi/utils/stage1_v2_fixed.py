"""Opt-in fixed shelves and complete env/sim config for the v2 experiment."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import tempfile

import yaml

MODE = 'fixed_warmup_v1'
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'data/cfg/multi_agent/stage1_v2_fixed.yaml'


def uses_fixed_platforms(env):
    reset = env['box']['reset']
    mode = reset.get('platformPhysics')
    if mode is None:
        return False
    ground = (env.get('before') is not None and reset.get('platformMode') == 'ground_only'
              and reset.get('randomHeight') is False and reset.get('randomHeightProb') == 0.)
    if (mode != MODE or env.get('relationReward', {}).get('stage1_variant') != 'stage1_v2'
            or (not ground and (reset.get('platformMode') != 'carry_payload_tokenhsi'
                                or reset.get('randomHeight') is not True))):
        raise ValueError('fixed_warmup_v1 platforms require the stage1_v2 carry platform config')
    return True


def resolve_config(source, native):
    """Keep the saved environment, explicitly opt into fixed shelves, retain sim.

    Old v2 snapshots contain no sim section. Only that missing section is taken
    from this experiment's native YAML; a partially specified sim is rejected.
    """
    env = deepcopy(source['env'])
    if env.get('relationReward', {}).get('stage1_variant') != 'stage1_v2':
        raise ValueError('stage1_v2_fixed requires a v2 config/checkpoint; stage1 is separate')
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
        raise ValueError('Fixed v2 needs complete sim settings with physx.rest_offset=0')
    return {'env': env, 'sim': sim}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if args.config:
        source_path = Path(args.config)
    elif args.checkpoint:
        source_path = Path(args.checkpoint).resolve().parent.parent / 'relation_config.yaml'
    else:
        source_path = DEFAULT_CONFIG
    if not source_path.is_file():
        parser.error('Saved config missing: {}. Set STAGE1_V2_FIXED_ENV_CONFIG explicitly.'.format(source_path))
    source = yaml.safe_load(source_path.read_text())
    native = yaml.safe_load(DEFAULT_CONFIG.read_text())
    resolved = resolve_config(source, native)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Retain the exact resolved input for inspection; concurrent runs get distinct files.
    with tempfile.NamedTemporaryFile(mode='w', prefix='fixed_env_', suffix='.yaml',
                                     dir=output_dir, delete=False) as stream:
        yaml.safe_dump(resolved, stream, sort_keys=False)
        path = stream.name
    sim_source = source_path if 'sim' in source else DEFAULT_CONFIG
    print('[stage1_v2_fixed] Env: {}; sim: {}; physics: {}'.format(
        source_path, sim_source, MODE), file=sys.stderr)
    print(path)


if __name__ == '__main__':
    main()

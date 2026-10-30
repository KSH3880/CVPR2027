"""The supported experiments and one configuration loading path."""
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
FIXED = 'stage1'
NEAR = 'stage2_before_nonemask_rsi433_near50'
MIX = 'stage2_before'
MIX_RSI = 'stage2_before_rsi433'
PUSH = "stage1_push"
BEFORE_PUSH = "stage2_before_push"
COUPLED = "stage2_coupled_push"
BEFORE_PUSH_NEAR50="stage2_before_push_near50"
COUPLED_REGION_GOALS = "stage2_coupled_push_region_goals"
NAMES = (FIXED, NEAR, MIX, MIX_RSI, PUSH, BEFORE_PUSH, COUPLED, BEFORE_PUSH_NEAR50, COUPLED_REGION_GOALS)


def experiment_horizon(name):
    return 1200 if name.startswith("stage2_") else 900
BEFORE_NAMES = (NEAR, MIX, MIX_RSI)


def native_config(name, training=False):
    if name not in NAMES:
        raise ValueError('Unsupported experiment: ' + name)
    folder = 'train/rlg' if training else 'multi_agent'
    return yaml.safe_load((ROOT / f'tokenhsi/data/cfg/{folder}/{name}.yaml').read_text())


def load_environment(name, config_path, checkpoint=None, saved=False):
    """Saved env/sim is authoritative for resume/evaluation; explicit input wins."""
    native = native_config(name)
    if config_path:
        path = Path(config_path)
    elif saved:
        if not checkpoint or checkpoint == 'Base':
            raise ValueError('Evaluation/resume requires a checkpoint')
        parent = Path(checkpoint).resolve().parent
        path = (parent.parent if parent.name == 'nn' else parent) / 'relation_config.yaml'
    else:
        path = ROOT / f'tokenhsi/data/cfg/multi_agent/{name}.yaml'
    embedded = None
    if saved and not config_path and not path.is_file() and Path(checkpoint).is_file():
        import torch
        embedded = torch.load(checkpoint, map_location='cpu', weights_only=False).get('relation_experiment_config')
        if not embedded:
            raise ValueError('Checkpoint has neither sidecar nor embedded environment')
    if not path.is_file() and embedded is None:
        raise ValueError(f'Environment config missing: {path}; set ENV_CONFIG explicitly')
    source = deepcopy(embedded) if embedded is not None else yaml.safe_load(path.read_text())
    if embedded is not None:
        path = Path(checkpoint).resolve()
    from utils.checkpoint_compat import environment
    source['env'] = environment(source['env'])
    if name in (FIXED, PUSH):
        from utils.stage1_config import resolve_config
        cfg = resolve_config(source, native)
        cfg["experiment"] = deepcopy(source.get("experiment", {}))
        cfg["experiment"]["name"] = name
    else:
        if 'sim' not in source:
            raise ValueError('BEFORE snapshots require saved sim settings')
        cfg = {'env': deepcopy(source['env']), 'sim': deepcopy(source['sim'])}
        cfg['experiment'] = deepcopy(source.get('experiment', native['experiment']))
        cfg['experiment']['name'] = name
        from utils.checkpoint_compat import checkpoint_path
        transfer = cfg['experiment'].get('transfer', {})
        if 'checkpoint' in transfer:
            transfer['checkpoint'] = checkpoint_path(transfer['checkpoint'])
    return cfg, path


def validate_experiment(name, env, train, evaluation=False):
    native = native_config(name)
    expected = native_config(name, training=True)['params']
    params = train['params']
    if params['config']['name'] != expected['config']['name']:
        raise ValueError('Training experiment name does not match config filename')
    if params['network']['name'] != expected['network']['name']:
        raise ValueError('Policy builder does not match experiment')
    if name == FIXED:
        if env.get('jointPush') is not None:
            raise ValueError('Ordinary Stage1 cannot load PUSH checkpoints')
        if env.get('before') is not None:
            raise ValueError('Fixed Stage1 cannot use a BEFORE environment')
    elif name == PUSH:
        if env.get('jointPush') is None or env.get('before') is not None:
            raise ValueError('PUSH policy requires its own saved environment')
        from utils.stage1_jointpush import validate_env
        validate_env(env)
    elif name in (COUPLED, COUPLED_REGION_GOALS):
        from utils.coupled_push_spec import checkpoint_contract,validate_env
        validate_env(env)
        if checkpoint_contract(env)!=checkpoint_contract(native['env']):
            raise ValueError('Coupled PUSH environment differs from selected experiment')
        if not evaluation and env['episodeLength']!=1200:
            raise ValueError('Stage2 training requires 1200 steps')
        params['config']['minibatch_size']=params['config'].get('scene_minibatch_size',4096)*env['numAgents']
    else:
        from utils.before_spec import checkpoint_contract
        if env.get('before') is None or checkpoint_contract(env) != checkpoint_contract(native['env']):
            raise ValueError('BEFORE environment does not match the selected experiment')
        if env['relationGraph'] != native['env']['relationGraph']:
            raise ValueError('BEFORE graph does not match the selected experiment')
        if not evaluation and env['episodeLength'] != experiment_horizon(name):
            raise ValueError('Stage2 training requires 1200 steps')
        params['config']['minibatch_size'] = params['config'].get('scene_minibatch_size', 4096) * env['numAgents']


def check_before_checkpoint(weights, env):
    from utils.before_spec import checkpoint_contract
    from utils.checkpoint_compat import before_contract
    if before_contract(weights.get('before_contract')) != checkpoint_contract(env):
        raise ValueError('BEFORE checkpoint contract differs')

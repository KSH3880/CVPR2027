"""Explicit locomotion-only viewer geometry, separate from the source contract."""
from copy import deepcopy
import math


def viewer_geometry(env, args, variables):
    raw = variables.get('TOKENHSI_VIEWER_PUSH_LENGTH_Y', '')
    if not raw:
        return env
    if not (args.test and args.eval) or args.headless or env['numEnvs'] > 16:
        raise ValueError('PUSH length override requires a GUI --test --eval with <=16 environments')
    if variables.get('TOKENHSI_GUI_GPU_MODE') != 'native':
        raise ValueError('PUSH length viewer requires native CUDA/Vulkan selection')
    if variables.get('TOKENHSI_EVAL_TRAIN_RSI') == '1' or args.eval_skills != 'loco':
        raise ValueError('Changed PUSH geometry supports locomotion starts only')
    if not env['relationGraph'].get('push_only') or not env['pushBox'].get('size_profiles'):
        raise ValueError('PUSH length viewer requires native variable-size PUSH')
    length = float(raw)
    if not math.isfinite(length) or not 1.5 < length <= 3.0:
        raise ValueError('Viewer PUSH local Y length must be >1.5 and <=3 metres')
    result = deepcopy(env)
    box = result['pushBox']
    box['size'][1] = length
    box['size_profiles'] = [[x, length, z] for x, _, z in box['size_profiles']]
    result['box']['build']['baseSize'][1] = length
    result['pushScene']['path_separation'] = max(
        result['pushScene']['path_separation'], math.hypot(*box['size'][:2]) + .3)
    return result

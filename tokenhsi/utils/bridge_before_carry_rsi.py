NAME = 'stage2_sensor_900_bridge_before_fixed_carry_rsi'
SPEC = dict(version=1, episode_probability=.5, boundary_probabilities=[.375, .3125, .3125],
            cache_directory='output/rsi_cache_bridge_carry_v1')
KINDS = ('entry_before', 'entry_after', 'exit_before')


def validate(env):
    scene = env['sensorScene']
    if scene['bridge'].get('carry_boundary_rsi') != SPEC:
        raise ValueError('Fixed BEFORE carry RSI requires its v1 specification')
    if scene.get('variant') != 'bridge_before_fixed_v1' or not scene['sensor'].get('exclude_own_payload'):
        raise ValueError('Fixed BEFORE carry RSI requires the fixed Bridge variant with hidebox')
    return SPEC

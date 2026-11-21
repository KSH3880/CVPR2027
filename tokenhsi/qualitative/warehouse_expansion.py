from copy import deepcopy


def configure(env, pushers, pairs, independent_at=0):
    if pushers not in (2, 3, 4) or pairs < 1 or independent_at < 0:
        raise ValueError('Use 2-4 pushers, at least one BEFORE pair, and nonnegative independent AT count')
    scene = env['sensorScene']
    if scene['mixture']['warehouse'] != 1.:
        raise ValueError('Warehouse expansion requires a warehouse-only checkpoint')
    warehouse = scene['warehouse']
    coop = warehouse['cooperative']
    if coop['direction_probabilities'] != [0., 0., 1., 0.]:
        raise ValueError('Warehouse expansion requires inward PUSH')
    extra_width = (pushers-2)*coop['box_size'][0]
    coop['room_size'][1] += extra_width
    coop['pusher_probabilities'] = [float(i == pushers) for i in range(1, max(3, pushers)+1)]
    coop['before_pairs_probability'] = 0.
    warehouse['worker_pairing_probability'] = 1.
    warehouse['viewer_side_shift'] = extra_width/2
    warehouse['viewer_independent_at'] = independent_at
    if pushers > 2 or pairs > 1 or independent_at:
        warehouse['layout_candidates'] = 128
        warehouse['human_candidates'] = 128
        scene['placement_tries'] = 64
    env['numAgents'] = pushers+2*pairs+independent_at
    env['numObjects'] = 2*env['numAgents']
    env['warehouseViewer'] = dict(pushers=pushers, before_pairs=pairs, independent_at=independent_at)


def install(pushers, pairs, independent_at=0):
    from utils import experiments, sensor_stage2_spec, coupled_push_spec
    load = experiments.load_environment
    validate = sensor_stage2_spec.validate_sensor
    contract = coupled_push_spec.checkpoint_contract
    original_scene = None

    def load_environment(*args, **kwargs):
        nonlocal original_scene
        cfg, path = load(*args, **kwargs)
        validate(cfg['env'])
        original_scene = deepcopy(cfg['env']['sensorScene'])
        configure(cfg['env'], pushers, pairs, independent_at)
        env = cfg['env']
        coop = env['sensorScene']['warehouse']['cooperative']
        print(f"[warehouse viewer] pushers={pushers} pairs={pairs} independent_at={independent_at} people={env['numAgents']} "
              f"tokens={env['numObjects']} room={coop['room_size']} "
              f"box_width={pushers*coop['box_size'][0]:.2f} "
              f"opening={pushers*coop['box_size'][0]+coop['frontal_opening_gap']:.2f}", flush=True)
        return cfg, path

    def validate_sensor(env):
        if 'warehouseViewer' not in env:
            return validate(env)
        base = deepcopy(env)
        base.update(numAgents=4, numObjects=8, sensorScene=original_scene)
        validate(base)
        people = pushers+2*pairs+independent_at
        if (env['numAgents'], env['numObjects']) != (people, 2*people):
            raise ValueError('Viewer people/tokens must match PUSH, BEFORE and independent AT counts')

    def checkpoint_contract(env, inference=False):
        if 'warehouseViewer' in env:
            env = dict(env, sensorScene=original_scene)
        return contract(env, inference=inference)

    experiments.load_environment = load_environment
    sensor_stage2_spec.validate_sensor = validate_sensor
    coupled_push_spec.checkpoint_contract = checkpoint_contract


def command_counts(command, pushers, pairs, independent_at=0):
    command = list(command)
    people = pushers+2*pairs+independent_at
    for flag, value in (('--num_agents', people), ('--num_objects', 2*people)):
        if flag in command:
            command[command.index(flag)+1] = str(value)
        else:
            command.extend((flag, str(value)))
    if people > 4 and '--group_spawn_over4' not in command:
        command.append('--group_spawn_over4')
    return command

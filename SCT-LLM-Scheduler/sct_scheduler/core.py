"""Input boundary, strict output validation and joint difference constraints."""
import json
import math
from decimal import Decimal
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seconds(value, centiseconds=False):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'Time must be finite, nonnegative seconds')
    if centiseconds:
        scaled = Decimal(str(value)) * 100
        require(scaled == scaled.to_integral_value(), 'Output times need at most two decimals')
        return int(scaled)
    return value


def unique(rows, field):
    values = [r[field] for r in rows]
    require(all(isinstance(v, str) and v for v in values), f'Invalid {field}')
    require(len(values) == len(set(values)), f'Duplicate {field}')
    return values


def scene_index(scene):
    agents = unique(scene['initial_observation']['agents'], 'agent_id')
    require(bool(agents), 'Scene needs agents')
    tasks = scene['tasks']
    unique(tasks, 'task_id')
    owners = dict(scene.get('event_agents', {}))
    definitions = scene['event_definitions']
    for task in tasks:
        require(task['agent_id'] in agents, 'Unknown task agent')
        require(bool(task.get('description')), 'Plain-language task description required')
        for key, event in task.items():
            if key.endswith('_event_id'):
                require(event not in owners or owners[event] == task['agent_id'], 'Ambiguous event owner')
                owners[event] = task['agent_id']
    require(set(owners) == set(definitions),
            'Every event needs one owner: task *_event_id or event_agents')
    require(all(a in agents for a in owners.values()), 'Unknown event agent')
    require(all(isinstance(v, str) and v.strip() for v in definitions.values()), 'Exact event definitions required')
    required = set()
    for c in scene['coordination_constraints']:
        if c['type'] == 'align_events':
            events = c['event_ids']
            require(len(events) >= 2 and len(set(events)) == len(events), 'Invalid align group')
        elif c['type'] == 'enabling':
            events = [c['enabling_event_id'], c['dependent_event_id']]
        else:
            raise ValueError('Unsupported constraint type: ' + str(c['type']))
        require(all(e in owners for e in events), 'Constraint references unknown event')
        required.update(events)
    return agents, owners, required


def validate_request(request):
    require(set(request) == {'environment_spec', 'demonstrations', 'current_scene'}, 'Invalid request keys')
    # Reject NaN/Infinity anywhere, including observations and demonstration samples.
    json.dumps(request, allow_nan=False)
    env = request['environment_spec']
    require(env['units'] == {'position': 'meters', 'time': 'seconds', 'angle': 'radians'}, 'Unsupported units')
    require(bool(env.get('coordinate_system')) and bool(env.get('trajectory_format')), 'Missing field definitions')
    scene = request['current_scene']
    allowed = {'scene_id', 'controller_id', 'coordination_type', 'initial_observation', 'tasks',
               'event_definitions', 'event_agents', 'coordination_constraints', 'planned_geometric_paths'}
    require(not set(scene) - allowed, 'CURRENT_SCENE must contain pre-execution fields only')
    forbidden = {'measured_events', 'agent_trajectories', 'object_trajectories', 'outcomes',
                 'executed_release_times_s', 'observation_end_time_s', 'future_trajectory'}
    def check_initial(value):
        if isinstance(value, dict):
            require(not forbidden.intersection(value), 'Historical/future data in CURRENT_SCENE')
            for item in value.values():
                check_initial(item)
        elif isinstance(value, list):
            for item in value:
                check_initial(item)
    check_initial(scene)
    scene_index(scene)
    require(bool(scene['scene_id']) and bool(scene['controller_id']), 'Scene/controller ID required')
    demos = request['demonstrations']
    require(len(demos) == 5, 'Exactly five separate historical demonstrations required')
    unique(demos, 'example_id')
    for d in demos:
        require(d['controller_id'] == scene['controller_id'], 'Controller/checkpoint mismatch')
        require(d.get('scene_id', d['example_id']) != scene['scene_id'], 'Current scene is not historical evidence')
        agents, owners, _ = scene_index(d)
        releases = d['executed_release_times_s']
        require(set(releases) == set(agents), 'Incomplete demonstration release times')
        for t in releases.values():
            seconds(t)
        end = seconds(d['observation_end_time_s'])
        seen = set()
        for e in d['measured_events']:
            event = e['event_id']
            require(event in owners and owners[event] == e['agent_id'], 'Invalid measured event owner')
            require(event not in seen, 'Duplicate measured event')
            seen.add(event)
            require(type(e['reached']) is bool, 'reached must be boolean')
            if e['reached']:
                t = seconds(e['episode_time_s'])
                elapsed = seconds(e['elapsed_since_release_s'])
                require(t <= end and abs(t - releases[e['agent_id']] - elapsed) < 1e-6,
                        'Event label includes waiting or is outside observation window')
            else:
                require(e.get('episode_time_s') is None and e.get('elapsed_since_release_s') is None,
                        'Censored event must have null timing labels')
        require(seen == set(owners), 'Annotate every event, including censored failures')
        for prefix, ids in [('agent', set(agents)),
                            ('object', {o['object_id'] for o in d['initial_observation'].get('objects', [])})]:
            columns = d['trajectory_columns' if prefix == 'agent' else 'object_trajectory_columns']
            traces = d[prefix + '_trajectories']
            require(columns and columns[0] == 'episode_time_s', 'Trajectory timestamps must be episode time')
            require(set(traces) == ids, 'Missing or extra trajectory IDs; use [] for unavailable samples')
            for rows in traces.values():
                previous = -1
                for row in rows:
                    require(len(row) == len(columns), 'Trajectory column mismatch')
                    t = seconds(row[0])
                    require(previous <= t <= end, 'Unordered or out-of-window trajectory')
                    previous = t
    return request


def build_messages(request):
    validate_request(request)
    prompt = (Path(__file__).resolve().parents[1] / 'prompts/system.txt').read_text()
    return [{'role': 'system', 'content': prompt},
            {'role': 'user', 'content': json.dumps(request, ensure_ascii=False, allow_nan=False)}]


def earliest_schedule(scene, predictions):
    """Componentwise earliest releases, in integer centiseconds.

    r[v] >= r[u] + duration[u_event] - duration[v_event].
    Alignment contributes edges in both directions. A positive cycle is infeasible.
    This is a checker/reference solver; never silently rewrites an LLM answer.
    """
    agents, owners, required = scene_index(scene)
    duration = {p['event_id']: seconds(p['estimated_elapsed_time_s'], True) for p in predictions}
    require(required <= duration.keys(), 'Missing prediction for constrained event')
    edges = []
    def edge(first, second):
        edges.append((owners[first], owners[second], duration[first] - duration[second]))
    for c in scene['coordination_constraints']:
        if c['type'] == 'enabling':
            edge(c['enabling_event_id'], c['dependent_event_id'])
        else:
            first = c['event_ids'][0]
            for other in c['event_ids'][1:]:
                edge(first, other)
                edge(other, first)
    release = dict.fromkeys(agents, 0)
    for _ in range(len(agents)):
        changed = False
        for u, v, weight in edges:
            if release[v] < release[u] + weight:
                release[v] = release[u] + weight
                changed = True
        if not changed:
            return {a: t / 100 for a, t in release.items()}
    raise ValueError('Infeasible timing constraints: positive cycle')


def validate_response(request, response):
    validate_request(request)
    require(set(response) == {'status', 'predictions', 'schedule', 'predicted_episode_events', 'limitations'},
            'Invalid response keys')
    json.dumps(response, allow_nan=False)
    for key in ('predictions', 'schedule', 'predicted_episode_events', 'limitations'):
        require(isinstance(response[key], list), f'{key} must be a list')
    status = response['status']
    require(status in ('ok', 'insufficient_information', 'infeasible'), 'Unknown status')
    require(all(isinstance(x, str) and x.strip() for x in response['limitations']), 'Invalid limitations')
    scene = request['current_scene']
    agents, owners, required = scene_index(scene)
    example_ids = {d['example_id'] for d in request['demonstrations']}
    predictions = response['predictions']
    unique(predictions, 'event_id')
    for p in predictions:
        require(set(p) == {'agent_id', 'event_id', 'estimated_elapsed_time_s',
                           'supporting_example_ids', 'confidence', 'basis'}, 'Invalid prediction keys')
        require(p['event_id'] in owners and p['agent_id'] == owners[p['event_id']], 'Invalid prediction owner')
        seconds(p['estimated_elapsed_time_s'], True)
        support = p['supporting_example_ids']
        require(isinstance(support, list) and bool(support) and
                all(isinstance(s, str) for s in support) and set(support) <= example_ids,
                'Prediction must cite supplied empirical examples')
        require(p['confidence'] in ('low', 'medium', 'high'), 'Invalid confidence')
        require(isinstance(p['basis'], str) and bool(p['basis'].strip()), 'Missing prediction evidence')
    if status != 'ok':
        require(response['schedule'] == [] and response['predicted_episode_events'] == [],
                'Non-ok answer cannot be executed or have episode event times')
        require(bool(response['limitations']), 'Explain non-ok status')
        return response
    require(required <= {p['event_id'] for p in predictions}, 'Missing required predictions')
    schedule = response['schedule']
    require(set(unique(schedule, 'agent_id')) == set(agents), 'Schedule must cover every agent exactly once')
    for row in schedule:
        require(set(row) == {'agent_id', 'release_time_s'}, 'Invalid schedule keys')
        seconds(row['release_time_s'], True)
    releases = {r['agent_id']: r['release_time_s'] for r in schedule}
    require(min(releases.values()) == 0, 'At least one release must be zero')
    require(releases == earliest_schedule(scene, predictions),
            'Schedule violates constraints or adds unnecessary waiting')
    events = response['predicted_episode_events']
    require(set(unique(events, 'event_id')) == {p['event_id'] for p in predictions}, 'Episode event coverage mismatch')
    duration = {p['event_id']: seconds(p['estimated_elapsed_time_s'], True) for p in predictions}
    for e in events:
        require(set(e) == {'agent_id', 'event_id', 'episode_time_s'}, 'Invalid episode event keys')
        require(e['agent_id'] == owners[e['event_id']], 'Invalid episode event owner')
        expected = seconds(releases[e['agent_id']], True) + duration[e['event_id']]
        require(seconds(e['episode_time_s'], True) == expected, 'Episode time != release + elapsed')
    return response

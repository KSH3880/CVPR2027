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
    require(set(request) in ({'environment_spec', 'current_scene'},
                             {'environment_spec', 'current_scene', 'demonstrations'}), 'Invalid request keys')
    if 'demonstrations' in request:
        demos = request['demonstrations']
        require(isinstance(demos, list) and len(demos) == 10, 'Expected ten demonstrations')
        unique(demos, 'example_id')
        require(sum(d.get('task_type') == 'AT' for d in demos) == 5 and
                sum(d.get('task_type') == 'PUSH' for d in demos) == 5, 'Expected five AT and five PUSH records')
        require(all(d.get('execution', {}).get('agents') == 1 and d.get('measured_events')
                    for d in demos), 'Expected single-agent annotated recordings')
    # Reject NaN/Infinity anywhere, including initial observations.
    json.dumps(request, allow_nan=False)
    env = request['environment_spec']
    require(env['units'] == {'position': 'meters', 'time': 'seconds', 'angle': 'radians'}, 'Unsupported units')
    require(bool(env.get('coordinate_system')), 'Missing field definitions')
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
    return request


def build_messages(request):
    validate_request(request)
    variant = 'prompts/fewshot/system.txt' if 'demonstrations' in request else 'prompts/system.txt'
    prompt = (Path(__file__).resolve().parents[1] / variant).read_text()
    return [{'role': 'system', 'content': prompt},
            {'role': 'user', 'content': json.dumps(request, ensure_ascii=False, allow_nan=False)}]


def earliest_schedule(scene, predictions):
    """Componentwise earliest releases, in integer centiseconds.

    Enabling uses strict event ordering on the output's centisecond grid.
    Alignment uses equality; no physical safety margin is added.
    Alignment contributes edges in both directions. A positive cycle is infeasible.
    Optional reference solver for fixtures/preflight, not the acceptance rule.
    Never silently rewrites an LLM answer; later enabling releases are valid.
    """
    agents, owners, required = scene_index(scene)
    duration = {p['event_id']: seconds(p['estimated_elapsed_time_s'], True) for p in predictions}
    require(required <= duration.keys(), 'Missing prediction for constrained event')
    edges = []
    fixed = set()
    def edge(first, second, strict=False):
        edges.append((owners[first], owners[second], duration[first] - duration[second] + int(strict)))
    for c in scene['coordination_constraints']:
        if c['type'] == 'enabling':
            edge(c['enabling_event_id'], c['dependent_event_id'], strict=True)
            fixed.add(owners[c['enabling_event_id']])
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
                require(v not in fixed, 'Infeasible timing constraints: enabling predecessor must start at zero')
                release[v] = release[u] + weight
                changed = True
        if not changed:
            return {a: t / 100 for a, t in release.items()}
    raise ValueError('Infeasible timing constraints: positive cycle')


def validate_response(request, response):
    validate_request(request)
    require(set(response) == {'predictions', 'schedule', 'predicted_episode_events'},
            'Invalid response keys')
    json.dumps(response, allow_nan=False)
    for key in ('predictions', 'schedule', 'predicted_episode_events'):
        require(isinstance(response[key], list), f'{key} must be a list')
    scene = request['current_scene']
    agents, owners, required = scene_index(scene)
    predictions = response['predictions']
    unique(predictions, 'event_id')
    for p in predictions:
        require(set(p) == {'agent_id', 'event_id', 'estimated_elapsed_time_s',
                           'confidence', 'basis'}, 'Invalid prediction keys')
        require(p['event_id'] in owners and p['agent_id'] == owners[p['event_id']], 'Invalid prediction owner')
        seconds(p['estimated_elapsed_time_s'], True)
        require(p['confidence'] in ('low', 'medium', 'high'), 'Invalid confidence')
        require(isinstance(p['basis'], str) and bool(p['basis'].strip()), 'Missing prediction evidence')
    require(required <= {p['event_id'] for p in predictions}, 'Missing required predictions')
    schedule = response['schedule']
    require(set(unique(schedule, 'agent_id')) == set(agents), 'Schedule must cover every agent exactly once')
    for row in schedule:
        require(set(row) == {'agent_id', 'release_time_s'}, 'Invalid schedule keys')
        seconds(row['release_time_s'], True)
    releases = {r['agent_id']: r['release_time_s'] for r in schedule}
    require(min(releases.values()) == 0, 'At least one release must be zero')
    participating = {owners[e] for e in required}
    require(all(releases[a] == 0 for a in agents if a not in participating),
            'Independent agents must start at zero')
    duration = {p['event_id']: seconds(p['estimated_elapsed_time_s'], True) for p in predictions}
    episode = {event: seconds(releases[owners[event]], True) + elapsed
               for event, elapsed in duration.items()}
    for constraint in scene['coordination_constraints']:
        if constraint['type'] == 'enabling':
            first, second = constraint['enabling_event_id'], constraint['dependent_event_id']
            require(releases[owners[first]] == 0, 'Enabling predecessor must start at zero')
            require(episode[second] > episode[first],
                    'Schedule violates constraints: follower event must be strictly later')
        else:
            require(len({episode[e] for e in constraint['event_ids']}) == 1,
                    'Schedule violates constraints: coupled events must align')
    events = response['predicted_episode_events']
    require(set(unique(events, 'event_id')) == {p['event_id'] for p in predictions}, 'Episode event coverage mismatch')
    duration = {p['event_id']: seconds(p['estimated_elapsed_time_s'], True) for p in predictions}
    for e in events:
        require(set(e) == {'agent_id', 'event_id', 'episode_time_s'}, 'Invalid episode event keys')
        require(e['agent_id'] == owners[e['event_id']], 'Invalid episode event owner')
        expected = seconds(releases[e['agent_id']], True) + duration[e['event_id']]
        require(seconds(e['episode_time_s'], True) == expected, 'Episode time != release + elapsed')
    return response

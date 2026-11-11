"""Execution-only validation; deliberately does not repair LLM schedules."""
from .core import require, seconds


def validate_executable(request, response):
    ids = [a['agent_id'] for a in request['current_scene']['initial_observation']['agents']]
    rows = response.get('schedule')
    require(isinstance(rows, list), 'Missing schedule')
    actual = [r['agent_id'] for r in rows]
    require(len(actual) == len(ids) and set(actual) == set(ids),
            'Schedule must cover every agent exactly once')
    for row in rows:
        seconds(row['release_time_s'], True)
    return response

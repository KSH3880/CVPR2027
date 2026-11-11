"""CPU plumbing fixture ONLY. Not empirical timing evidence or evaluation data."""
import json
from pathlib import Path


def fixture(mode='coupled'):
    scene = {
        'scene_id': 'SYNTHETIC_CURRENT', 'controller_id': 'SYNTHETIC_NOT_A_CHECKPOINT',
        'coordination_type': mode,
        'initial_observation': {'agents': [
            {'agent_id': 'A', 'root_position_m': [-3., 0., .9], 'root_yaw_rad': 0., 'task_id': 'task_A'},
            {'agent_id': 'B', 'root_position_m': [-5., 0., .9], 'root_yaw_rad': 0., 'task_id': 'task_B'}],
            'objects': [], 'obstacles': []},
        'tasks': [
            {'task_id': 'task_A', 'agent_id': 'A', 'description': 'Synthetic first agent event.', 'readiness_event_id': 'A_ready'},
            {'task_id': 'task_B', 'agent_id': 'B', 'description': 'Synthetic second agent event.', 'readiness_event_id': 'B_ready'}],
        'event_definitions': {'A_ready': 'Synthetic detector A, not a physical task definition.',
                              'B_ready': 'Synthetic detector B, not a physical task definition.'},
        'coordination_constraints': [{'type': 'align_events', 'event_ids': ['A_ready', 'B_ready']}]}
    durations = [4., 6.]
    releases = [2., 0.]
    if mode == 'enabling':
        scene['coordination_constraints'] = [{'type': 'enabling', 'enabling_event_id': 'A_ready', 'dependent_event_id': 'B_ready'}]
        durations, releases = [5., 3.], [0., 2.01]
    env = {'units': {'position': 'meters', 'time': 'seconds', 'angle': 'radians'},
           'coordinate_system': {'frame': 'shared world frame within this scene', 'up_axis': 'z',
                                 'heading': 'yaw measured counterclockwise from +x'},
           'controller': {'description': 'A fixed pretrained controller executes each assigned task after release.',
                          'scheduler_control': 'One initial release time per agent.',
                          'before_release': 'The agent is physically frozen.',
                          'after_release': 'The agent acts autonomously without further scheduling intervention.'},
           'task_glossary': {'synthetic': 'TEST DATA ONLY; not empirical demonstrations.'}}
    request = {'environment_spec': env, 'current_scene': scene}
    response = {'predictions': [
        {'agent_id': a, 'event_id': a + '_ready', 'estimated_elapsed_time_s': t,
         'confidence': 'low', 'basis': 'Synthetic plumbing fixture only.'}
        for a, t in zip(('A', 'B'), durations)],
        'schedule': [{'agent_id': a, 'release_time_s': r} for a, r in zip(('A', 'B'), releases)],
        'predicted_episode_events': [{'agent_id': a, 'event_id': a + '_ready', 'episode_time_s': t+r}
                                     for a, t, r in zip(('A', 'B'), durations, releases)]}
    return request, response


if __name__ == '__main__':
    for mode in ('coupled', 'enabling'):
        for label, data in zip(('request', 'response'), fixture(mode)):
            path = Path(__file__).parent / f'synthetic_{mode}_{label}.json'
            path.write_text(json.dumps(data, indent=2))

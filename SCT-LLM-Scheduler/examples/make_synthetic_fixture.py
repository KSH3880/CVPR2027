"""CPU plumbing fixture ONLY. Not empirical timing evidence or evaluation data."""
import json
from copy import deepcopy
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
        durations, releases = [5., 3.], [0., 2.]
    demos = []
    for i in range(5):
        demo = deepcopy(scene)
        demo.update(scene_id=f'SYNTHETIC_HISTORY_{i}', example_id=f'synthetic_{i}',
                    executed_release_times_s={'A': 1., 'B': 0.},
                    trajectory_columns=['episode_time_s', 'root_x_m'],
                    agent_trajectories={'A': [[0., -3.], [1., -3.]], 'B': [[0., -5.]]},
                    object_trajectory_columns=['episode_time_s', 'x_m', 'y_m', 'z_m'], object_trajectories={},
                    measured_events=[{'agent_id': a, 'event_id': a + '_ready', 'reached': True,
                                      'episode_time_s': t + (1 if a == 'A' else 0), 'elapsed_since_release_s': t}
                                     for a, t in zip(('A', 'B'), durations)],
                    observation_end_time_s=10., outcomes={'synthetic': True, 'success': None})
        demos.append(demo)
    env = {'units': {'position': 'meters', 'time': 'seconds', 'angle': 'radians'},
           'coordinate_system': {'frame': 'shared world frame within this scene', 'up_axis': 'z',
                                 'heading': 'yaw measured counterclockwise from +x'},
           'controller': {'description': 'A fixed pretrained controller executes each assigned task after release.',
                          'scheduler_control': 'One initial release time per agent.',
                          'before_release': 'The agent is physically frozen.',
                          'after_release': 'The agent acts autonomously without further scheduling intervention.'},
           'trajectory_format': {'timestamps': 'Episode time, including initial frozen waiting.',
                                  'sampling': 'Downsampled observations with extra event samples.',
                                  'missing_values': 'null means unavailable.'},
           'task_glossary': {'synthetic': 'TEST DATA ONLY; not empirical demonstrations.'}}
    request = {'environment_spec': env, 'demonstrations': demos, 'current_scene': scene}
    response = {'status': 'ok', 'predictions': [
        {'agent_id': a, 'event_id': a + '_ready', 'estimated_elapsed_time_s': t,
         'supporting_example_ids': ['synthetic_0'], 'confidence': 'low', 'basis': 'Synthetic plumbing fixture only.'}
        for a, t in zip(('A', 'B'), durations)],
        'schedule': [{'agent_id': a, 'release_time_s': r} for a, r in zip(('A', 'B'), releases)],
        'predicted_episode_events': [{'agent_id': a, 'event_id': a + '_ready', 'episode_time_s': t+r}
                                     for a, t, r in zip(('A', 'B'), durations, releases)],
        'limitations': ['Synthetic data; no physical success or LLM quality measured.']}
    return request, response


if __name__ == '__main__':
    for mode in ('coupled', 'enabling'):
        for label, data in zip(('request', 'response'), fixture(mode)):
            path = Path(__file__).parent / f'synthetic_{mode}_{label}.json'
            path.write_text(json.dumps(data, indent=2))

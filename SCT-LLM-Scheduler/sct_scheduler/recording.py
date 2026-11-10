"""Record authoritative events without estimating them from downsampled paths."""
from copy import deepcopy
from .core import require, seconds, scene_index


class DemonstrationRecorder:
    def __init__(self, initial_scene, example_id, executed_releases,
                 agent_columns, object_columns):
        self.data = deepcopy(initial_scene)
        agents, owners, _ = scene_index(initial_scene)
        require(set(executed_releases) == set(agents), 'Actual release times required for all agents')
        for t in executed_releases.values():
            seconds(t)
        require(agent_columns[0] == object_columns[0] == 'episode_time_s', 'Episode timestamp first')
        self.owners = owners
        self.events = {}
        self.data.update(example_id=example_id, executed_release_times_s=dict(executed_releases),
                         trajectory_columns=list(agent_columns), object_trajectory_columns=list(object_columns),
                         agent_trajectories={a: [] for a in agents},
                         object_trajectories={o['object_id']: [] for o in initial_scene['initial_observation'].get('objects', [])})
        self.last_time = 0

    def sample(self, episode_time_s, agent_values, object_values):
        t = seconds(episode_time_s)
        require(t >= self.last_time, 'Time must be monotonic')
        pending = []
        for values, traces, columns in (
                (agent_values, self.data['agent_trajectories'], self.data['trajectory_columns']),
                (object_values, self.data['object_trajectories'], self.data['object_trajectory_columns'])):
            require(set(values) == set(traces), 'Sample all IDs; use null for unavailable values')
            for key, row in values.items():
                require(len(row) + 1 == len(columns), 'Sample column mismatch')
                pending.append((traces[key], [t] + list(row)))
        for rows, row in pending:
            rows.append(row)
        self.last_time = t

    def event(self, event_id, episode_time_s):
        """Call at the exact detector frame, independently of trajectory sampling."""
        require(event_id in self.owners, 'Unknown event')
        t = seconds(episode_time_s)
        agent = self.owners[event_id]
        elapsed = t - self.data['executed_release_times_s'][agent]
        require(elapsed >= 0, 'Event before release')
        if event_id in self.events:
            require(t >= self.events[event_id]['episode_time_s'], 'Event detector time went backwards')
            return
        self.events[event_id] = dict(agent_id=agent, event_id=event_id, reached=True,
                                     episode_time_s=t, elapsed_since_release_s=elapsed)

    def finish(self, observation_end_time_s, outcomes):
        end = seconds(observation_end_time_s)
        require(end >= self.last_time and all(e['episode_time_s'] <= end for e in self.events.values()), 'End before observations')
        data = deepcopy(self.data)
        data['measured_events'] = [deepcopy(self.events.get(event, dict(
            agent_id=agent, event_id=event, reached=False, episode_time_s=None,
            elapsed_since_release_s=None))) for event, agent in self.owners.items()]
        data.update(observation_end_time_s=end, outcomes=deepcopy(outcomes))
        return data

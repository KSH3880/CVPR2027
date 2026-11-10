"""Immutable initial schedule; simulation time only, no observations or gates."""
from decimal import Decimal, ROUND_CEILING
from .core import require, seconds


class ReleaseClock:
    def __init__(self, releases, control_dt_s):
        require(bool(releases), 'Empty schedule cannot execute')
        self.dt = seconds(control_dt_s)
        require(self.dt > 0, 'Control dt must be positive')
        self._ticks = {}
        for agent, value in releases.items():
            seconds(value, True)
            self._ticks[agent] = int((Decimal(str(value)) / Decimal(str(self.dt))).to_integral_value(rounding=ROUND_CEILING))
        self.frame = -1
        self.released = set()
        self.events = []

    def advance(self, frame):
        require(type(frame) is int and frame >= self.frame and frame >= 0, 'Simulation clock cannot go backwards')
        self.frame = frame
        for agent, tick in self._ticks.items():
            if frame >= tick and agent not in self.released:
                self.released.add(agent)
                self.events.append({'agent_id': agent, 'release_frame': frame,
                                    'actual_release_time_s': frame * self.dt})
        return {agent: agent not in self.released for agent in self._ticks}

    def quantized_release_times(self):
        return {agent: tick * self.dt for agent, tick in self._ticks.items()}

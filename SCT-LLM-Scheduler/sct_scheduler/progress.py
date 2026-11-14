"""Stage ETA from completed work; never count reused API answers as new throughput."""
import time
from datetime import datetime, timezone

class StageETA:
    def __init__(self):
        self.stage = None
        self.started = None

    def annotate(self, state, now=None):
        now = time.time() if now is None else now
        stage = state.get('stage')
        if stage != self.stage:
            self.stage, self.started = stage, now
        progress = state.get('progress', state)
        seconds = None
        basis = 'Insufficient completed work for this stage'
        if state.get('completed'):
            seconds, basis = 0., 'Completed'
        elif stage == 'api':
            started = progress.get('started_at')
            start = time.mktime(time.strptime(started, '%Y-%m-%d %H:%M:%S')) if started else self.started
            elapsed = max(0., now-start)
            done = progress.get('api_done', 0)
            reused = progress.get('api_reused', 0)
            total = progress.get('total', state.get('total_scenes', 0))
            completed_new = max(0, done-reused)
            if total and done >= total:
                seconds = 0.
            elif completed_new and elapsed > 0:
                seconds = elapsed/completed_new * max(0, total-done)
            basis = 'Current API throughput, excluding reused responses; excludes later GPU evaluation'
        elif stage == 'eval':
            done = len(progress.get('verified', []))
            total = progress.get('total_evaluations', state.get('total_evaluations', 0))
            if done:
                seconds = max(0., now-self.started)/done * max(0, total-done)
            basis = 'Observed verified-job throughput; approximate across GPU lanes and different scene sizes'
        if stage in ('failed', 'stopped_by_user', 'stopped_for_2h_validation'):
            seconds, basis = None, 'Not running'
        state.update(eta_seconds=round(seconds, 1) if seconds is not None else None,
                     eta_at=datetime.fromtimestamp(now+seconds, timezone.utc).astimezone().isoformat(timespec='seconds') if seconds is not None else None,
                     eta_scope='current_stage', eta_basis=basis,
                     eta_updated_at=datetime.fromtimestamp(now, timezone.utc).astimezone().isoformat(timespec='seconds'))
        return state

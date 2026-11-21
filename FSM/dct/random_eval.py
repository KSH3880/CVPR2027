import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).parent), str(ROOT/'FSM'), str(ROOT/'eval'), str(ROOT/'tokenhsi')]
import isaacgym
import torch

from common import read, write, sha
import freeze


class RandomDelayFreeze(freeze.DelayFreeze):
    def __init__(self, task, delay, use_fsm):
        path = Path(os.environ['DCT_DELAY_SCHEDULE'])
        self.schedule = read(path)
        self.schedule_hash = sha(path)
        steps = self.schedule['steps']
        if len(steps) != task.num_envs or min(steps) < 30 or max(steps) > 150:
            raise ValueError('Wrong per-environment delays')
        if abs(task.dt-1/30) > 1e-8:
            raise ValueError('Unexpected control timestep')
        super().__init__(task, 1, use_fsm)
        self.delay_steps = torch.tensor(steps, device=task.device)[:, None]

    def finish(self):
        proof = super().finish()
        proof.update(delay_steps=self.delay_steps[:, 0].cpu().tolist(),
            release_before_action_step=(self.delay_steps[:, 0]+1).cpu().tolist(),
            schedule_sha256=self.schedule_hash, delay_condition='random_1_5')
        return proof


if __name__ == '__main__':
    freeze.DelayFreeze = RandomDelayFreeze
    entry = 'tokenhsi_eval.py' if os.environ['DCT_MODEL'] == 'tokenhsi' else 'evaluate.py'
    runpy.run_path(str(ROOT/'FSM/dct'/entry), run_name='__main__')
    path = Path(os.environ['DCT_OUTPUT'])
    schedule_path = Path(os.environ['DCT_DELAY_SCHEDULE'])
    for name in ('table2.json', 'result.json'):
        value = read(path/name)
        value.update(delay_seconds='random_1_5', delay_schedule=str(schedule_path),
            delay_schedule_sha256=sha(schedule_path))
        if name == 'result.json':
            value['table2'] = read(path/'table2.json')
        write(path/name, value)

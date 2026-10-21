import hashlib
import os
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from joint_carry.artifacts import contract_digest


def profile_name(size):
    return '_'.join(str(round(float(value)*20)) for value in size)


class JointRsiLibrary:
    def __init__(self, task):
        self.task = task
        self.directory = Path(task.cfg['env']['stage2']['rsi_directory'])
        self.inset = task.cfg['env']['stage2']['anchor_inset']
        self.pools = {}

    def sample(self, skill, sizes):
        result = []
        requests = {}
        for size in sizes.cpu().tolist():
            key = profile_name(size)
            path = self.directory/key/(skill+'.npz')
            metadata = path.with_suffix(".json")
            saved = json.loads(metadata.read_text()) if metadata.exists() else {}
            ready = path.exists() and saved.get('accepted', 0) > 0 and saved.get("contract_digest") == contract_digest('single', self.inset) and np.allclose(saved.get("size", []), size, atol=1e-5)
            if not ready:
                requests[key] = dict(size=size)
        if requests:
            output = Path('output_etc/check/stage2_rsi_build')/str(os.getpid())
            output.mkdir(parents=True, exist_ok=True)
            request_file = output/'requests.json'
            request_file.write_text(json.dumps(list(requests.values()), indent=2)+'\n')
            subprocess.run([sys.executable, 'joint_carry/build_size_rsi.py', '--requests', str(request_file),
                            '--directory', str(self.directory), '--source', 'single', '--inset', str(self.inset),
                            '--report', str(output/'report.json')], check=True)
        for size in sizes.cpu().tolist():
            key = (profile_name(size), skill)
            if key not in self.pools:
                path = self.directory/key[0]/(skill+'.npz')
                metadata = json.loads(path.with_suffix('.json').read_text())
                if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['sha256']:
                    raise ValueError('Joint RSI changed: '+str(path))
                with np.load(path, allow_pickle=False) as archive:
                    pool = {name: torch.tensor(archive[name], device=self.task.device,
                                              dtype=torch.long if name == 'source_frame' else torch.float)
                            for name in ('root_state', 'dof_position', 'dof_velocity', 'body_state',
                                         'box_state', 'terminal_box_xy', 'source_frame')}
                if not len(pool['source_frame']):
                    raise ValueError('No screened joint RSI for '+str(key))
                self.pools[key] = pool
            pool = self.pools[key]
            index = torch.randint(len(pool['source_frame']), (), device=self.task.device)
            result.append({name: value[index].clone() for name, value in pool.items()})
        return {name: torch.stack([row[name] for row in result]) for name in result[0]}

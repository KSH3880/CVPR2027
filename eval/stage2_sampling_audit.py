import hashlib
import json
from pathlib import Path

import torch
from env.tasks.multi_agent.humanoid_ma_stage2 import HumanoidMAStage2
from utils import coupled_push_scene


def record(task, name, ids, result=None):
    if not hasattr(task, '_sampling_audit'):
        task._sampling_audit = []
    values = dict(ids=ids.cpu(), rng=torch.cuda.get_rng_state().cpu(), cpu_rng=torch.get_rng_state(),
        roots=task._humanoid_root_states.cpu().clone(), boxes=task._box_states.cpu().clone(),
        goals=task._tar_pos.cpu().clone())
    if isinstance(result, torch.Tensor):
        values['result'] = result.cpu()
    hashes = {key: hashlib.sha256(value.numpy().tobytes()).hexdigest() for key, value in values.items()}
    folder = Path(task.cfg['args'].output_path)
    index = len(task._sampling_audit)
    if index < 12:
        torch.save(values, folder/f'audit_{index:03d}_{name}.pt')
    task._sampling_audit.append(dict(index=index, stage=name, count=len(ids), hashes=hashes))
    (folder/'sampling_audit.json').write_text(json.dumps(task._sampling_audit, indent=2))


def wrap(original, name):
    def method(self, ids):
        record(self, name+'_before', ids)
        result = original(self, ids)
        record(self, name+'_after', ids, result)
        return result
    return method


for name in ('_reset_actors', '_reset_boxes', '_configure_ontop_targets', '_ontop_scene_infeasible'):
    setattr(HumanoidMAStage2, name, wrap(getattr(HumanoidMAStage2, name), name))
for name in ('sample_push', 'place_distractors'):
    setattr(coupled_push_scene, name, wrap(getattr(coupled_push_scene, name), name))

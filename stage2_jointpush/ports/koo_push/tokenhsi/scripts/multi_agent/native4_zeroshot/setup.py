"""Resolve source Stage1 checkpoint and materialize its saved native4 configs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
import yaml
TOKENHSI = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TOKENHSI))
from utils.partition_push_paths import resolve_stage1_checkpoint


def prepare(output, checkpoint=None, agents=4):
    if agents not in (2, 4):
        raise ValueError('Native diagnostic supports 2 or 4 agents')
    source = resolve_stage1_checkpoint(checkpoint)
    saved = torch.load(str(source), map_location='cpu', weights_only=False)
    experiment = saved['relation_experiment_config']
    if (experiment['num_agents'], experiment['num_objects']) != (2, 4):
        raise ValueError('Expected original two-person Stage1 checkpoint')
    if 'joint_push' in saved.get('open_push_contract', {}) or saved['relation_metadata']['schema_version'] != 12:
        raise ValueError('Expected original semantic OPEN/PUSH Stage1 weights')
    env = experiment['env']
    if env['stateInit'] != 'Start' or env['openPushLayout']['lane_spacing'] != 10:
        raise ValueError('This diagnostic reproduces Start initialization and 10m native lane spacing')
    env['numAgents'], env['numObjects'] = agents, 2 * agents
    native = yaml.safe_load((TOKENHSI/'data/cfg/multi_agent/open_push_push180_contact_at_warmstart.yaml').read_text())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/'env.yaml').write_text(yaml.safe_dump(dict(env=env, sim=native['sim']), sort_keys=False))
    (output/'train.yaml').write_text(yaml.safe_dump(experiment['train'], sort_keys=False))
    manifest = dict(source_checkpoint=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    source_epoch=int(saved['epoch']), agents=agents, physical_doors=agents, physical_boxes=agents,
                    stateInit='Start', door_lane_y=[-15, -5, 5, 15] if agents == 4 else [-5, 5],
                    push_pair_centers_y=[-10, 10] if agents == 4 else [0], learning=False,
                    task_sha256=hashlib.sha256((Path(__file__).parent/'native4_task.py' if agents == 4 else
                        TOKENHSI/'env/tasks/multi_agent/humanoid_ma_open_push.py').read_bytes()).hexdigest(),
                    overrides=(['counts and duplicated workspace', 'PUSH-only reward branch extended'] if agents == 4 else []) +
                              ['installed converted motion manifest', 'unused evaluation interaction RSI loading skipped'])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('checkpoint', nargs='?')
    parser.add_argument('--agents', type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    print(prepare(args.output, args.checkpoint or None, args.agents))

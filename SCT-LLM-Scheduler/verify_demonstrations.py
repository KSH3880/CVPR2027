"""CPU replay check: exported samples/events must match the retained dense recording."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from export_demonstrations import ROOT, QUOTAS, candidate, build_demo
from sct_scheduler.core import scene_index


def verify(folder):
    demos = json.loads((folder/'demonstrations.json').read_text())
    counts = dict.fromkeys(QUOTAS, 0)
    loaded = {}
    report = []
    for index,demo in enumerate(demos,1):
        scene_index(demo)
        raw = ROOT/demo['provenance']['raw_directory']
        if raw not in loaded:
            meta = json.loads((raw/'recording.json').read_text())
            initial = torch.load(raw/'record_initial.pt',map_location='cpu',weights_only=False)
            with np.load(raw/'dense_trajectories.npz') as archive:
                data = {k:archive[k] for k in archive.files}
            loaded[raw] = initial,data,meta
        initial,data,meta = loaded[raw]
        counts[meta['setting']] += 1
        e = demo['provenance']['environment_index']
        item = candidate(raw,initial,data,meta,e)
        assert item is not None, 'Not a valid successful candidate'
        expected = build_demo(item,initial,data,meta,demo['controller_id'],index,30)
        assert expected == demo, 'Export differs from recorded observations/events'
        assert demo['outcomes']['group_success'] is True
        for event in demo['measured_events']:
            assert event['reached'] and event['episode_time_s'] <= demo['observation_end_time_s']
        assert len({a['agent_id'] for a in demo['initial_observation']['agents']}) == len(demo['tasks'])
        report.append(dict(example_id=demo['example_id'],setting=meta['setting'],environment=e,
            agents=len(demo['tasks']),completion_time_s=demo['outcomes']['first_group_success_time_s'],
            samples_per_agent=len(next(iter(demo['agent_trajectories'].values())))))
    assert counts == QUOTAS, 'Expected 5 enabling and 5 coupled with task/group diversity'
    result = dict(passed=True,count=len(demos),counts=counts,all_samples_replayed=True,
                  recorded_scene_schema_passed=True,examples=report,actual_llm_calls=0)
    (folder/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    verify(parser.parse_args().folder)

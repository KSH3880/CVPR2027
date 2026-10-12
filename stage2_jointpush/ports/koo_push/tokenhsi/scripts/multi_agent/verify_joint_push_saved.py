"""CPU verification of frozen source encoder/RMS and trained coordination."""
import argparse,json,math
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('target');p.add_argument('report');args=p.parse_args()
torch.set_num_threads(1)
source=torch.load(args.source,map_location='cpu',weights_only=False)
target=torch.load(args.target,map_location='cpu',weights_only=False)
encoder=[k for k in source['model'] if k.startswith('a2c_network.actor_encoder.')]
assert encoder and all(torch.equal(source['model'][k],target['model'][k]) for k in encoder)
assert set(source['running_mean_std'])==set(target['running_mean_std'])
assert all(torch.equal(source['running_mean_std'][k],target['running_mean_std'][k]) for k in source['running_mean_std'])
assert all(torch.isfinite(v).all() for v in target['model'].values())
first='a2c_network.action_head.0.0.weight'
assert target['model'][first].shape[1]==128
assert torch.count_nonzero(target['model'][first][:,64:])>0
bias=[k for k in target['model'] if '.push_at_coordination.blocks.' in k and k.endswith('relation_bias')]
assert len(bias)==4 and all(torch.count_nonzero(target['model'][k])>0 for k in bias)
contract=target['open_push_contract']
assert contract['joint_push']['rsi']=='loco_only' and contract['push_box']=={'size':[2.6,2.6,2.6],'mass':40.,'friction':.6}
assert target['relation_metadata']['relation_reward_config']['state_reward_weight']==.2
summaries=Path(args.target).parent.parent/'summaries';events=EventAccumulator(str(summaries));events.Reload()
scalars=events.Tags()['scalars'];assert scalars
assert all(math.isfinite(e.value) for tag in scalars for e in events.Scalars(tag))
report=dict(epoch=int(target['epoch']),actor_encoder_tensors=len(encoder),actor_encoder_unchanged=True,
    observation_rms_unchanged=True,head_context_columns_updated=True,ca4_relation_bias_updated=True,
    model_finite=True,scalar_tags_finite=len(scalars),contract=contract['joint_push'])
Path(args.report).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='contract'},indent=2))

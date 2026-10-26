"""Make the exact FT initialization checkpoint for evaluation; never train weights."""
import argparse,json,hashlib
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--template',required=True);p.add_argument('--output',required=True);args=p.parse_args()
source=torch.load(args.source,map_location='cpu',weights_only=False)
template=torch.load(args.template,map_location='cpu',weights_only=False)
converted={};audit={}
for name,target in template['model'].items():
 key=name.replace('.tokenizers.1.base.','.tokenizers.1.').replace('.tokenizers.1.chair.','.tokenizers.1.')
 weight=source['model'][key]
 if name.endswith('.tokenizers.1.chair.0.weight'):
  assert target.shape[1]==35 and weight.shape[1]==30
  result=torch.zeros_like(target);result[:,:30].copy_(weight)
  assert torch.equal(result[:,:30],weight) and not result[:,30:].any()
 else:
  assert weight.shape==target.shape,(name,weight.shape,target.shape)
  result=weight.clone();assert torch.equal(result,weight)
 converted[name]=result;audit[name]=key
# All original model tensors must be accounted for.
assert set(audit.values())==set(source['model'])
result={key:value for key,value in source.items() if key!='model'}
result['model']=converted
for key in ['relation_experiment_config','chair_ft_contract']:
 result[key]=template[key]
output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
assert not output.exists()
torch.save(result,output)
report=dict(source=str(Path(args.source).resolve()),source_sha256=hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),source_epoch=source['epoch'],model_tensors=len(converted),all_source_tensors_used=True,normalizers='copied from original 78k unchanged',adapter='same FT initialization: copy30D weights, extra5D zero',mapping=audit)
output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='mapping'},indent=2))

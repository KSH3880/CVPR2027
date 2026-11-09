"""Report recorded reference state differences without initializing CUDA."""
import json
import sys
from pathlib import Path
import torch

root = Path('output_etc/check/backend_parity')
a = torch.load(root/'trace_worker.pt', map_location='cpu', weights_only=True)['records']
label = sys.argv[1]
b = torch.load(root/('trace_'+label+'.pt'), map_location='cpu', weights_only=True)['records']
report = []
stages = {x['stage']: x for x in b}
for x in a:
    if x['stage'] not in stages:
        continue
    y = stages[x['stage']]
    result = dict(stage=x['stage'])
    for key in x:
        if key != 'stage' and key in y and isinstance(x[key], torch.Tensor):
            delta = (x[key].float()-y[key].float()).abs()
            result[key] = dict(max=delta.max().item(), mean=delta.mean().item())
    report.append(result)
    print(x['stage'], ' '.join('%s=%.5g' % (k,result[k]['max']) for k in ('root','dof','body','obs','reward','sizes','target','native_obs') if k in result))
(root/('comparison_'+label+'.json')).write_text(json.dumps(report, indent=2)+'\n')
for trace, name in ((a,'worker'),(b,label)):
    for i,x in enumerate(trace):
        if x['stage'].startswith('reset_native'):
            delta=(x['root']-trace[i-1]['root']).abs().max().item()
            print('REFERENCE_CHANGED_BY_NATIVE_RESET',name,x['stage'],delta)

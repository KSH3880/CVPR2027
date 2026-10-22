"""Check equality to original ObjectLib probabilities and replacement draws."""
import torch
from sampling import CATEGORIES,model_weights,sample_ids
catalog=[dict(category=c,id=str(i)) for c,n in zip(CATEGORIES,[18,9,8]) for i in range(n)]
weights=model_weights(catalog)
original=torch.tensor([1.0/3*(1.0/n) for n in [18,9,8] for _ in range(n)])
assert torch.equal(weights,original)
for category in CATEGORIES:
    selected=torch.tensor([e['category']==category for e in catalog])
    torch.testing.assert_close(weights[selected].sum(),torch.tensor(1/3))
for seed in [0,1,2]:
    g=torch.Generator().manual_seed(seed);got=sample_ids(catalog,512,'cpu',g)
    ref=torch.multinomial(original,512,replacement=True,generator=torch.Generator().manual_seed(seed))
    assert torch.equal(got,ref)
assert len(sample_ids(catalog,1,'cpu',torch.Generator().manual_seed(0)))==1
try:model_weights([{'category':'Box'}])
except ValueError:pass
else:raise AssertionError('Missing category accepted')
print('PASS original category prior, model weights, exact replacement draws seeds0/1/2, singleton shape, invalid catalog rejection')

"""TokenHSI ObjectLib's category-balanced replacement distribution."""
from collections import Counter
import torch
CATEGORIES = ('Box', 'Cabinet', 'Table_Square')

def model_weights(catalog, device='cpu'):
    counts=Counter(entry['category'] for entry in catalog)
    if set(counts)!=set(CATEGORIES):
        raise ValueError('Need nonempty Box, Cabinet, Table_Square test categories')
    return torch.tensor([1.0/len(CATEGORIES)/counts[e['category']] for e in catalog],device=device)

def sample_ids(catalog, num_envs, device, generator=None):
    return torch.multinomial(model_weights(catalog,device),num_samples=num_envs,replacement=True,generator=generator)

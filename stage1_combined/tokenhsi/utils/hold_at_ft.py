"""Read-only dynamic object catalog and strict 78k transfer contracts."""
import json
from pathlib import Path

def object_catalog(root, split, category):
    folder = Path(root) / split / category
    paths = sorted(folder.glob('*/config.json'))
    if not paths:
        raise FileNotFoundError(f'Dynamic object dataset missing: {folder}; set HOLD_AT_OBJECT_ROOT. SIT assets are not substituted.')
    result = []
    for path in paths:
        meta = json.loads(path.read_text())
        if any(abs(x)>1e-6 for x in meta['center']):
            raise ValueError(f'Expected zero object center: {path}')
        if not (path.parent/'asset.urdf').is_file():
            raise FileNotFoundError(path.parent/'asset.urdf')
        result.append(dict(meta, path=str(path.parent.resolve())))
    return result

def transfer_78k(model, checkpoint):
    if checkpoint['epoch'] != 78000:
        raise ValueError('Fresh transfer requires the original 78000 checkpoint')
    state = checkpoint['model']
    for branch in ('actor_encoder', 'critic_encoder'):
        key = f'a2c_network.{branch}.tokenizers.1.0.weight'
        if state[key].shape[1] != 30:
            raise ValueError(f'Expected original 30D object tokenizer: {key}')
    model.load_state_dict(state, strict=True)

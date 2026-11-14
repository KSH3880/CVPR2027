import hashlib
import io
import sys
from copy import deepcopy
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.before_spec import check_checkpoint, checkpoint_contract
from utils.experiments import BEFORE_TOTAL, BEFORE_TOTAL_PROGRESS0, ROOT, native_config


def prepare(source, destination):
    source = Path(source).resolve()
    data = source.read_bytes()
    weights = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
    target = native_config(BEFORE_TOTAL_PROGRESS0)
    if weights.get('before_contract') == checkpoint_contract(target['env']):
        return source
    check_checkpoint(weights, native_config(BEFORE_TOTAL)['env'])
    saved = deepcopy(weights['relation_experiment_config'])
    if weights['epoch'] != 9000 or saved['experiment']['name'] != BEFORE_TOTAL:
        raise ValueError('Initial progress0 resume requires total epoch 9000')
    if saved['env']['relationGraph'] != target['env']['relationGraph'] or saved['env']['episodeLength'] != 900:
        raise ValueError('Progress0 resume requires the total graph and 900 steps')
    digest = hashlib.sha256(data).hexdigest()
    folder = Path(destination) / digest[:16]
    checkpoint = folder / 'initial_00009000.pth'
    if checkpoint.is_file() and (folder / 'relation_config.yaml').is_file():
        cached = torch.load(checkpoint, map_location='cpu', weights_only=False)
        check_checkpoint(cached, target['env'])
        if cached['before_resume_source']['sha256'] != digest:
            raise ValueError('Prepared checkpoint source differs')
        return checkpoint.resolve()
    saved['env']['before']['progress_radius'] = 0.0
    saved['experiment']['name'] = BEFORE_TOTAL_PROGRESS0
    saved['train']['params']['config']['name'] = native_config(BEFORE_TOTAL_PROGRESS0, True)['params']['config']['name']
    weights['before_contract'] = checkpoint_contract(saved['env'])
    check_checkpoint(weights, target['env'])
    weights['relation_experiment_config'] = saved
    weights['before_resume_source'] = dict(checkpoint=str(source), sha256=digest, epoch=9000,
                                          change='BEFORE follower progress radius -> 0')
    folder.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix('.tmp')
    torch.save(weights, temporary)
    temporary.replace(checkpoint)
    (folder / 'relation_config.yaml').write_text(yaml.safe_dump(saved, sort_keys=False))
    return checkpoint.resolve()


if __name__ == '__main__':
    print(prepare(sys.argv[1], ROOT / 'output_etc/check/before_total_progress0_initial'))

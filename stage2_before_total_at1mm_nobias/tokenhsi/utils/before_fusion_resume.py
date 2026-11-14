import hashlib
import io
import sys
from copy import deepcopy
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.before_spec import check_checkpoint, checkpoint_contract
from utils.experiments import BEFORE_PUSH_RELFUSION, BEFORE_PUSH_RELFUSION_HUMAN3, ROOT, native_config


def prepare(source, destination):
    source = Path(source).resolve()
    data = source.read_bytes()
    weights = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
    target = native_config(BEFORE_PUSH_RELFUSION_HUMAN3)
    if weights.get('before_contract') == checkpoint_contract(target['env']):
        return source
    check_checkpoint(weights, native_config(BEFORE_PUSH_RELFUSION)['env'])
    if weights['epoch'] != 21000:
        raise ValueError('Initial human3 continuation requires fusion v1 epoch 21000')
    digest = hashlib.sha256(data).hexdigest()
    folder = Path(destination) / digest[:16]
    checkpoint = folder / 'initial_00021000.pth'
    if checkpoint.is_file() and (folder / 'relation_config.yaml').is_file():
        cached = torch.load(checkpoint, map_location='cpu', weights_only=False)
        check_checkpoint(cached, target['env'])
        if cached['before_resume_source']['sha256'] != digest:
            raise ValueError('Prepared checkpoint source differs')
        return checkpoint.resolve()
    saved = deepcopy(weights['relation_experiment_config'])
    saved['env']['before']['waiting_penalty']['ontop_distance_source'] = 'human'
    saved['experiment']['name'] = BEFORE_PUSH_RELFUSION_HUMAN3
    saved['train']['params']['config']['name'] = native_config(BEFORE_PUSH_RELFUSION_HUMAN3, True)['params']['config']['name']
    weights['before_contract'] = checkpoint_contract(saved['env'])
    check_checkpoint(weights, target['env'])
    weights['relation_experiment_config'] = saved
    weights['before_resume_source'] = dict(checkpoint=str(source), sha256=digest, epoch=21000,
                                          change='ON_TOP waiting entry: payload -> human')
    folder.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix('.tmp')
    torch.save(weights, temporary)
    temporary.replace(checkpoint)
    (folder / 'relation_config.yaml').write_text(yaml.safe_dump(saved, sort_keys=False))
    return checkpoint.resolve()


if __name__ == '__main__':
    destination = ROOT / 'output_etc/check/before_fusion_human3_resume21k_initial'
    print(prepare(sys.argv[1], destination))

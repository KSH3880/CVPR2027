"""Prepare a new RSI contract, or explicitly transfer the supplied 78000 policy on CPU."""
import argparse
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import torch
import yaml
from utils.stage1_push_spec import validate_env, checkpoint_contract, check_checkpoint
from utils.native_push_rsi import native_contract
from utils.edge_context_spec import task_instance
from utils.relation_task_spec import checkpoint_metadata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expand_model(model):
    result = {k: v.clone() for k, v in model.items()}
    for side in ('actor', 'critic'):
        key = f'a2c_network.{side}_encoder.edge_encoder.category_embed.weight'
        assert model[key].shape == (4, 64)
        # PUSH is a placement task: initialise its new row from carry_at (ID 2).
        result[key] = torch.cat((model[key], model[key][2:3]), 0)
    key = 'a2c_network._disc_mlp.0.weight'
    old = model[key].reshape(1024, 10, 132)
    result[key] = torch.cat((old, old[..., 129:130]), -1).reshape(1024, 1330)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('bank', 'transfer'))
    parser.add_argument('--source', type=Path, default=Path('weights/Stage1V2Fixed_00078000.pth'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(Path('tokenhsi/data/cfg/multi_agent/stage1_push_finetune.yaml').read_text())
    env = config['env']; env['motion_file'] = 'tokenhsi/data/dataset_stage1_push_finetune.yaml'
    validate_env(env)
    source_env = yaml.safe_load(Path(env['pushExtension']['source_env']).read_text())['env']
    source_env['motion_file'] = 'tokenhsi/data/dataset_push_native_phase_xy.yaml'
    bank = Path(env['openPushRSI']['bank_file'])
    if args.action == 'bank':
        if bank.exists():
            raise FileExistsError(bank)
        original = Path('tokenhsi/data/push_native_phase_y2m_rsi_bank.json')
        payload = json.loads(original.read_text())
        rows = payload['candidates']['push']
        assert len(rows) == 949 and {r['motion_id'] for r in rows} == {0,3,4,8}
        payload = dict(contract=native_contract(source_env), screen='pending_cpu_physics',
            candidates={'push': rows}, provenance=dict(source=str(original), sha256=sha(original)),
            limitations='Same requested 949 poses; new sizes/friction/square/cone require CPU geometry and physics checks.')
        bank.write_text(json.dumps(payload, indent=2)+'\n')
        print('Prepared pending bank:', bank)
        return
    payload = json.loads(bank.read_text())
    if payload['contract'] != native_contract(source_env) or payload['screen'] != 'PD_0.1s_all_profiles':
        raise ValueError('Complete the new RSI physics screen before creating a training checkpoint')
    target = args.output or Path('../../output/stage1_base_push_finetune_stable/resume/Stage1PushFinetuneStable_from78000.pth')
    if target.exists():
        raise FileExistsError(target)
    before = sha(args.source)
    original = torch.load(args.source, map_location='cpu', weights_only=False)
    saved = original['relation_experiment_config']
    assert original['epoch'] == 78000 and saved['observation_size'] == 634
    assert saved['env']['box']['reset']['platformPhysics'] == 'fixed_warmup_v1'
    assert saved['env']['relationReward'] == env['relationReward']
    train = yaml.safe_load(Path('tokenhsi/data/cfg/train/rlg/stage1_push_finetune.yaml').read_text())
    assert saved['train']['params']['network'] == train['params']['network']
    weights = deepcopy(original)
    weights['model'] = expand_model(original['model'])
    stats = weights['amp_input_mean_std']
    assert stats['running_mean'].shape == (132,)
    for key in ('running_mean','running_var'):
        stats[key] = torch.cat((stats[key], stats[key][129:130]))
    weights['relation_task_instance'] = task_instance(env['relationGraph'], 2, 4)
    weights['relation_metadata'] = checkpoint_metadata(env['relationReward'])
    weights['relation_experiment_config'] = dict(env=env, train=train, num_envs=2048,
        num_agents=2, num_objects=4, control_dt=saved['control_dt'], observation_size=634,
        experiment={}, sim=config['sim'])
    weights['stage1_push_contract'] = checkpoint_contract(env)
    weights['optimizer']['state'] = {}
    weights.update(epoch=0, frame=0, last_mean_rewards=-100500, env_state=None)
    weights['finetune_source'] = dict(path=str(args.source.resolve()), sha256=before,
        epoch=original['epoch'], frame=original['frame'], new_task_embedding='copy carry_at ID2',
        new_amp_family='copy carry family column and RMS', optimizer='fresh', counters='reset')
    check_checkpoint(weights, env, for_training=True)
    # All original task rows and every original discriminator feature are retained.
    unchanged = [k for k in original['model'] if original['model'][k].shape == weights['model'][k].shape]
    assert all(torch.equal(original['model'][k], weights['model'][k]) for k in unchanged)
    assert torch.equal(original['model']['a2c_network._disc_mlp.0.weight'].reshape(1024,10,132),
        weights['model']['a2c_network._disc_mlp.0.weight'].reshape(1024,10,133)[...,:132])
    assert before == sha(args.source)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, target)
    report = dict(source=weights['finetune_source'], target=str(target.resolve()),
        bank_sha256=sha(bank), unchanged_model_tensors=len(unchanged), changed_shapes=3,
        original_preserved=True, training_started=False)
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()

"""CPU-only preparation of frozen checkpoint/config and test mesh snapshots."""
from pathlib import Path
import argparse, copy, json, shutil, xml.etree.ElementTree as ET
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]

def prepare(checkpoint, output, object_root, count, template_checkpoint=None, public_chair_checkpoint=None):
    checkpoint, output, object_root = map(lambda x: Path(x).resolve(), (checkpoint, output, object_root))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(checkpoint, output / 'checkpoint.pth')
    weights = torch.load(output / 'checkpoint.pth', map_location='cpu', weights_only=False)
    saved = weights['relation_experiment_config']
    config_source = str(checkpoint)
    if template_checkpoint is not None:
        if int(weights['epoch']) != 78000:
            raise ValueError('A separate evaluation template is supported only for original 78k zero-shot')
        template_checkpoint = Path(template_checkpoint).resolve()
        template = torch.load(template_checkpoint, map_location='cpu', weights_only=False)
        saved = template['relation_experiment_config']
        config_source = str(template_checkpoint)
    if 'holdAtFT' not in saved['env']:
        raise ValueError('Original 78k needs --template-checkpoint with native Chair HOLDING_AT env/sim; weights are not converted')
    if saved['env']['holdAtFT']['category'] != 'ArmChair_Normal':
        raise ValueError('This protocol is specifically for native ArmChair_Normal')
    for branch in ('actor_encoder', 'critic_encoder'):
        assert weights['model'][f'a2c_network.{branch}.tokenizers.1.0.weight'].shape[1] == 30
    source = output / 'source'
    for name in ('tokenhsi', 'lpanlib', 'mps', 'body_models'):
        shutil.copytree(ROOT / name, source / name, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(Path(__file__).parent, source / 'evaluation/hold_at',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    assets = output / 'assets/test/ArmChair_Normal'
    shutil.copytree(object_root / 'test/ArmChair_Normal', assets, symlinks=False)
    entries = sorted(assets.glob('*/config.json'))
    if len(entries) != 27:
        raise ValueError(f'Expected original test27 ArmChair models, got {len(entries)}')
    asset_hashes = {}
    from metrics import sha256
    for meta in entries:
        urdf = meta.parent / 'asset.urdf'
        for mesh in ET.parse(urdf).iter('mesh'):
            assert (urdf.parent / mesh.attrib['filename']).is_file()
        for file in meta.parent.rglob('*'):
            if file.is_file():
                asset_hashes[str(file.relative_to(output))] = sha256(file)
    cfg = {k: copy.deepcopy(saved[k]) for k in ('env', 'sim')}
    env = cfg['env']
    env.update(numAgents=1, numObjects=1, numEnvs=count, episodeLength=900)
    env['eval'].update(skill=['loco'], skillInitProb=[1.])
    env['holdAtFT'].update(asset_split='test', object_root=str(output / 'assets'))
    if public_chair_checkpoint is not None:
        baseline = ROOT.parent / 'TokenHSI/tokenhsi'
        original = output / 'original'
        original.mkdir()
        shared = Path('/home/hwanhee/CVPR2027/TokenHSI/output/tokenhsi')
        shutil.copy2(public_chair_checkpoint, output / 'public_chair.pth')
        shutil.copy2(shared / 'ckpt_stage1.pth', output / 'public_stage1.pth')
        files = dict(human_source='env/tasks/humanoid.py',
                     carry_source='env/tasks/adapt_interaction_skills/humanoid_adapt_carry_box2objs.py',
                     train_config='data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt_disc3.yaml')
        policy = {key: str(original / Path(path).name) for key,path in files.items()}
        for key,path in files.items():
            shutil.copy2(baseline / path, policy[key])
        network_files = ['learning/amp_network_builder.py', 'learning/transformer/composer.py',
            'learning/transformer/amp_network_builder_transformer.py',
            'learning/transformer/amp_network_builder_transformer_adapt.py']
        for path in network_files:
            assert sha256(baseline / path) == sha256(source / 'tokenhsi' / path), path
        policy.update(checkpoint=str(output / 'public_chair.pth'), base_checkpoint=str(output / 'public_stage1.pth'))
        env['holdAtFT']['public_policy'] = policy
        (output / 'public_policy_provenance.json').write_text(json.dumps(dict(
            original_checkpoint=str(Path(public_chair_checkpoint).resolve()),
            checkpoint_sha256=sha256(output / 'public_chair.pth'),
            base_checkpoint_sha256=sha256(output / 'public_stage1.pth'),
            network_original_hashes={path:sha256(baseline/path) for path in network_files}), indent=2))
    env['debug']['reward'] = False
    assert env['enableEarlyTermination'] and env['contactBodies'] == ['right_foot', 'left_foot']
    assert env['holdAtFT']['ground_only'] and env['holdAtFT']['face_human']
    train = copy.deepcopy(saved['train'])
    train['params']['algo']['name'] = 'ma'  # Inference player only; never construct an FT agent.
    train['params']['network']['name'] = 'amp_multi_agent'
    train['params']['load_checkpoint'] = False
    train['params'].pop('load_path', None)
    (output / 'env.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / 'train.yaml').write_text(yaml.safe_dump(train, sort_keys=False))
    # Use exactly the evaluated training's locomotion library, with eval selecting loco only.
    (output / 'asset_sha256.json').write_text(json.dumps(asset_hashes, indent=2))
    (output / 'preparation.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(output / 'checkpoint.pth'), source_epoch=int(weights['epoch']),
        asset_count=len(entries), source=str(source), reward_preserved=True,
        object_input_size=30, config_source=config_source,
        config_only_template=template_checkpoint is not None,
        checkpoint_byte_copy_unchanged=sha256(checkpoint)==sha256(output / 'checkpoint.pth')), indent=2))
    return source

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--object-root', required=True)
    p.add_argument('--num-envs', type=int, default=512)
    p.add_argument('--template-checkpoint')
    p.add_argument('--public-chair-checkpoint')
    a = p.parse_args()
    print(prepare(a.checkpoint, a.output, a.object_root, a.num_envs, a.template_checkpoint, a.public_chair_checkpoint))

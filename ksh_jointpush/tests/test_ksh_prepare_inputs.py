"""Local data migration must preserve the frozen source and fail before copying."""
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from ksh_jointpush.scripts import ksh_jointpush_prepare as prepare


@pytest.fixture
def local_inputs(tmp_path, monkeypatch):
    source = tmp_path / 'ksh_jointpush'
    download = tmp_path / 'download'
    source.mkdir()
    download.mkdir()
    monkeypatch.setattr(prepare, 'SOURCE', source)
    payload = b'original metadata'
    relative = 'dataset_amass_loco/dataset_amass_loco.yaml'
    metadata = download / relative
    metadata.parent.mkdir()
    metadata.write_bytes(payload)
    (download / 'dataset_amass_loco/motions').mkdir()
    manifest = dict(files={'tokenhsi/data/' + relative: hashlib.sha256(payload).hexdigest()},
                    data_links={'tokenhsi/data/dataset_amass_loco/motions': '/old/server/motions'})
    return source, download, manifest


def test_import_preserves_hash_and_links_local_payload(local_inputs):
    source, download, manifest = local_inputs
    prepare.restore_inputs(manifest, download)
    metadata = source / next(iter(manifest['files']))
    assert hashlib.sha256(metadata.read_bytes()).hexdigest() == next(iter(manifest['files'].values()))
    link = source / next(iter(manifest['data_links']))
    assert link.is_symlink() and link.resolve() == download / 'dataset_amass_loco/motions'
    prepare.restore_inputs(manifest, download)  # Repeated preparation is safe.


def test_missing_payload_and_changed_metadata_fail_before_import(local_inputs):
    source, download, manifest = local_inputs
    (download / 'dataset_amass_loco/motions').rmdir()
    with pytest.raises(FileNotFoundError, match='Missing local KSH data'):
        prepare.restore_inputs(manifest, download)
    assert not (source / 'tokenhsi').exists()
    (download / 'dataset_amass_loco/dataset_amass_loco.yaml').write_text('changed')
    with pytest.raises(ValueError, match='Snapshot hash differs'):
        prepare.restore_inputs(manifest, download)
    assert not (source / 'tokenhsi').exists()


def test_relative_and_previous_server_motion_paths_relocate(local_inputs):
    _, download, _ = local_inputs
    motion = download / 'dataset_amass_loco/motions/example.npy'
    motion.write_bytes(b'motion')
    for value in ('dataset_amass_loco/motions/example.npy',
                  'tokenhsi/data/dataset_amass_loco/motions/example.npy',
                  '../../../../CVPR2027/TokenHSI/tokenhsi/data/dataset_amass_loco/motions/example.npy',
                  '/home/hwanhee/CVPR2027/TokenHSI/tokenhsi/data/dataset_amass_loco/motions/example.npy'):
        assert prepare.resolve_motion_path(value, download) == str(motion)
    with pytest.raises(FileNotFoundError, match='Motion payload missing'):
        prepare.resolve_motion_path('dataset_amass_loco/motions/absent.npy', download)


def test_soft_balanced_preparation_uses_local_19500_and_motion_paths(tmp_path, monkeypatch):
    original = prepare.SOURCE
    source = tmp_path / 'ksh_jointpush'
    cfg = source / 'tokenhsi/data/cfg'
    (cfg / 'multi_agent').mkdir(parents=True)
    (cfg / 'train/rlg').mkdir(parents=True)
    for name in ('multi_agent/stage2_jointpush_rsi.yaml', 'multi_agent/stage1_jointpush.yaml',
                 'train/rlg/stage2_jointpush_rsi.yaml'):
        (cfg / name).write_bytes((original / 'tokenhsi/data/cfg' / name).read_bytes())
    for name in ('ksh_goal_cone.py', 'ksh_soft_gate.py', 'ksh_push_only.py', 'ksh_balanced_rsi.py'):
        (source / name).write_bytes((original / name).read_bytes())
    (source / 'pth').mkdir()
    checkpoint = source / 'pth/Stage1PushFinetune_00019500.pth'
    checkpoint.write_bytes(b'checkpoint fixture')
    download = tmp_path / 'download'
    download.mkdir()
    (download / 'motions').mkdir()
    motion = download / 'motions/example.npy'
    motion.write_bytes(b'motion fixture')
    metadata = download / 'dataset_stage1_jointpush.yaml'
    metadata.write_text(yaml.safe_dump(dict(motions=dict(push=[dict(file='motions/example.npy')]))))
    manifest = dict(commit='fixture', files={
        'tokenhsi/data/dataset_stage1_jointpush.yaml': hashlib.sha256(metadata.read_bytes()).hexdigest()},
        data_links={'tokenhsi/data/jointpush_motions': '/old/server/motions'}, checkpoints={})
    (download / 'jointpush_motions').mkdir()
    (source / 'source_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(prepare, 'ROOT', tmp_path)
    monkeypatch.setattr(prepare, 'SOURCE', source)
    monkeypatch.setattr(prepare, 'WORK', tmp_path / 'runs/workspace')
    prepare.prepare(data_root=download)
    work = next((tmp_path / 'runs').iterdir())
    env = yaml.safe_load((work / 'configs/env.yaml').read_text())
    assert env['experiment']['transfer']['checkpoint'] == str(checkpoint)
    spec = env['env']['stage2JointPush']
    assert spec['reward_gate']['mode'] == 'mean_hand_face_quality_v1'
    assert spec['coupled_rsi_selection'] == 'balanced_group_ternary_v1'
    assert set(spec['rsi_group_state_probabilities'].values()) == {1 / 3}
    motions = yaml.safe_load((work / 'configs/motions.yaml').read_text())
    assert motions['motions']['push'][0]['file'] == str(motion)
    report = json.loads((work / 'preparation.json').read_text())
    assert report['source_epoch'] == 19500 and not report['training_started']

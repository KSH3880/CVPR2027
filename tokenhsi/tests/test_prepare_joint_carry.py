import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'prepare_joint_carry', Path(__file__).resolve().parents[1] / 'scripts/prepare_joint_carry.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / 'clone with spaces'
    data = root / 'tokenhsi/data'
    data.mkdir(parents=True)
    (root / prepare.MOTION_FILE).write_text(
        'motions:\n  loco:\n    - file: dataset_amass_loco/motions/walk.npy\n')
    source = tmp_path / 'existing data'
    (source / 'dataset_amass_loco/motions').mkdir(parents=True)
    (source / 'dataset_amass_loco/motions/walk.npy').write_bytes(b'motion')
    asset = data / 'assets/mjcf/phys_humanoid_v3.xml'
    asset.parent.mkdir(parents=True)
    asset.write_text('<mujoco/>')
    bundle = root / 'joint_carry'
    bundle.mkdir()
    (bundle / 'sample.npy').write_bytes(b'bundled')
    (bundle / 'runtime_manifest.json').write_text(json.dumps({'sha256': {
        'joint_carry/sample.npy': hashlib.sha256(b'bundled').hexdigest()}}))
    return root, source


def test_relocated_clone_repairs_broken_links_and_is_idempotent(checkout):
    root, source = checkout
    link = root / 'tokenhsi/data/dataset_amass_loco/motions'
    link.parent.mkdir()
    link.symlink_to('/nonexistent/old/server/motions')
    prepare.connect_data(root, source)
    prepare.connect_data(root, source)
    assert link.resolve() == source / 'dataset_amass_loco/motions'
    assert prepare.check_data(root) == (1, 1)


def test_existing_data_is_not_replaced(checkout):
    root, source = checkout
    destination = root / 'tokenhsi/data/dataset_amass_loco/motions'
    destination.mkdir(parents=True)
    (destination / 'keep').write_text('keep')
    with pytest.raises(ValueError, match='Existing data preserved'):
        prepare.connect_data(root, source)
    assert (destination / 'keep').read_text() == 'keep'


def test_bad_source_does_not_change_broken_link(checkout):
    root, source = checkout
    link = root / 'tokenhsi/data/dataset_amass_loco/motions'
    link.parent.mkdir()
    link.symlink_to('/nonexistent/old/server/motions')
    with pytest.raises(ValueError, match='TOKENHSI_DATA_ROOT'):
        prepare.connect_data(root, source / 'wrong')
    assert os.readlink(link) == '/nonexistent/old/server/motions'


def test_missing_original_and_corrupt_bundle_fail_early(checkout):
    root, source = checkout
    with pytest.raises(ValueError, match='Missing original'):
        prepare.check_data(root)
    prepare.connect_data(root, source)
    (root / 'joint_carry/sample.npy').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Missing or changed bundled'):
        prepare.check_data(root)

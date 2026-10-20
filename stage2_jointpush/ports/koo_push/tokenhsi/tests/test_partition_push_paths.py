from pathlib import Path
import pytest
from utils.partition_push_paths import NAME,checkpoint_path,resolve_stage1_checkpoint


def touch(p):
    p.parent.mkdir(parents=True,exist_ok=True);p.touch();return p.resolve()


def test_local_server_defaults_and_priority(tmp_path):
    server=touch(tmp_path/'runs/jointpush/stage1_sources'/NAME)
    assert resolve_stage1_checkpoint(project_root=tmp_path,environ={})==server
    local=touch(tmp_path/'stage2_jointpush/output'/NAME)
    assert resolve_stage1_checkpoint(project_root=tmp_path,environ={})==local
    alternate=touch(tmp_path/'custom.pth')
    assert resolve_stage1_checkpoint(project_root=tmp_path,environ={'STAGE1_CHECKPOINT':str(alternate)})==alternate
    with pytest.raises(ValueError,match='No Stage1'):
        resolve_stage1_checkpoint(project_root=tmp_path,environ={'STAGE1_CHECKPOINT':str(tmp_path/'missing.pth')})


def test_local_yaml_relative_path_and_explicit_override(tmp_path):
    yaml=tmp_path/'stage2_jointpush/paths.local.yaml';model=touch(yaml.parent/'weights'/NAME)
    yaml.write_text('stage1_checkpoint: weights/'+NAME+'\n')
    assert resolve_stage1_checkpoint(project_root=tmp_path,environ={})==model
    override=touch(tmp_path/'override.pth')
    assert resolve_stage1_checkpoint(str(override),project_root=tmp_path,environ={})==override
    assert resolve_stage1_checkpoint(project_root=tmp_path,environ={'PARTITION_PUSH_PATHS':str(yaml)})==model
    with pytest.raises(ValueError,match='PARTITION_PUSH_PATHS'):
        resolve_stage1_checkpoint(project_root=tmp_path,environ={'PARTITION_PUSH_PATHS':str(tmp_path/'missing.yaml')})


def test_directory_latest_and_ambiguous(tmp_path):
    touch(tmp_path/'first/OpenPushPush180ContactAtWarmstart_00001000.pth')
    latest=touch(tmp_path/'first/OpenPushPush180ContactAtWarmstart_00002000.pth')
    assert checkpoint_path(tmp_path)==latest
    touch(tmp_path/'second/OpenPushPush180ContactAtWarmstart_00002000.pth')
    with pytest.raises(ValueError,match='Multiple Stage1'):checkpoint_path(tmp_path)

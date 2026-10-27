"""Reference source and recipe parity for our-model Table adaptation."""
import ast
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]


def test_reward_and_reset_kernels_are_verbatim_reference():
    source=ast.parse((ROOT.parent/'TokenHSI/tokenhsi/env/tasks/adapt_interaction_skills/humanoid_adapt_carry_box2objs.py').read_text())
    port=ast.parse((ROOT/'tokenhsi/utils/tokenhsi_table_kernels.py').read_text())
    original={v.name:v for v in source.body if isinstance(v,ast.FunctionDef)}
    copied={v.name:v for v in port.body if isinstance(v,ast.FunctionDef)}
    assert len(copied)==7
    for name,node in copied.items():
        assert ast.dump(node,include_attributes=False)==ast.dump(original[name],include_attributes=False),name


def test_ppo_amp_recipe_matches_original_with_our_network():
    base=yaml.safe_load((ROOT.parent/'TokenHSI/tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_adapt_disc3.yaml').read_text())
    port=yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/table_tokenhsi_ft.yaml').read_text())
    a=dict(base['params']['config']);b=dict(port['params']['config']);a.pop('name');b.pop('name');assert a==b
    ours=yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/table_hold_at_full_ft.yaml').read_text())['params']['network']
    network=port['params']['network'];assert network['transformer']==ours['transformer']
    assert network['space']['continuous']['learn_sigma'] is False


def test_reference_environment_and_no_rsi():
    env=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/table_tokenhsi_ft.yaml').read_text())['env']
    assert env['numEnvs']==4096 and env['episodeLength']==600 and env['envSpacing']==5
    assert env['power_reward'] is False and env['box_vel_pen_threshold']==2.
    assert 'tableCarryRsi' not in env
    assert env['holdAtFT']['source_checkpoint']=='checkpoints/Stage1V2Fixed_00078000.pth'
    assert env['holdAtFT']['object_tokenizer_input_size']==30

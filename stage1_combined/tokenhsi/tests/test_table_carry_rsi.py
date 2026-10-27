"""CPU contracts for our unchanged HOLDING_AT setup and audited carry windows."""
from pathlib import Path
import ast
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[2]


def test_relation_reward_and_78k_inputs_unchanged():
    base=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/table_hold_at_full_ft.yaml').read_text())
    rsi=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/table_carry_rsi_full_ft.yaml').read_text())
    assert base['env']['relationReward']==rsi['env']['relationReward']
    assert base['env']['holdAtFT']==rsi['env']['holdAtFT']
    assert base['env']['power_reward']==rsi['env']['power_reward']
    a=yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/table_hold_at_full_ft.yaml').read_text())
    b=yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/table_carry_rsi_full_ft.yaml').read_text())
    assert a['params']['network']==b['params']['network']
    for key in ('learning_rate','disc_reward_scale','task_reward_w','disc_reward_w'):
        assert a['params']['config'][key]==b['params']['config'][key]
    tree=ast.parse((ROOT/'tokenhsi/env/tasks/multi_agent/humanoid_table_carry_rsi_ft.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    assert '_compute_reward' not in [n.name for n in cls.body if isinstance(n,ast.FunctionDef)]


def test_generated_bank_only_contains_screened_candidates():
    cfg=yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/table_carry_rsi_full_ft.yaml').read_text())
    bank=Path(cfg['env']['tableCarryRsi']['bank'])
    import json
    m=json.loads((bank/'manifest.json').read_text())
    assert m['physics_validated'] and m['history_steps']==9 and m['history_dt']==1/30
    for p in bank.glob('*/carryWith.npz'):
        with np.load(p) as d:
            assert d['physics_pass'].all()
            assert (d['hand_error']<=.02).all()
            assert d['history_body'].shape[1:]==(9,15,13)
            assert np.isfinite(d['history_body']).all()
            assert np.isfinite(d['history_dof_vel']).all()
            # A corrected history is nine DIFFERENT reference poses, not endpoint duplication.
            assert np.max(np.abs(d['history_dof'][:,0]-d['history_dof'][:,-1]))>1e-4

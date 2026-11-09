"""Exercise actual resume validation without starting Isaac Gym."""
import ast
import copy
import os
from pathlib import Path
import pytest


def agent_class():
    tree = ast.parse(Path('tokenhsi/learning/push_kd_agent.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'set_full_state_weights')
    class Parent:
        def set_full_state_weights(self, weights):
            self.restored = weights
            self.epoch_num = weights['epoch']
    module = ast.Module(body=[ast.ClassDef(name='Agent', bases=[ast.Name(id='Parent', ctx=ast.Load())], keywords=[], body=[method], decorator_list=[])], type_ignores=[])
    namespace = dict(Parent=Parent, copy=copy, os=os)
    exec(compile(ast.fix_missing_locations(module), '<resume-test>', 'exec'), namespace)
    a = namespace['Agent']();a._kd_coef=.01;a._contract={'kd_coef':.01,'teacher':'same'}
    return a


def test_resume_coefficient_change_requires_opt_in(monkeypatch):
    monkeypatch.delenv('PUSH_KD_ALLOW_COEF_CHANGE', raising=False)
    a=agent_class()
    with pytest.raises(ValueError):
        a.set_full_state_weights({'epoch':6500,'push_kd_contract':{'kd_coef':.001,'teacher':'same'}})


def test_resume_allows_only_coefficient_change(monkeypatch):
    monkeypatch.setenv('PUSH_KD_ALLOW_COEF_CHANGE','1')
    a=agent_class();weights={'epoch':6500,'push_kd_contract':{'kd_coef':.001,'teacher':'same'}}
    a.set_full_state_weights(weights)
    assert a.epoch_num == 6500 and a.restored is weights
    assert weights['push_kd_contract']['kd_coef'] == .001
    weights['push_kd_contract']['teacher']='different'
    with pytest.raises(ValueError):a.set_full_state_weights(weights)

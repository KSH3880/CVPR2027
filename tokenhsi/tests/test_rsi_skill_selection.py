import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sa_helpers import configs
from utils.edge_scenario_spec import agent_object_indices, classify_templates
from utils.mixed_carry_spec import sample_graph


class SelectedSkill(Exception):
    pass


def capture_skill(name, *args):
    raise SelectedSkill(name)


@pytest.mark.parametrize('evaluation, expected', [(False, 'carryWith'), (True, 'loco')])
def test_training_keeps_resolved_skill_and_evaluation_samples_its_distribution(evaluation, expected):
    path = Path(__file__).resolve().parents[1] / 'env/tasks/multi_agent/humanoid_ma_carry.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'HumanoidMACarry')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == '_reset_ref_state_init')
    namespace = dict(torch=torch, classify_templates=classify_templates,
                     agent_object_indices=agent_object_indices)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    env, _ = configs()
    skills = env['skill']
    graph = sample_graph(1, env['relationGraph'], preset='independent_at_at')
    requested = torch.zeros(len(skills))
    requested[skills.index('carryWith')] = 1
    eval_weights = torch.zeros(len(skills))
    eval_weights[skills.index('loco')] = 1
    task = SimpleNamespace(
        _template_rsi=env['templateRsi'], _is_eval=evaluation,
        relation_runtime=SimpleNamespace(graph=graph), _scenario_with_climb=True,
        _template_rsi_weights=torch.ones(8, len(skills)),
        cfg={'env': {'independentCarryRsi': requested.tolist()}},
        num_agents=2, device='cpu', _skill=skills, _skill_init_prob=eval_weights,
        _size_aware_rsi=True,
        _size_rsi_cache=SimpleNamespace(
            availability=lambda *args: torch.ones(2, len(skills), dtype=torch.bool),
            sample=capture_skill),
        _size_rsi_requested=torch.zeros(8, len(skills)), _size_rsi_fallbacks=torch.zeros(8),
        _motion_lib={name: SimpleNamespace(
            sample_motions=lambda n, name=name: capture_skill(name)) for name in skills},
        _agent_spawn_offsets=torch.zeros(2, 3), _env_origins=torch.zeros(1, 3),
    )
    with pytest.raises(SelectedSkill) as selected:
        namespace['_reset_ref_state_init'](task, torch.tensor([0, 0]), torch.tensor([0, 1]))
    assert selected.value.args[0] == expected

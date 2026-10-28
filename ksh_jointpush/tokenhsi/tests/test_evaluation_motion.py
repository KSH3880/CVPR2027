from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from utils.evaluation_motion import active_eval_skills, push_eval_skills
from utils.stage1_push import load_bank

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT/'tokenhsi/data/dataset_stage1_push.yaml'
MOTIONS = yaml.safe_load(MANIFEST.read_text())['motions']
AVAILABLE = list(MOTIONS)


@pytest.mark.parametrize('skills,weights,has_push,has_old,expected', [
    (['loco'], [1.], True, False, ['push_loco']),
    (['push'], [1.], True, False, ['push']),
    (['loco','push'], [.5,.5], True, False, ['push','push_loco']),
    (['loco','push'], [1.,0.], True, False, ['push_loco']),
    (['sit'], [1.], False, True, ['sit']),
    (['climb'], [1.], False, True, ['climb']),
    (['loco','carryWith'], [.5,.5], False, True, ['carryWith','loco']),
    (['loco'], [1.], True, True, ['loco','push_loco']),
])
def test_only_start_libraries_are_selected(skills, weights, has_push, has_old, expected):
    config = dict(skill=skills, skillInitProb=weights)
    selected = push_eval_skills(AVAILABLE, config, has_push, has_old)
    assert selected == [name for name in AVAILABLE if name in expected]


def test_training_keeps_every_library_and_zero_weight_eval_skips_loading():
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    from env.tasks.multi_agent.humanoid_ma_stage1_push import HumanoidMAStage1Push
    stub = SimpleNamespace(cfg={'args':SimpleNamespace(eval=False)})
    assert HumanoidMACarry._motion_skills_to_load(stub, AVAILABLE) is AVAILABLE
    assert HumanoidMAStage1Push._motion_skills_to_load(stub, AVAILABLE) is AVAILABLE
    stub.cfg = {'args':SimpleNamespace(eval=True), 'env':{'eval':dict(skill=['loco','sit'], skillInitProb=[1.,0.])}}
    assert HumanoidMACarry._motion_skills_to_load(stub, AVAILABLE) == ['loco']


def test_push_rsi_subset_preserves_original_motion_ids_and_hashes(tmp_path):
    from env.tasks.multi_agent.push_rsi import EvalPushMotionLib
    env = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_agent/stage1_v2_fixed_push.yaml').read_text())['env']
    bank = load_bank(env['stage1Push'], MANIFEST)
    ids = sorted({row['motion_id'] for row in bank['candidates']['push']})
    assert ids == [0,1,2,5,6,19,20]
    lib = object.__new__(EvalPushMotionLib)
    lib.source_ids = ids
    files, weights, skipped, objects = lib._fetch_motion_files(str(MANIFEST), 'push')
    assert len(files) == len(weights) == len(skipped) == 7
    assert not objects
    assert [Path(path).resolve() for path in files] == [(MANIFEST.parent/MOTIONS['push'][i]['file']).resolve() for i in ids]
    subset_manifest = deepcopy(MOTIONS)
    for index, row in enumerate(subset_manifest['push']):
        row['file'] = str((MANIFEST.parent/row['file']).resolve()) if index in ids else 'unused_missing_motion.npy'
    path = tmp_path/'motions.yaml'
    path.write_text(yaml.safe_dump({'motions':subset_manifest}))
    checked = load_bank(env['stage1Push'], path, ids)
    remapped = [ids.index(row['motion_id']) for row in checked['candidates']['push']]
    assert [ids[index] for index in remapped] == [row['motion_id'] for row in bank['candidates']['push']]


def test_incompatible_task_start_is_rejected():
    with pytest.raises(ValueError, match='PUSH RSI cannot initialize'):
        push_eval_skills(AVAILABLE, dict(skill=['push'], skillInitProb=[1.]), False, True)
    with pytest.raises(ValueError, match='support loco and push'):
        push_eval_skills(AVAILABLE, dict(skill=['sit'], skillInitProb=[1.]), True, False)

from copy import deepcopy
from pathlib import Path
import math
import pytest
import torch
import yaml
from utils.stage1_push_scene import sample_four_box_scene, resample_uniform_rsi_goals
from utils.push_scene import forward_xy, random_at_goal_valid
from utils.stage1_push_spec import validate_env

DATA = Path(__file__).resolve().parents[1]/'data'
ENV = yaml.safe_load((DATA/'cfg/multi_agent/stage1_push_finetune.yaml').read_text())['env']
SOURCE = yaml.safe_load((DATA/'cfg/multi_agent/stage1_push_finetune_source.yaml').read_text())['env']

@pytest.mark.parametrize('seed', [42, 7, 20261026])
def test_rsi_bounds_cone_fixed_boxes_loco_goals_and_path_clearance(seed):
    g = torch.Generator().manual_seed(seed)
    xy, q, goal = sample_four_box_scene(512, SOURCE, 'cpu', g)
    plan = torch.zeros(512, 4, 13)
    plan[..., :2], plan[..., 3:7] = xy, q
    before = plan.clone()
    active = torch.rand(512, 2, generator=g) < .5
    out = resample_uniform_rsi_goals(plan, goal, active, SOURCE, g, [2., 10.])
    assert torch.equal(plan, before)
    assert torch.equal(out[~active], goal[~active])
    distance = (out-xy[:, :2]).norm(dim=-1)
    assert distance[active].min() >= 2.-1e-5
    assert distance[active].max() <= 10.+1e-5
    assert ((out-xy[:, :2])/distance[..., None]*forward_xy(q[:, :2])).sum(-1)[active].min() >= math.cos(math.pi/4)-1e-6
    assert random_at_goal_valid(xy[:, :2], q[:, :2], out, SOURCE, active, [2., 10.]).all()
    assert distance[active].min() < 2.1 and distance[active].max() > 9.8
    assert ((out[active].abs()>4.5).any(-1)).any()  # no legacy square clamp for RSI


def test_no_rsi_preserves_scene_and_random_generator():
    g = torch.Generator().manual_seed(3)
    xy, q, goal = sample_four_box_scene(4, SOURCE, 'cpu', g)
    plan = torch.zeros(4, 4, 13);plan[..., :2], plan[..., 3:7] = xy, q
    state = g.get_state().clone()
    out = resample_uniform_rsi_goals(plan, goal, torch.zeros(4, 2, dtype=torch.bool), SOURCE, g, [2.,10.])
    assert torch.equal(out, goal) and torch.equal(state, g.get_state())


def test_explicit_contract_distinguishes_uniform_from_old_rsi():
    validate_env(ENV)
    bad = deepcopy(ENV);bad['openPushRSI']['goal_distance_range'] = [2.,12.]
    with pytest.raises(ValueError): validate_env(bad)
    assert ENV['pushBox'] == SOURCE['pushBox']
    assert ENV['openPushRSI']['interaction_probability'] == .5


def test_shareable_named_config_preserves_baseline_settings():
    new = yaml.safe_load((DATA/'cfg/multi_agent/stage1_push_rsi_uniform.yaml').read_text())
    named = deepcopy(new['env'])
    assert named.pop('episodeLength') == 1200
    old = deepcopy(ENV);old.pop('episodeLength')
    assert named['pushExtension'].pop('distractor_mode') == 'stage1_support_v1'
    named['pushExtension']['source_env'] = old['pushExtension']['source_env']
    assert named == old
    source = yaml.safe_load((DATA/'cfg/multi_agent/stage1_push_rsi_uniform_source.yaml').read_text())['env']
    assert source == SOURCE
    old_train = yaml.safe_load((DATA/'cfg/train/rlg/stage1_push_finetune.yaml').read_text())
    new_train = yaml.safe_load((DATA/'cfg/train/rlg/stage1_push_rsi_uniform.yaml').read_text())
    new_train['params']['config']['name'] = old_train['params']['config']['name']
    assert new_train == old_train
    validate_env(new['env'])


def test_stage1_support_sizes_and_push_payloads_are_separate():
    from utils.stage1_push_spec import sample_asset_sizes
    from utils.size_rsi import STAGE1_V2_RANGES
    env = yaml.safe_load((DATA/'cfg/multi_agent/stage1_push_rsi_uniform.yaml').read_text())['env']
    torch.manual_seed(123)
    sizes, posterior = sample_asset_sizes(10000, env, 'cpu')
    bounds = torch.tensor(STAGE1_V2_RANGES['SUPPORT'])
    assert ((sizes[:,2:] >= bounds[:,0]) & (sizes[:,2:] <= bounds[:,1])).all()
    push = posterior[...,5] > 0
    assert (sizes[:,:2][push] >= sizes.new_tensor([1.6,1.6,1.45])).all()
    assert (sizes[:,2:,2] < 1.).all()
    assert len(env['pushBox']['size_profiles']) == 167
    torch.testing.assert_close(posterior.mean((0,1)),torch.tensor([0.,.05,.125,.1625,.1625,.5]),atol=.012,rtol=0)


def test_distractor_path_check_uses_actual_small_support_radius():
    from utils.stage1_push_scene import support_push_paths_clear
    boxes = torch.zeros(1,4,13);sizes = torch.tensor([[[2.,2.,1.5],[2.,2.,1.5],[.4,.4,.25],[.4,.4,.25]]])
    boxes[0,0,:2] = torch.tensor([-4.,0.]);boxes[0,1,:2] = torch.tensor([4.,0.])
    boxes[0,2,:2] = torch.tensor([-4.,3.]);boxes[0,3,:2] = torch.tensor([0.,0.])
    goals = torch.tensor([[[-4.,6.,.75],[4.,6.,.75]]])
    assert support_push_paths_clear(boxes,sizes,goals,torch.ones(1,2,dtype=torch.bool)).tolist() == [[False,True]]
    assert support_push_paths_clear(boxes,sizes,goals,torch.zeros(1,2,dtype=torch.bool)).all()


def test_impossible_uniform_goal_returns_failure_for_whole_scene_retry():
    g = torch.Generator().manual_seed(4)
    xy, q, goal = sample_four_box_scene(1, SOURCE, 'cpu', g)
    plan = torch.zeros(1,4,13);plan[..., :2], plan[..., 3:7] = xy, q
    impossible = deepcopy(SOURCE);impossible['pushScene']['path_separation'] = 1e6
    out, failed = resample_uniform_rsi_goals(plan, goal, torch.ones(1,2,dtype=torch.bool),
        impossible, g, [2.,10.],check_spares=False,return_failures=True)
    assert failed.tolist() == [True]
    assert torch.equal(out,goal)

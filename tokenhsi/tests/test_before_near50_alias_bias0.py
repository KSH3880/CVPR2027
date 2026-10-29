from copy import deepcopy

import pytest
import torch

from test_before_adapter import build, config
from test_before_object_alias import packet, observations
from test_before_object_alias import test_stage1_transfer_frozen_encoder_head_updates_and_reload as _check_transfer
from utils.before_relative_sampling import DISTANCE_MIXTURE
from utils.before_spec import validate_env, checkpoint_contract
from utils.stage2_spec import check_checkpoint
from learning.multi_agent.stage2_network_builder import freeze_actor

NAME = 'stage2_before_ground_atprogress_bias0_near50_objalias_1200'
BASE = 'stage2_before_ground_mix226_atprogress_objalias_1200'


def test_loco_alias_changes_only_distances_and_both_bias_modes():
    old, new = config(BASE), config(NAME)
    validate_env(new['env'])
    plain = deepcopy(new['env'])
    for side in ('actor', 'critic'):
        assert plain['relationGraph'].pop(side+'_human_bias') == 'before_only'
    assert plain['before']['distance_mixture'] == DISTANCE_MIXTURE
    plain['before']['distance_mixture'] = old['env']['before']['distance_mixture']
    assert plain == old['env'] and new['sim'] == old['sim']
    assert new['experiment']['transfer'] == old['experiment']['transfer']
    a, b = config(BASE, 'train/rlg'), config(NAME, 'train/rlg')
    b['params']['config']['name'] = a['params']['config']['name']
    assert a == b
    others = [BASE, 'stage2_before_ground_atprogress_nonemask_rsi433_near50_objalias_1200',
              'stage2_before_ground_mix226_atprogress_bias0_1200']
    for other in others:
        for src, dst in ((config(other), new), (new, config(other))):
            with pytest.raises(ValueError, match='BEFORE checkpoint'):
                check_checkpoint({'before_contract': checkpoint_contract(src['env'])}, dst['env'])


def test_both_sides_keep_none_visible_and_only_train_before_bias():
    torch.set_num_threads(1)
    net = build(NAME)
    model = torch.nn.Module(); model.a2c_network = net; freeze_actor(model)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    seen = []

    def capture(module, args, kwargs):
        mask = kwargs['attn_mask']
        assert torch.isfinite(mask).all()  # Includes NONE keys for all four humans.
        seen.append(mask.detach().clone())

    for branch in (net.coordination, net.critic_coordination):
        for block in branch.blocks:
            assert block.bias_mode == 'before_only' and block.relation_bias.shape == (2, 1)
            block.attention.register_forward_pre_hook(capture, with_kwargs=True)
    obs = observations(packet())
    for step in range(3):
        opt.zero_grad()
        (net.eval_actor(obs)[0].square().mean()+net.eval_critic(obs).square().mean()).backward()
        for branch in (net.coordination, net.critic_coordination):
            for block in branch.blocks:
                assert torch.count_nonzero(block.effective_relation_bias()[:, :2]) == 0
                # The coordination output projection starts at zero.
                if step:
                    assert block.relation_bias.grad.abs().sum() > 0
                    assert block.attention.in_proj_weight.grad.abs().sum() > 0
        opt.step()
    assert len(seen) == 24
    for branch in (net.coordination, net.critic_coordination):
        for block in branch.blocks:
            assert torch.count_nonzero(block.effective_relation_bias()[:, :2]) == 0
            assert block.relation_bias.abs().sum() > 0


def test_transfer_freeze_head_update_and_reload():
    _check_transfer(BASE, NAME)

from copy import deepcopy

import pytest
import torch

from learning.amp_models import ModelAMPContinuous
from learning.multi_agent.grasp_view import initial_grasp_sides, make_grasp_views
from learning.multi_agent.scene_normalizer import SceneRunningMeanStd
from sa_helpers import network as pair_network, joint_observations
from test_group_carry import network as group_network, scene


@pytest.mark.parametrize('humans', [2, 4])
@pytest.mark.parametrize('grasp', [False, True])
def test_cached_actor_matches_updates_and_shuffles(humans, grasp):
    torch.manual_seed(271)
    if humans == 4:
        net = group_network(grasp)
        obs = torch.cat([scene(preset, n=3)[0] for preset in
                         ('all_solo_at', 'mixed_groups', 'two_pairs_ontop')])
    else:
        net = pair_network('stage2_grasp' if grasp else 'stage2').a2c_network
        obs = joint_observations(9)
    for name, parameter in net.named_parameters():
        if 'relation_bias' in name:
            torch.nn.init.uniform_(parameter, -.2, .2)
    net.actor_encoder.requires_grad_(False)
    if grasp:
        obs = make_grasp_views(obs, initial_grasp_sides(obs, humans, humans*2), .15)
    rms = SceneRunningMeanStd([223, 30, 1], [humans, humans*2, humans],
        [223, 30, 0], kinematic_size=7, extra_passthrough_size=5*humans,
        num_views=humans if grasp else 1).eval()
    obs = rms(obs)
    direct = ModelAMPContinuous.Network(net)
    cached = deepcopy(direct)
    features = torch.cat([cached.a2c_network.encode_actor(x) for x in obs.split(3)])
    assert not features.requires_grad
    encoder_before = deepcopy(cached.a2c_network.actor_encoder.state_dict())
    critic_before = deepcopy(cached.a2c_network.critic_encoder.state_dict())
    optimizers = [torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=2e-5)
                  for m in (direct, cached)]
    actions = torch.randn(len(obs)*humans, 32)*.1
    amp = torch.randn(8, 1320)*.1
    for iteration in range(6):
        order = torch.randperm(len(obs))
        inputs = dict(is_train=True, obs=obs[order],
                      prev_actions=actions.reshape(len(obs), humans, 32)[order].flatten(0, 1),
                      amp_obs=amp, amp_obs_replay=amp, amp_obs_demo=amp)
        outputs = []
        for model, optimizer, use_cache in zip((direct, cached), optimizers, (False, True)):
            optimizer.zero_grad(set_to_none=True)
            output = model(dict(inputs, **({'actor_features': features[order]} if use_cache else {})))
            loss = (output['prev_neglogp'].mean()*.01 + output['values'].square().mean()
                    + output['disc_demo_logit'].square().mean())
            loss.backward()
            outputs.append(output)
        for key in ('mus', 'sigmas', 'values', 'prev_neglogp', 'disc_demo_logit'):
            atol = 1e-4 if key == 'prev_neglogp' else 3e-5
            torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=atol, rtol=3e-5)
        for (name, first), (_, second) in zip(direct.named_parameters(), cached.named_parameters()):
            assert (first.grad is None) == (second.grad is None), name
            if first.grad is not None:
                torch.testing.assert_close(first.grad, second.grad, atol=3e-5, rtol=3e-5)
        for optimizer in optimizers:
            optimizer.step()
    for key, value in cached.a2c_network.actor_encoder.state_dict().items():
        torch.testing.assert_close(value, encoder_before[key], atol=0, rtol=0)
    assert any(not torch.equal(value, critic_before[key])
               for key, value in cached.a2c_network.critic_encoder.state_dict().items())

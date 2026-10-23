"""The private fine-tune updates only the existing object tokenizer/critic/disc."""
import importlib.util
from pathlib import Path
import pytest
import torch
from torch import nn

SPEC = importlib.util.spec_from_file_location('ft_contract',
    Path(__file__).parents[2] / 'ports/ksh_object_finetune/finetune_contract.py')
FT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FT)


def model():
    m=nn.Module();m.a2c_network=nn.Module();net=m.a2c_network
    net.actor_encoder=nn.Module()
    net.actor_encoder.tokenizers=nn.ModuleList([nn.Linear(3,3),nn.Linear(30,3),nn.Linear(1,3)])
    net.actor_encoder.backbone=nn.Linear(3,3)
    net.action_head=nn.Linear(3,2)
    net.critic_encoder=nn.Linear(3,3);net.value_head=nn.Linear(3,1)
    net._disc_mlp=nn.Linear(3,3);net._disc_logits=nn.Linear(3,1)
    return m


def test_optimizer_steps_preserve_every_frozen_actor_parameter():
    m=model();before={n:p.detach().clone() for n,p in m.named_parameters()}
    FT.freeze_actor(m);frozen=FT.frozen_snapshot(m)
    optimizer=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        # Nonzero gradient for every eligible parameter makes this detect an
        # accidentally unfrozen backbone/action head, rather than unused weights.
        loss=sum((p-7).square().sum() for p in m.parameters())
        loss.backward();optimizer.step()
        FT.verify_frozen(m,frozen)
    changed={n for n,p in m.named_parameters() if not torch.equal(p,before[n])}
    assert changed=={n for n in before if FT.trainable_name(n)}
    with torch.no_grad():m.a2c_network.action_head.weight.add_(1)
    with pytest.raises(RuntimeError,match='Frozen actor'):FT.verify_frozen(m,frozen)


def test_new_tokenizer_or_adapter_is_not_added():
    m=model();keys=set(m.state_dict());FT.freeze_actor(m)
    assert set(m.state_dict())==keys
    assert len(m.a2c_network.actor_encoder.tokenizers)==3


@pytest.mark.parametrize('mode', ['adapter', 'head'])
def test_adaptation_modes_update_exactly_the_selected_weights(mode):
    spec=importlib.util.spec_from_file_location('action_adaptation',
        Path(__file__).parents[2]/'ports/ksh_object_finetune/action_adaptation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    m=model();net=m.a2c_network
    net.action_head=nn.Sequential(nn.Sequential(nn.Linear(3,5),nn.ReLU()),nn.Linear(5,2))
    x=torch.randn(7,3)
    before_action=net.action_head(x).detach()
    if mode=='adapter':
        net.action_adapter=module.make_adapters(net.action_head)
        assert torch.equal(module.adapted_action(net.action_head,net.action_adapter,x),before_action)
    FT.freeze_actor(m,mode);frozen=FT.frozen_snapshot(m)
    before={n:p.detach().clone() for n,p in m.named_parameters()}
    optimizer=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        # Exercise the real policy path so frozen head layers must propagate
        # gradients to the tokenizer and all residual adapters.
        z=net.actor_encoder.tokenizers[1](torch.randn(7,30))
        action=(module.adapted_action(net.action_head,net.action_adapter,z)
                if mode=='adapter' else net.action_head(z))
        action.square().sum().backward();optimizer.step()
        FT.verify_frozen(m,frozen)
    changed={n for n,p in m.named_parameters() if not torch.equal(p,before[n])}
    assert any('actor_encoder.tokenizers.1.' in n for n in changed)
    assert any(('action_adapter.' if mode=='adapter' else 'action_head.') in n for n in changed)
    assert all(FT.trainable_name(n,mode) for n in changed)
    if mode=='adapter':assert not any('action_head.' in n for n in changed)
    else:assert not hasattr(net,'action_adapter')


def test_copied_tokenizer_preserves_base_policy_after_training_and_reload():
    import io
    base=Path(__file__).parents[2]/'ports/ksh_object_finetune'
    def load(name):
        spec=importlib.util.spec_from_file_location(name,base/(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    copy_module=load('object_tokenizer_copy');adaptation=load('action_adaptation')
    m=model();net=m.a2c_network
    net.action_head=nn.Sequential(nn.Sequential(nn.Linear(3,5),nn.ReLU()),nn.Linear(5,2))
    copy_module.install_copy(net.actor_encoder)
    # Simulate loading source weights after construction and syncing the new copy.
    with torch.no_grad():net.actor_encoder.tokenizers[1].weight.add_(.3)
    copy_module.sync_copy(net.actor_encoder)
    net.action_adapter=adaptation.make_adapters(net.action_head)
    def action(x):
        encoder=net.actor_encoder
        tokenizer=copy_module.select_tokenizer(encoder,1,encoder.tokenizers[1])
        z=tokenizer(x)
        return (adaptation.adapted_action(net.action_head,net.action_adapter,z)
                if net.use_action_adapter else net.action_head(z))
    x=torch.randn(16,30)
    copy_module.set_branch(net,'base');original=action(x).detach().clone()
    copy_module.set_branch(net,'adapt');assert torch.equal(action(x),original)
    FT.freeze_actor(m,'copied_adapter');frozen=FT.frozen_snapshot(m)
    optimizer=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.01)
    for _ in range(5):
        optimizer.zero_grad();(action(x)-3).square().sum().backward();optimizer.step()
    adapted=action(x).detach().clone();assert not torch.equal(adapted,original)
    FT.verify_frozen(m,frozen)
    copy_module.set_branch(net,'base');assert torch.equal(action(x),original)
    assert not any(p.requires_grad for p in net.actor_encoder.tokenizers[1].parameters())
    assert all(p.requires_grad for p in net.actor_encoder.adapted_object_tokenizer.parameters())
    stream=io.BytesIO();torch.save(m.state_dict(),stream);stream.seek(0)
    m.load_state_dict(torch.load(stream,weights_only=True))
    copy_module.set_branch(net,'adapt');assert torch.equal(action(x),adapted)
    copy_module.set_branch(net,'base');assert torch.equal(action(x),original)

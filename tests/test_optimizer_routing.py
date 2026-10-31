"""Execute the actual training method on CPU, without loading Isaac Gym."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from learning.push_kd_losses import gaussian_forward_kl, masked_mean


def test_actual_optimizer_step_only_updates_push_critic_and_old_kd():
    path = Path(__file__).resolve().parents[1]/'tokenhsi/learning/push_kd_agent.py'
    cls = next(x for x in ast.parse(path.read_text()).body if isinstance(x, ast.ClassDef))
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'calc_gradients')
    namespace = dict(torch=torch, nn=nn, masked_mean=masked_mean, gaussian_forward_kl=gaussian_forward_kl,
                     torch_ext=SimpleNamespace(policy_kl=lambda *args: torch.tensor(0.)))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.mu = nn.Parameter(torch.ones(8, 32)*.1)
            self.value = nn.Parameter(torch.ones(8, 1))
            self.disc = nn.Parameter(torch.ones(1))
        def forward(self, data):
            # AMP inputs must always carry PUSH=1 even for mixed PPO minibatches.
            assert (data['amp_obs'][:, 133] == 1).all()
            assert (data['amp_obs_replay'][:, 133] == 1).all()
            assert (data['amp_obs_demo'][:, 133] == 1).all()
            logits = self.disc.expand(len(data['amp_obs']), 1)
            return dict(prev_neglogp=self.mu.sum(-1), values=self.value,
                        entropy=self.mu.sum(-1)*0, mus=self.mu, sigmas=torch.ones_like(self.mu),
                        disc_agent_logit=logits, disc_agent_replay_logit=logits, disc_demo_logit=logits)
    model = Toy()
    actor = SimpleNamespace(model=model, set_train=lambda: None, _preproc_obs=lambda x: x,
        _preproc_amp_obs=lambda x: x, _amp_minibatch_size=4, last_lr=.001, e_clip=.2,
        mixed_precision=False, is_rnn=False, critic_coef=1., entropy_coef=0.,
        bounds_loss_coef=0., _disc_coef=1., _reg_loss_enabled=False, _kd_coef=.001,
        multi_gpu=False, truncate_grads=False, clip_value=False,
        optimizer=torch.optim.Adam(model.parameters(), lr=.01),
        scaler=torch.cuda.amp.GradScaler(enabled=False))
    actor._actor_loss = lambda old, new, adv, clip: dict(actor_loss=new*adv, actor_clipped=torch.zeros_like(new))
    actor._critic_loss = lambda old, new, clip, ret, enabled: dict(critic_loss=(new-ret).square())
    actor.bound_loss = lambda mu: mu.square().sum(-1)
    actor._disc_loss = lambda a,b,c: dict(disc_loss=a.square().mean())
    amp = torch.zeros(8, 1340); amp[:, 133] = 1
    actor._sample_push_amp = lambda n: amp[:n]
    push = torch.arange(8) < 4
    kd = (torch.arange(8) >= 4) & (torch.arange(8) < 6)
    teacher_mu = torch.zeros(8, 32, requires_grad=True)
    data = dict(old_values=torch.zeros(8,1), old_logp_actions=torch.zeros(8), advantages=torch.ones(8),
                mu=torch.zeros(8,32), sigma=torch.ones(8,32), returns=torch.zeros(8,1),
                actions=torch.zeros(8,32), obs=torch.zeros(8,397), amp_obs=amp,
                amp_obs_replay=amp, amp_obs_demo=amp, rand_action_mask=push.float(),
                push_mask=push, kd_mask=kd, teacher_mu=teacher_mu, teacher_sigma=torch.ones(8,32))
    namespace['calc_gradients'](actor, data)
    assert torch.equal(model.value[~push].detach(), torch.ones(4,1))
    assert (model.value[push].detach() < 1).all()
    assert (model.mu[kd].detach() < .1).all()
    assert torch.equal(model.mu[~push & ~kd].detach(), torch.full((2,32), .1))
    assert teacher_mu.grad is None
    assert all(torch.isfinite(p).all() for p in model.parameters())

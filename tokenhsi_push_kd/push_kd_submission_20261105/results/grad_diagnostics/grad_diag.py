"""Gradient-conflict diagnostic for a PUSH KD checkpoint (no learning).

Resumes a checkpoint through the normal training entry point with every optimizer lr forced to 0, so the
policy, critic and discriminator stay at the checkpoint. After DIAG_WARMUP_EPOCHS rollout epochs (to move away
from the reset distribution), it measures, on the first mini-epoch of DIAG_MEASURE_EPOCHS epochs, the gradient
of each loss term that the training loss adds up:
  ppo        PUSH actor loss (+ bounds, entropy) exactly as in calc_gradients
  kd_<task>  lambda_k * (n_k / n) * mean KL_k, the term of task k inside lagrange_kd_loss
and reports cosine similarities and norms of the summed gradients per parameter group.
Lagrange coefficients are held at the checkpoint values.
Usage: python scripts/grad_diag.py <run.py args...>   (env: DIAG_OUT, DIAG_WARMUP_EPOCHS, DIAG_MEASURE_EPOCHS)
"""
import json
import os
import runpy
import sys

WARM = int(os.environ.get('DIAG_WARMUP_EPOCHS', 40))
MEAS = int(os.environ.get('DIAG_MEASURE_EPOCHS', 3))
OUT = os.environ.get('DIAG_OUT', 'grad_diag.json')


def main():
    from isaacgym import gymapi  # noqa: F401  (Isaac Gym must be imported before torch)
    import torch
    import learning.push_kd_agent as A
    from learning.push_kd_losses import gaussian_forward_kl_per_sample

    st = dict(epoch=0, call=0, lam=None, names=None, sums={}, halves=[{}, {}], mb=[], counts=[])
    orig_train_epoch, orig_calc = A.PushKDAgent.train_epoch, A.PushKDAgent.calc_gradients

    def add(acc, key, grads):
        if key not in acc:
            acc[key] = [g.detach().clone() for g in grads]
        else:
            for a, g in zip(acc[key], grads):
                a.add_(g)

    def flat(grads, idx):
        return torch.cat([grads[i].reshape(-1) for i in idx]).double()

    def cos(a, b):
        na, nb = a.norm(), b.norm()
        return float((a @ b) / (na * nb)) if na > 0 and nb > 0 else float('nan')

    def measure(self, d):
        if st['lam'] is None:
            st['lam'] = self._kd_lambda.detach().clone()
            st['names'] = [n for n, p in self.model.named_parameters() if p.requires_grad]
        params = [p for p in self.model.parameters() if p.requires_grad]
        rms = (self.running_mean_std, self._amp_input_mean_std)
        modes = [m.training for m in rms]
        for m in rms:
            m.eval()  # the original calc_gradients call below does the usual statistics update once
        self.set_train()
        obs = self._preproc_obs(d['obs'])
        amp = self._preproc_amp_obs(self._sample_push_amp(self._amp_minibatch_size))
        demo, replay = self._amp_minibatch()
        batch = {'is_train': True, 'prev_actions': d['actions'], 'obs': obs, 'not_normalized_obs': d['obs'],
                 'amp_obs': amp, 'amp_obs_replay': self._preproc_amp_obs(replay),
                 'amp_obs_demo': self._preproc_amp_obs(demo)}
        for m, mode in zip(rms, modes):
            m.train(mode)
        res = self.model(batch)
        mu = res['mus']
        sigma = res['sigmas'].expand_as(mu)
        mask = d['rand_action_mask']
        total = mask.sum().clamp_min(1)
        a = self._actor_loss(d['old_logp_actions'], res['prev_neglogp'], d['advantages'], self.e_clip)['actor_loss']
        losses = {'ppo': (mask * a).sum() / total - self.entropy_coef * (mask * res['entropy']).sum() / total
                  + self.bounds_loss_coef * (mask * self.bound_loss(mu)).sum() / total}
        old = d['kd_mask']
        per = gaussian_forward_kl_per_sample(mu[old], sigma[old], d['teacher_mu'][old], d['teacher_sigma'][old])
        task = d['obs'][old][:, A.KD_TASK_ONEHOT].argmax(-1)
        n = max(len(per), 1)
        counts = dict(push=int(d['push_mask'].sum()), kd=len(per))
        for k, name in enumerate(A.KD_TASKS):
            sel = task == k
            counts[name] = int(sel.sum())
            counts['kl_' + name] = float(per[sel].mean()) if sel.any() else float('nan')
            losses['kd_' + name] = st['lam'][k] * per[sel].sum() / n
        st['counts'].append(counts)
        grads = {}
        keys = list(losses)
        for i, key in enumerate(keys):
            g = torch.autograd.grad(losses[key], params, retain_graph=i < len(keys) - 1, allow_unused=True)
            grads[key] = [torch.zeros_like(p) if x is None else x for p, x in zip(params, g)]
            add(st['sums'], key, grads[key])
            add(st['halves'][st['call'] % 2], key, grads[key])
        idx = range(len(params))
        f = {k: flat(v, idx) for k, v in grads.items()}
        st['mb'].append({k: cos(f['ppo'], f[k]) for k in keys if k != 'ppo'})

    def calc_gradients(self, input_dict):
        if st['epoch'] > WARM and st['call'] < len(self.dataset):
            measure(self, input_dict)
        st['call'] += 1
        for group in self.optimizer.param_groups:
            group['lr'] = 0.0  # keep every parameter at the checkpoint
        return orig_calc(self, input_dict)

    def train_epoch(self):
        st['epoch'] += 1
        st['call'] = 0
        keep = self._kd_lambda.detach().clone()
        info = orig_train_epoch(self)
        self._kd_lambda = keep  # hold the Lagrange coefficients at the checkpoint
        print('GRAD_DIAG epoch {} ({} warmup, {} measure)'.format(st['epoch'], WARM, MEAS), flush=True)
        if st['epoch'] >= WARM + MEAS:
            finish()
            os._exit(0)
        return info

    def finish():
        names = st['names']
        groups = {'all': list(range(len(names)))}
        for depth in (2, 3):
            for i, name in enumerate(names):
                groups.setdefault('.'.join(name.split('.')[:depth]), []).append(i)
        keys = list(st['sums'])
        result = dict(lambda_=st['lam'].tolist(), warmup_epochs=WARM, measure_epochs=MEAS,
                      minibatches=len(st['mb']), counts=st['counts'], groups={})
        for gname, idx in groups.items():
            f = {k: flat(st['sums'][k], idx) for k in keys}
            h = [{k: flat(half[k], idx) for k in keys} for half in st['halves']]
            norms = {k: float(v.norm()) for k, v in f.items()}
            if all(v == 0 for v in norms.values()):
                continue
            kd = sum(f[k] for k in keys if k != 'ppo')
            result['groups'][gname] = dict(
                numel=int(f['ppo'].numel()), norm=norms,
                cos_ppo={k: cos(f['ppo'], f[k]) for k in keys if k != 'ppo'},
                cos_ppo_kd_total=cos(f['ppo'], kd),
                cos_kd_pairs={a + '|' + b: cos(f[a], f[b]) for i, a in enumerate(keys) for b in keys[i + 1:]
                              if a != 'ppo' and b != 'ppo'},
                half_consistency={k: cos(h[0][k], h[1][k]) for k in keys})
        mb = st['mb']
        result['minibatch_cos_ppo_mean'] = {k: sum(x[k] for x in mb) / len(mb) for k in mb[0]}
        result['param_names'] = names
        with open(OUT, 'w') as fh:
            json.dump(result, fh, indent=1)
        print('GRAD_DIAG wrote', OUT, flush=True)

    A.PushKDAgent.train_epoch = train_epoch
    A.PushKDAgent.calc_gradients = calc_gradients
    sys.argv = ['tokenhsi/run.py'] + sys.argv[1:]
    runpy.run_path('tokenhsi/run.py', run_name='__main__')


if __name__ == '__main__':
    main()

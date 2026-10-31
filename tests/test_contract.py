import importlib.util
import json
from pathlib import Path
import hashlib
import torch
import yaml
from learning.push_kd_losses import gaussian_forward_kl, masked_mean, push_advantages, teacher_observation, normalize_policy
from utils.push_reward import reward, contact
from utils.jointpush_geometry import push_rear_xy

ROOT = Path(__file__).resolve().parents[1]


def test_kl_direction_and_gradient_isolation():
    mu = torch.randn(8, 32, requires_grad=True)
    teacher = torch.randn(8, 32, requires_grad=True)
    sigma = torch.full_like(mu, .055)
    push = torch.arange(8) < 4
    kd = gaussian_forward_kl(mu[~push], sigma[~push], teacher[~push], sigma[~push])
    kd.backward()
    assert mu.grad[push].abs().sum() == 0
    assert mu.grad[~push].abs().sum() > 0
    assert teacher.grad is None
    expected = torch.distributions.kl_divergence(torch.distributions.Normal(teacher[~push].detach(), sigma[~push]), torch.distributions.Normal(mu[~push], sigma[~push])).sum(-1).mean()
    torch.testing.assert_close(kd, expected)


def test_ppo_and_advantage_ignore_old_tasks():
    x = torch.tensor([1., 3., 1e9, -1e9], requires_grad=True)
    mask = torch.tensor([True, True, False, False])
    masked_mean(x, mask).backward()
    assert x.grad.tolist() == [.5, .5, 0., 0.]
    adv = push_advantages(x.detach()[:, None], torch.zeros(4, 1), mask)
    torch.testing.assert_close(adv, torch.tensor([-1., 1., 0., 0.]))
    assert torch.isfinite(masked_mean(x, ~torch.ones(4, dtype=torch.bool)))


def test_reference_reward_success_and_saturation():
    spec = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_task/push_kd.yaml').read_text())['env']['jointPush']['contact']
    box = torch.zeros(3, 13); box[:, 2] = .75; box[:, 6] = 1
    goals = box[:, :3].clone(); goals[:, 1] = torch.tensor([.15, .3, 1.])
    hands = torch.tensor([[[-.2, -.8, 1.], [.2, -.8, 1.]]]).expand(3, -1, -1)
    size = torch.tensor([[1.6, 1.6, 1.5]]).expand(3, -1)
    force = torch.tensor([[[0., -1., 0.], [0., -1., 0.]]]).expand(3, -1, -1)
    phi, hit = contact(hands, box, goals, size, force, torch.tensor([[0., 1., 0.]]).expand(3, -1), spec)
    assert hit.all()
    r, at, _ = reward(torch.tensor([[0., -2., .9]]).expand(3, -1), hands, box, goals, size, hit, spec)
    assert at.tolist() == [True, False, False]
    torch.testing.assert_close(r[0], torch.tensor(1.2))
    assert (r <= 1.2).all()


def test_reference_data_integrity():
    cfg = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_task/push_kd.yaml').read_text())['env']
    bank = json.loads((ROOT/cfg['jointPush']['rsi']['bank_file']).read_text())
    rows = bank['candidates']['push']
    assert len(rows) == 949
    assert {r['motion_id'] for r in rows} == {0, 3, 4, 8}
    manifest = yaml.safe_load((ROOT/'tokenhsi/data/dataset_push_kd.yaml').read_text())['motions']
    assert len(manifest['push']) == 21 and len(manifest['loco_push']) == 4
    assert [hashlib.sha256(Path(r['file']).read_bytes()).hexdigest() for r in manifest['push']] == bank['contract']['motions']
    assert cfg['jointPush']['box']['density'] == 3
    assert len(cfg['jointPush']['box']['profiles']) == 167


def test_training_sampler_excludes_traj():
    import ast
    from enum import Enum
    from types import SimpleNamespace
    path = ROOT/'tokenhsi/env/tasks/multi_task/humanoid_push_kd.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_reset_task_indicator')
    ns = dict(torch=torch, os=__import__('os'))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    class IDs(Enum):
        traj=0; sit=1; carry=2; climb=3; push=4
    # No task libraries are needed to test the assignment branch itself.
    task = SimpleNamespace(_is_eval=False, num_envs=2048, device='cpu', TaskUID=IDs,
        _task_indicator=torch.zeros(2048,dtype=torch.long),
        _push_cand_idx=torch.zeros(2048,dtype=torch.long), _multiple_task_names=[])
    ns['_reset_task_indicator'](task, torch.arange(2048))
    assert (task._task_indicator[:1024] == 4).all()
    assert set(task._task_indicator[1024:].tolist()) == {1,2,3}
    task._is_eval=True; task._eval_task='traj'
    import pytest
    with pytest.raises(ValueError, match='TRAJ'):
        ns['_reset_task_indicator'](task, torch.arange(2048))


def test_real_checkpoints_preserve_old_policy():
    from learning.transformer.amp_network_builder_transformer import AMPTransformerMultiTaskBuilder
    from learning.amp_models import ModelAMPContinuous
    from rl_games.algos_torch.running_mean_std import RunningMeanStd
    torch.set_num_threads(2)
    params = yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/push_kd.yaml').read_text())['params']['network']
    def build(sizes, dim):
        builder = AMPTransformerMultiTaskBuilder(); builder.load(params)
        edges = [0]
        for s in sizes: edges.append(edges[-1]+s)
        info = dict(onehot_size=len(sizes), tota_subtask_obs_size=sum(sizes), each_subtask_obs_size=sizes, each_subtask_obs_indx=edges, each_subtask_name=['traj','sit','carry','climb','push'][:len(sizes)], enable_task_mask_obs=True)
        return ModelAMPContinuous.Network(builder.build('amp', actions_num=32, input_shape=(dim,), num_seqs=1, amp_input_shape=(10*(129+len(sizes)),), self_obs_size=223, task_obs_size=dim-223, multi_task_info=info, device='cpu')).eval()
    t, s = build([20,38,42,27], 354), build([20,38,42,27,42], 397)
    tc = torch.load(ROOT/'weights/teacher_stage1.pth', map_location='cpu', weights_only=True)
    sc = torch.load(ROOT/'weights/push_init.pth', map_location='cpu', weights_only=True)
    t.load_state_dict(tc['model'], strict=True); s.load_state_dict(sc['model'], strict=True)
    tr, sr = RunningMeanStd((354,)).eval(), RunningMeanStd((397,)).eval()
    tr.load_state_dict(tc['running_mean_std']); sr.load_state_dict(sc['running_mean_std'])
    x = torch.randn(4, 397); x[:, -5:] = 0; x[torch.arange(4), 392+torch.arange(4)] = 1
    # Inactive task blocks are masked in environment observations.
    edges = [223,243,281,323,350,392]
    for row in range(4):
        for k in range(5):
            if k != row: x[row, edges[k]:edges[k+1]] = 0
    tx = teacher_observation(x)
    with torch.no_grad():
        tm, ts = t.a2c_network.eval_actor(normalize_policy(tx, tr, 4), tx)
        sm, ss = s.a2c_network.eval_actor(normalize_policy(x, sr, 5), x)
    torch.testing.assert_close(tm, sm, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(ts, ss)

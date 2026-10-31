import hashlib
import json
from pathlib import Path
import torch
import yaml
from env.reference_projection import amp_to_tokenhsi, NATIVE_TASK_PROBS
from push_reference.utils.jointpush_scene import sample_layout, paths_clear, goal_angles_valid

ROOT = Path(__file__).resolve().parents[1]


def test_original_sampler_code_is_unchanged_except_namespace():
    provenance = json.loads((ROOT/'push_reference/PROVENANCE.json').read_text())
    for path in ('utils/jointpush_scene.py', 'utils/stage1_jointpush.py',
                 'utils/jointpush_rsi.py', 'utils/jointpush_rsi_geometry.py',
                 'env/tasks/multi_agent/humanoid_ma_stage1_jointpush.py',
                 'env/tasks/multi_agent/jointpush_reward.py'):
        source = (ROOT/'push_reference'/path).read_text()
        source = source.replace('from push_reference.', 'from ').replace('import push_reference.', 'import ')
        assert hashlib.sha256(source.encode()).hexdigest() == provenance[path], path


def test_reference_layout_enforces_cone_paths_and_human_goal_clearance():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = yaml.safe_load((ROOT/'tokenhsi/data/cfg/multi_task/push_reference.yaml').read_text())['env']['jointPush']
    n = 1024
    roots = torch.zeros(n,2,3); roots[:,1,0] = 25
    boxes = torch.zeros(n,4,13); boxes[...,6] = 1
    profiles = torch.tensor(cfg['box']['profiles'])
    size = torch.zeros(n,4,3); size[:,:2] = profiles[torch.randint(167,(n,2))]; size[:,2:] = .5
    push = torch.ones(n,2,dtype=torch.bool)
    result, goals = sample_layout(roots, boxes, size, push, cfg['scene'])
    distance = (result[:,:2,:2]-roots[...,:2]).norm(dim=-1)
    goal_distance = (result[:,:2,:2]-goals[...,:2]).norm(dim=-1)
    assert ((distance >= 2-1e-5) & (distance <= 10+1e-5)).all()
    assert ((goal_distance >= 2-1e-5) & (goal_distance <= 10+1e-5)).all()
    assert ((goals[...,:2]-roots[...,:2]).norm(dim=-1) >= 1).all()
    assert goal_angles_valid(result, goals, push, cfg['scene']).all()
    assert paths_clear(result, size, goals, roots, push, cfg['scene']).all()
    assert torch.unique(result[:,0,:2],dim=0).shape[0] == n
    torch.testing.assert_close(roots[:,0], torch.zeros(n,3))


def test_family_conversion_keeps_history_and_push_label():
    x = torch.randn(8,10,133)
    x[:,:,-4:] = 0
    for i in range(8): x[i,:,129+i%4] = 1
    converted = amp_to_tokenhsi(x.flatten(1)).reshape(8,10,134)
    torch.testing.assert_close(converted[...,:129], x[...,:129])
    mapping = [2,1,3,4]
    for i in range(8):
        assert (converted[i,:,-5:].argmax(-1) == mapping[i%4]).all()


def test_native_kd_excludes_traj_and_reference_reset_maps_both_actors(monkeypatch):
    from types import SimpleNamespace
    import env.mixed_reference_env as module
    assert NATIVE_TASK_PROBS[0] == 0
    assert sum(NATIVE_TASK_PROBS) == 1
    resets = []
    env = module.MixedReferenceEnv.__new__(module.MixedReferenceEnv)
    env.rl_device = 'cpu'
    env.reference_rows = 8
    env.reference = SimpleNamespace(device='cpu', reset=lambda ids: resets.append(('reference', ids.tolist())))
    env.native = SimpleNamespace(device='cpu', reset=lambda ids: resets.append(('native', ids.tolist())))
    env._ever = torch.ones(8, dtype=torch.bool)
    env._obs = lambda: torch.zeros(12,397)
    env.reset(torch.tensor([0,1,6,7,9]))
    assert resets == [('reference',[0,3]),('native',[1])]
    assert env._ever.tolist() == [False,False,True,True,True,True,False,False]


def test_native_pipe_transport_uses_arrays_without_tensor_sharing():
    import numpy as np
    from types import SimpleNamespace
    from env.native_pool_worker import NativePool
    payload = {key: np.zeros((2,3), dtype=np.float32) for key in
               ('obs','reward','done','terminate','amp','success','precision')}
    sent = []
    pool = NativePool.__new__(NativePool)
    pool._pipe = SimpleNamespace(poll=lambda timeout: True, recv=lambda: ('ok',payload),
                                 send=lambda message: sent.append(message))
    pool.step(torch.ones(2,32))
    assert isinstance(sent[0][1], np.ndarray)
    assert sent[0][1].shape == (2,32)
    assert isinstance(pool.obs_buf, torch.Tensor)
    pool.reset(torch.tensor([1]))
    assert isinstance(sent[1][1], np.ndarray)
    assert sent[1][1].dtype == np.int64

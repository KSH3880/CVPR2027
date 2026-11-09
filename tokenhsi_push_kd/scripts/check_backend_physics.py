"""Compare reference physics under identical reset RNG/actions in both backends."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tokenhsi')]
os.chdir(ROOT)


def diagnostic_worker(pipe, cfg, description):
    # Test-only instrumentation, keeping the production worker protocol intact.
    import isaacgym
    import torch
    from env.native_pool_worker import _worker
    from env.tasks.multi_task.humanoid_traj_sit_carry_climb import HumanoidTrajSitCarryClimb as Task
    offset = int(os.environ.get('PARITY_NATIVE_OFFSET', '0'))
    if offset:
        from isaacgym import gymapi
        original_create = Task._create_envs
        def create_with_offset(self, num_envs, spacing, num_per_row):
            # Test-only: reproduce the native partition's world origins without
            # creating reference actors or changing native actor tensor indices.
            for _ in range(offset):
                self.gym.create_env(self.sim, gymapi.Vec3(-spacing, -spacing, 0),
                                    gymapi.Vec3(spacing, spacing, spacing), num_per_row)
            return original_create(self, num_envs, spacing, num_per_row)
        Task._create_envs = create_with_offset
    records = []
    for name in ('step', 'reset'):
        original = getattr(Task, name)
        def wrapped(self, *args, _original=original, _name=name, **kwargs):
            if _name == 'reset' and not records and os.environ.get('PARITY_NATIVE_WARMUP') == '1':
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self._refresh_sim_tensors()
            result = _original(self, *args, **kwargs)
            records.append(dict(stage=_name, **native_state(self)))
            return result
        setattr(Task, name, wrapped)
    try:
        _worker(pipe, cfg, description)
    finally:
        torch.save(records, ROOT/'output_etc/check/backend_parity/native_worker.pt')
        torch.save(records, ROOT/'output_etc/check/backend_parity'/('native_'+os.environ['PARITY_LABEL']+'.pt'))


def native_state(task):
    return {key: getattr(task, key).detach().cpu().clone() for key in
            ('_root_states', '_dof_state', '_rigid_body_state', '_pd_action_offset',
             '_pd_action_scale', '_box_size', 'obs_buf') if hasattr(task, key)}


def main():
    label = sys.argv[1]
    os.environ['PARITY_LABEL'] = label
    backend = os.environ['TOKENHSI_SIM_BACKEND']
    from scripts.local_cuda_compat import install
    install()
    import torch
    import numpy as np
    import random
    from utils.config import get_args, load_cfg, parse_sim_params
    from env.mixed_reference_env import MixedReferenceEnv
    if backend == 'worker':
        import env.native_pool_worker as worker_module
        worker_module._worker = diagnostic_worker
    def seed(value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
    sys.argv = ['physics_check', '--task', 'HumanoidTokenHSIPushKD',
        '--cfg_env', 'tokenhsi/data/cfg/multi_task/push_kd.yaml',
        '--cfg_train', 'tokenhsi/data/cfg/train/rlg/push_kd.yaml',
        '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--pipeline', 'gpu',
        '--headless', '--num_envs', '8', '--seed', '42']
    a = get_args()
    cfg, train, _ = load_cfg(a)
    cfg['env']['referenceNumEnvs'] = cfg['env']['numEnvs']
    cfg['seed'] = 42
    seed(42)
    e = MixedReferenceEnv(a, cfg, parse_sim_params(a, cfg, train))
    r = e.reference
    records = []
    def capture(stage):
        record = dict(stage=stage)
        for key, value in dict(root=r._root_states, dof=r._dof_state,
                body=r._rigid_body_state, obs=e._obs()[:e.reference_rows],
                reward=r.rew_buf, done=r.reset_buf, contact=r._contact_forces,
                force=r.dof_force_tensor, sizes=r._box_size, target=r._tar_pos,
                native_obs=e.native.obs_buf, native_done=e.native.reset_buf).items():
            record[key] = value.detach().cpu().clone()
        records.append(record)
        if e.shared_sim is not None:
            record['native_state'] = native_state(e.native)
        print('PARITY_STAGE', stage, 'root_z', r._humanoid_root_states[..., 2].mean().item(),
              'reward', r.rew_buf.mean().item(), flush=True)
    try:
        capture('constructed')
        for step in range(12):
            if step in (0, 4, 8):
                seed(900+step)
                ids = torch.arange(r.num_envs, device=e.rl_device) if step == 0 else torch.tensor([0, 3], device=e.rl_device)
                e.reset(ids)
                capture('reset_reference_'+str(step))
                e.reset(ids+e.reference_rows)
                capture('reset_native_'+str(step))
            # Identical, deterministic actions independent of each backend's RNG.
            actions = torch.sin(torch.arange(e.num_envs*32, device=e.rl_device).float()+step).reshape(e.num_envs,32)*.05
            e.step(actions)
            capture('step_'+str(step))
        path=ROOT/'output_etc/check/backend_parity'/('trace_'+label+'.pt')
        torch.save(dict(backend=backend, records=records), path)
        print('PARITY_SAVED', path, flush=True)
    finally:
        e.close()


if __name__ == '__main__':
    main()

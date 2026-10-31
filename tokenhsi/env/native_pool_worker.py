"""Native TokenHSI simulator isolation via a spawned process.

CPU array transport is intentional: avoid sharing simulator/CUDA pointers
between two Isaac Gym instances. Measure transport cost before long runs.
"""
import multiprocessing as mp
import traceback
import copy


def _worker(pipe, cfg, sim_description):
    task = None
    try:
        import isaacgym
        from isaacgym import gymapi
        import torch
        torch.set_num_threads(2)
        from utils.config import parse_sim_params
        from env.tasks.multi_task.humanoid_traj_sit_carry_climb import HumanoidTrajSitCarryClimb
        args = cfg['args']
        args.physics_engine = gymapi.SIM_PHYSX
        import random
        import numpy as np
        seed = int(cfg.get('seed', 42))
        seed = seed if seed >= 0 else 42
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        params = parse_sim_params(args, cfg, sim_description['train'])
        task = HumanoidTrajSitCarryClimb(cfg, params, args.physics_engine, args.device, args.device_id, args.headless)
        task.reset(torch.arange(task.num_envs, device=task.device))
        def payload():
            array = lambda tensor: tensor.detach().cpu().numpy().copy()
            return dict(obs=array(task.obs_buf), reward=array(task.rew_buf),
                done=array(task.reset_buf), terminate=array(task._terminate_buf),
                amp=array(task.extras.get('amp_obs', task._amp_obs_buf.reshape(task.num_envs, -1))),
                success=array(getattr(task, '_success_buf', torch.zeros_like(task.reset_buf))),
                precision=array(getattr(task, '_precision_buf', torch.zeros_like(task.rew_buf))))
        pipe.send(('ok', payload()))
        while True:
            command, data = pipe.recv()
            if command == 'close':
                break
            if command == 'step':
                task.step(torch.from_numpy(data).to(task.device))
            elif command == 'reset':
                task.reset(torch.from_numpy(data).to(task.device))
            else:
                raise ValueError(command)
            pipe.send(('ok', payload()))
    except BaseException:
        pipe.send(('error', traceback.format_exc()))
    finally:
        if task is not None:
            task.gym.destroy_sim(task.sim)
        pipe.close()


class NativePool:
    def __init__(self, cfg, train_cfg):
        cfg = copy.deepcopy(cfg)
        cfg['args'].physics_engine = int(cfg['args'].physics_engine)
        context = mp.get_context('spawn')
        self._pipe, child = context.Pipe()
        self._process = context.Process(target=_worker, args=(child, cfg, dict(train=train_cfg)), daemon=True)
        self._process.start()
        child.close()
        self.device = 'cpu'
        self._receive()

    def _receive(self):
        import torch
        if not self._pipe.poll(300):
            raise RuntimeError('Native TokenHSI worker did not respond within 300 seconds')
        try:
            status, data = self._pipe.recv()
        except EOFError as error:
            raise RuntimeError('Native TokenHSI simulator worker exited unexpectedly') from error
        if status != 'ok':
            raise RuntimeError('Native TokenHSI worker failed:\n'+data)
        data = {key: torch.from_numpy(value) for key, value in data.items()}
        self.obs_buf, self.rew_buf, self.reset_buf = data['obs'], data['reward'], data['done']
        self._terminate_buf, self._success_buf, self._precision_buf = data['terminate'], data['success'], data['precision']
        self.extras = dict(amp_obs=data['amp'])

    def step(self, actions):
        self._pipe.send(('step', actions.detach().cpu().numpy().copy()))
        self._receive()

    def reset(self, ids):
        self._pipe.send(('reset', ids.detach().cpu().numpy().copy()))
        self._receive()

    def close(self):
        if self._process.is_alive():
            self._pipe.send(('close', None))
            self._process.join(10)
        self._pipe.close()

"""Two simulator pools: unchanged 26k scenes and native TokenHSI KD scenes.

Each reference scene contributes two independent policy rows. Non-PUSH
reference rows control their actors but do not receive RL or KD losses.
"""
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from gym import spaces
from env.reference_projection import amp_to_tokenhsi, project, NATIVE_TASK_PROBS


class MixedReferenceEnv:
    def __init__(self, args, cfg, sim_params):
        from push_reference.env.tasks.multi_agent.humanoid_ma_stage1_jointpush import HumanoidMAStage1JointPush
        from env.native_pool_worker import NativePool
        self.rl_device = args.rl_device
        evaluation = args.eval and args.test
        push_eval = evaluation and args.eval_task == 'push'
        if push_eval and os.environ.get('TOKENHSI_PUSH_INIT', 'loco') != 'loco':
            raise ValueError('Reference evaluation follows 26k loco-only protocol; RSI eval is not enabled')
        old_eval = evaluation and not push_eval
        if old_eval and args.eval_task not in ('sit', 'carry', 'climb'):
            raise ValueError('Supported tasks: push, sit, carry, climb (TRAJ excluded)')
        seed = int(cfg.get('seed', 42))
        seed = seed if seed >= 0 else 42
        n = cfg['env']['numEnvs']
        self.reference = self.native = None
        self._projection_reset = None
        self.reference_rows = 0
        if not old_eval:
            rcfg = yaml.safe_load(Path('tokenhsi/data/cfg/multi_task/push_reference.yaml').read_text())
            rcfg['env']['numEnvs'] = n
            rcfg['args'] = copy.copy(args)
            rcfg['args'].task_graph = 'push' if push_eval else 'random_scenario'
            rcfg['args'].task_camera = 'stack'
            rcfg['headless'] = args.headless
            rcfg['seed'] = seed
            self.reference = HumanoidMAStage1JointPush(rcfg, copy.deepcopy(sim_params), args.physics_engine, args.device, args.device_id, args.headless)
            self.reference_rows = 2*n
            self.reference.reset(torch.arange(n, device=self.reference.device))
            self._projection_reset = torch.arange(n, device=self.reference.device)
        if not push_eval:
            native_cfg = yaml.safe_load(Path('tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml').read_text())
            native_cfg['env']['numEnvs'] = n
            native_cfg['env']['motion_file'] = 'tokenhsi/data/dataset_loco_sit_carry_climb_native.yaml'
            native_cfg['env']['taskInitProb'] = list(NATIVE_TASK_PROBS)
            native_cfg['args'] = copy.copy(args)
            native_cfg['headless'] = args.headless
            native_cfg['seed'] = seed
            self.native = NativePool(native_cfg, yaml.safe_load(Path('tokenhsi/data/cfg/train/rlg/push_kd.yaml').read_text()))
        self.num_envs = self.reference_rows+(n if self.native else 0)
        self.num_obs, self.num_actions, self.num_states = 397, 32, 0
        self.num_agents = 1  # learner rows, not physical humans per reference scene
        self.action_space = spaces.Box(-np.ones(32), np.ones(32), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(397,), dtype=np.float32)
        self.amp_observation_space = spaces.Box(-np.inf, np.inf, shape=(1340,), dtype=np.float32)
        kd = torch.arange(self.num_envs, device=self.rl_device) >= self.reference_rows
        self.task = SimpleNamespace(num_envs=self.num_envs, num_agents=1, num_actions=32,
            viewer=None, kd_mask=kd, _num_amp_obs_steps=10, _enable_task_obs=True,
            _enable_task_mask_obs=True, _multiple_task_names=['sit','carry','climb','push'],
            cfg=cfg, get_obs_size=lambda:397, get_task_obs_size=lambda:174,
            get_multi_task_info=self.multi_task_info)
        self._ever = torch.zeros(self.reference_rows, dtype=torch.bool, device=self.rl_device)

    @staticmethod
    def multi_task_info():
        return dict(onehot_size=5, tota_subtask_obs_size=169, each_subtask_obs_size=[20,38,42,27,42],
                    each_subtask_obs_indx=[0,20,58,100,127,169],
                    each_subtask_name=['traj','sit','carry','climb','push'], enable_task_mask_obs=True)

    def get_number_of_agents(self):
        return 1

    def close(self):
        if self.native is not None:
            self.native.close()
        if self.reference is not None:
            self.reference.gym.destroy_sim(self.reference.sim)
            self.reference = None

    def _obs(self):
        values = []
        if self.reference:
            values.append(project(self.reference, self._projection_reset).to(self.rl_device))
            self._projection_reset = None
        if self.native:
            native = self.native.obs_buf
            x = native.new_zeros(len(native), 397)
            x[:, :350], x[:, 392:396] = native[:, :350], native[:, 350:354]
            values.append(x.to(self.rl_device))
        return torch.cat(values)

    def reset(self, env_ids=None):
        if env_ids is not None and len(env_ids):
            ids = torch.as_tensor(env_ids, device=self.rl_device, dtype=torch.long)
            ref = ids[ids < self.reference_rows]
            if len(ref):
                scenes = (ref//2).unique()
                self.reference.reset(scenes.to(self.reference.device))
                self._projection_reset = scenes.to(self.reference.device)
                self._ever[(scenes[:, None]*2+torch.arange(2, device=scenes.device)).flatten()] = False
            old = ids[ids >= self.reference_rows]-self.reference_rows
            if len(old):
                self.native.reset(old.to(self.native.device))
        return self._obs()

    def step(self, actions):
        rewards, dones, amps, terms, successes, precision = [], [], [], [], [], []
        if self.reference:
            r = self.reference
            r.step(actions[:self.reference_rows].clamp(-1, 1))
            rewards.append(r.rew_buf.flatten().to(self.rl_device))
            dones.append(r.reset_buf.repeat_interleave(2).to(self.rl_device))
            amps.append(amp_to_tokenhsi(r.extras['amp_obs']).to(self.rl_device))
            terms.append(r._terminate_buf.flatten().to(self.rl_device))
            current = r.relation_runtime.done.flatten().to(self.rl_device)
            self._ever |= current
            successes.append(current)
            precision.append((r._box_states[:, :2, :2]-r._tar_pos[..., :2]).norm(dim=-1).flatten().to(self.rl_device))
        if self.native:
            r = self.native
            r.step(actions[self.reference_rows:].clamp(-1, 1))
            rewards.append(torch.zeros_like(r.rew_buf).to(self.rl_device))
            dones.append(r.reset_buf.to(self.rl_device))
            amp = r.extras['amp_obs'].reshape(-1,10,133)
            expanded = torch.cat((amp, amp.new_zeros(*amp.shape[:-1],1)), -1)
            amps.append(expanded.flatten(1).to(self.rl_device))
            terms.append(r._terminate_buf.to(self.rl_device))
            successes.append(getattr(r, '_success_buf', torch.zeros_like(r.reset_buf)).bool().to(self.rl_device))
            precision.append(getattr(r, '_precision_buf', torch.zeros_like(r.rew_buf)).to(self.rl_device))
        obs = self._obs()
        ever = torch.cat((self._ever, torch.zeros(self.num_envs-self.reference_rows, dtype=torch.bool, device=self.rl_device)))
        info = dict(amp_obs=torch.cat(amps), terminate=torch.cat(terms), success=torch.cat(successes), precision=torch.cat(precision), push_success_ever=ever)
        for name, index in [('sit',1),('carry',2),('climb',3),('push',4)]:
            info[name] = obs[:, 392+index] > .5
        return obs, torch.cat(rewards), torch.cat(dones), info

    def fetch_amp_obs_demo(self, count):
        # Reference expert mixtures stay unchanged; keep only PUSH-family rows.
        result = []
        total = 0
        while total < count:
            x = self.reference.fetch_amp_obs_demo(max(count, 512))
            x = x[x.reshape(-1,10,133)[:,0,132] > .5]
            result.append(amp_to_tokenhsi(x)); total += len(x)
        return torch.cat(result)[:count].to(self.rl_device)

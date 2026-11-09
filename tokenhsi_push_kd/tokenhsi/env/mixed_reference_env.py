"""Reference scenes and native TokenHSI KD scenes in separate environment pools.

Each reference scene contributes one policy row per physical person. Non-PUSH
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
    def __new__(cls, args=None, cfg=None, sim_params=None):
        if (args is not None and args.eval and args.test and args.eval_task in ('sit', 'climb')
                and os.environ.get('TOKENHSI_EVAL_PROTOCOL') == 'cvpr2027_single'
                and os.environ.get('CVPR_PROTOCOL_REVISION', '1109') == '1109'):
            from env.cvpr2027_native import NativeProtocolEnv
            return NativeProtocolEnv(args, cfg, sim_params)
        return super().__new__(cls)

    def __init__(self, args, cfg, sim_params):
        from push_reference.env.tasks.multi_agent.humanoid_ma_stage1_jointpush import HumanoidMAStage1JointPush
        from env.native_pool_worker import NativePool
        self.rl_device = args.rl_device
        evaluation = args.eval and args.test
        self.cvpr_single = evaluation and os.environ.get('TOKENHSI_EVAL_PROTOCOL') == 'cvpr2027_single'
        if self.cvpr_single:
            from env.cvpr2027_single import CVPRSingleTask, PRESETS
            if args.eval_task not in PRESETS:
                raise ValueError('CVPR single tasks: sit, climb, at, on_top, push')
            if os.environ.get('NATIVE_EVAL_WARMUP', '0') != '0':
                raise ValueError('CVPR evaluation forbids warm-up episodes')
            HumanoidMAStage1JointPush = CVPRSingleTask
        push_eval = evaluation and (args.eval_task == 'push' or self.cvpr_single)
        if push_eval and os.environ.get('TOKENHSI_PUSH_INIT', 'loco') != 'loco':
            raise ValueError('Reference evaluation follows 26k loco-only protocol; RSI eval is not enabled')
        old_eval = evaluation and not push_eval
        if old_eval and args.eval_task not in ('sit', 'carry', 'climb'):
            raise ValueError('Supported tasks: push, sit, carry, climb (TRAJ excluded)')
        seed = int(cfg.get('seed', 42))
        seed = seed if seed >= 0 else 42
        n = cfg['env']['numEnvs']
        reference_n = n if evaluation else int(cfg['env'].get('referenceNumEnvs', n))
        if reference_n <= 0:
            raise ValueError('referenceNumEnvs must be positive')
        self.reference = self.native = None
        self._projection_reset = None
        self.reference_rows = 0
        if not push_eval:
            native_cfg = yaml.safe_load(Path('tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml').read_text())
            native_cfg['env']['numEnvs'] = n
            native_cfg['env']['motion_file'] = 'tokenhsi/data/dataset_loco_sit_carry_climb_native.yaml'
            native_cfg['env']['taskInitProb'] = list(NATIVE_TASK_PROBS)
            native_cfg['args'] = copy.copy(args)
            native_cfg['headless'] = args.headless
            native_cfg['seed'] = seed
        self.shared_sim = None
        backend = os.environ.get('TOKENHSI_SIM_BACKEND', cfg['env'].get('simBackend', 'worker'))
        if backend not in ('worker', 'shared'):
            raise ValueError('TOKENHSI_SIM_BACKEND must be worker or shared')
        # Record the effective training backend in the checkpoint environment contract.
        if not evaluation:
            cfg['env']['simBackend'] = backend
            if backend == 'shared':
                cfg['env']['sharedSimVersion'] = 2  # merged setters and isolated native RNG
        if backend == 'shared' and not evaluation:
            if not args.headless:
                raise ValueError('Shared simulator currently requires headless mode')
            from env.shared_sim import SharedSim
            self.shared_sim = SharedSim(native_cfg, copy.deepcopy(sim_params), args, reference_n)
        if not old_eval:
            rcfg = yaml.safe_load(Path('tokenhsi/data/cfg/multi_task/push_reference.yaml').read_text())
            rcfg['env']['numEnvs'] = reference_n
            rcfg['args'] = copy.copy(args)
            rcfg['args'].task_graph = 'push' if push_eval else 'random_scenario'
            rcfg['args'].task_camera = 'stack'
            rcfg['headless'] = args.headless
            rcfg['seed'] = seed
            if self.cvpr_single:
                rcfg['cvpr_task'] = args.eval_task
                rcfg['args'].task_graph = PRESETS[args.eval_task]
                rcfg['env'].update(numAgents=1, numObjects=2, episodeLength=900)
            if self.shared_sim is not None:
                for key in ('sim',):
                    if rcfg[key] != native_cfg[key]:
                        raise ValueError('Shared simulator requires identical physics configuration')
                for key in ('plane', 'controlFrequencyInv'):
                    if rcfg['env'][key] != native_cfg['env'][key]:
                        raise ValueError('Shared simulator requires matching '+key)
                rcfg['_shared_gym'] = self.shared_sim.parts[0]
            self.reference = HumanoidMAStage1JointPush(rcfg, copy.deepcopy(sim_params), args.physics_engine, args.device, args.device_id, args.headless)
            self.reference_rows = self.reference.num_agents*reference_n
            self.reference.reset(torch.arange(reference_n, device=self.reference.device))
            self._projection_reset = torch.arange(reference_n, device=self.reference.device)
        if not push_eval:
            if self.shared_sim is not None:
                self.native = self.shared_sim.native
                self.shared_sim.reset_native(torch.arange(n, device=self.native.device))
            else:
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
        if self.shared_sim is not None:
            self.shared_sim.close()
            self.reference = self.native = None
            return
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
                scenes = (ref//self.reference.num_agents).unique()
                self.reference.reset(scenes.to(self.reference.device))
                self._projection_reset = scenes.to(self.reference.device)
                self._ever[(scenes[:, None]*self.reference.num_agents+torch.arange(self.reference.num_agents, device=scenes.device)).flatten()] = False
            old = ids[ids >= self.reference_rows]-self.reference_rows
            if len(old):
                if self.shared_sim is not None:
                    self.shared_sim.reset_native(old.to(self.native.device))
                else:
                    self.native.reset(old.to(self.native.device))
        return self._obs()

    def step(self, actions):
        rewards, dones, amps, terms, successes, precision = [], [], [], [], [], []
        if self.shared_sim is not None:
            self.shared_sim.step(self.reference, actions[:self.reference_rows].clamp(-1, 1),
                                 actions[self.reference_rows:].clamp(-1, 1))
        if self.reference:
            r = self.reference
            if self.shared_sim is None:
                r.step(actions[:self.reference_rows].clamp(-1, 1))
            rewards.append(r.rew_buf.flatten().to(self.rl_device))
            dones.append(r.reset_buf.repeat_interleave(r.num_agents).to(self.rl_device))
            amps.append(amp_to_tokenhsi(r.extras['amp_obs']).to(self.rl_device))
            terms.append(r._terminate_buf.flatten().to(self.rl_device))
            current = r.relation_runtime.done.flatten().to(self.rl_device)
            self._ever |= current
            successes.append(current)
            precision.append((r._box_states[:, :r.num_agents, :2]-r._tar_pos[..., :2]).norm(dim=-1).flatten().to(self.rl_device))
        if self.native:
            r = self.native
            if self.shared_sim is None:
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

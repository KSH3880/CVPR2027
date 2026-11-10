"""User-requested original TokenHSI SIT/CLIMB scenes, with the common eval loop.

Native target-point success is retained; no artificial chair-top XY region is used.
"""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from gym import spaces


class NativeStateAdapter:
    scoring_mode = 'native_point'

    def __init__(self, task, name):
        self.native = task
        self.cvpr_task = name
        self.cfg = task.cfg
        self.humanoid_handles_all = [[h] for h in task.humanoid_handles]
        origins = [task.gym.get_env_origin(e) for e in task.envs]
        self._env_origins = torch.tensor([[p.x,p.y,p.z] for p in origins], device=task.device)
        lib = task._sit_obj_lib if name == 'sit' else task._climb_obj_lib
        self.asset_paths = [lib._obj_urdfs[i] for i in lib._every_env_object_ids.cpu().tolist()]
        size = lib._every_env_object_bbox_lengths
        self._box_size = torch.stack((size, torch.zeros_like(size)), 1)
        # Density is inapplicable for fixed-base native furniture; archive marks this explicitly.
        self._box_density = torch.zeros(task.num_envs, 2, device=task.device)
        self.cvpr_observed_initial_boxes = self._box_states.clone()

    def __getattr__(self, name):
        return getattr(self.native, name)

    @property
    def _humanoid_root_states(self):
        return self.native._humanoid_root_states[:,None]

    @property
    def _box_states(self):
        obj = self.native._sit_object_states if self.cvpr_task == 'sit' else self.native._climb_object_states
        placeholder = torch.zeros_like(obj); placeholder[:,6] = 1
        return torch.stack((obj,placeholder),1)

    @property
    def _tar_pos(self):
        return (self.native._sit_tar_pos if self.cvpr_task == 'sit' else self.native._climb_tar_pos)[:,None]

    def snapshot_extras(self):
        return dict(native_root_states=self.native._root_states.detach().cpu().numpy().copy(),
                    asset_paths=np.asarray(self.asset_paths))

    def restore_initial(self, archive):
        if not np.array_equal(archive['asset_paths'], np.asarray(self.asset_paths)):
            raise ValueError('Native replay furniture asset mismatch')
        n = self.native
        for key,tensor in [('native_root_states', n._root_states), ('dof_pos', n._dof_pos),
                           ('dof_vel', n._dof_vel), ('kinematic', n._kinematic_humanoid_rigid_body_states)]:
            tensor.copy_(torch.as_tensor(archive[key], device=n.device))
        self._tar_pos.copy_(torch.as_tensor(archive['goals'], device=n.device))
        self.cvpr_observed_initial_boxes.copy_(torch.as_tensor(archive['observed_boxes'],device=n.device))
        ids = torch.arange(n.num_envs,device=n.device)
        n._reset_env_tensors(ids)
        n._refresh_sim_tensors()
        n._compute_observations(ids)


class NativeProtocolEnv:
    cvpr_single = True

    def __init__(self, args, cfg, sim_params):
        from env.tasks.multi_task.humanoid_traj_sit_carry_climb import HumanoidTrajSitCarryClimb
        from env.mixed_reference_env import MixedReferenceEnv
        native_cfg = yaml.safe_load(Path('tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml').read_text())
        native_cfg['env'].update(numEnvs=cfg['env']['numEnvs'], episodeLength=900, episodeLengthShort=900,
                                 motion_file='tokenhsi/data/dataset_loco_sit_carry_climb_native.yaml')
        native_cfg.update(args=copy.copy(args), headless=args.headless, seed=cfg['seed'])
        self.native = HumanoidTrajSitCarryClimb(native_cfg, copy.deepcopy(sim_params), args.physics_engine,
                                               args.device,args.device_id,args.headless)
        import os
        if os.environ.get('CVPR_CLIMB_RESET_DIAGNOSTIC') == '1':
            if args.eval_task != 'climb':
                raise ValueError('Reset diagnostic is CLIMB-only')
            from learning.climb_reset_diagnostic import capture_rng
            self.diagnostic_reset_rng = capture_rng(self.native.device)
        self.native.reset(torch.arange(self.native.num_envs,device=self.native.device))
        self.reference = NativeStateAdapter(self.native,args.eval_task)
        self.rl_device = args.rl_device
        self.num_envs = self.native.num_envs
        self.num_obs,self.num_actions,self.num_states,self.num_agents = 397,32,0,1
        self.action_space = spaces.Box(-np.ones(32), np.ones(32),dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf,np.inf,shape=(397,),dtype=np.float32)
        self.amp_observation_space = spaces.Box(-np.inf,np.inf,shape=(1340,),dtype=np.float32)
        self.task = SimpleNamespace(num_envs=self.num_envs,num_agents=1,num_actions=32,viewer=None,
            cfg=cfg,_num_amp_obs_steps=10,_enable_task_obs=True,_enable_task_mask_obs=True,
            get_obs_size=lambda:397,get_task_obs_size=lambda:174,
            get_multi_task_info=MixedReferenceEnv.multi_task_info)
        if abs(self.native.dt-1/30)>1e-6:
            raise ValueError('Native eval requires 30Hz')

    def get_number_of_agents(self):
        return 1

    def _obs(self):
        raw = self.native.obs_buf
        result = raw.new_zeros(len(raw),397)
        result[:,:350],result[:,392:396] = raw[:,:350],raw[:,350:354]
        return result.to(self.rl_device)

    def reset(self, env_ids=None):
        if env_ids is not None and len(env_ids):
            raise RuntimeError('Native first-episode protocol forbids mid-evaluation resets')
        return self._obs()

    def step(self, actions):
        self.native.step(actions.to(self.native.device).clamp(-1,1))
        n = self.native
        return self._obs(),n.rew_buf.to(self.rl_device),n.reset_buf.to(self.rl_device),dict(terminate=n._terminate_buf.to(self.rl_device))

    def close(self):
        self.native.gym.destroy_sim(self.native.sim)

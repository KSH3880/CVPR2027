"""Original TokenHSI furniture adaptation policy on a shared native scene."""
import ast
import json
from pathlib import Path
import torch
import yaml
from isaacgym.torch_utils import quat_rotate, quat_mul
from utils import torch_utils
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from learning.transformer.amp_network_builder_transformer_adapt import AMPTransformerMultiTaskAdaptBuilder
from evaluation.hold_at.metrics import sha256


def original_function(path, name):
    node = next(x for x in ast.parse(Path(path).read_text()).body if isinstance(x, ast.FunctionDef) and x.name == name)
    node.decorator_list = []
    scope = dict(torch=torch, torch_utils=torch_utils, quat_rotate=quat_rotate, quat_mul=quat_mul)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


class PublicChairPolicy:
    def __init__(self, task, player, config):
        self.task, self.player, self.first = task, player, True
        self.config = config
        self._cpu_validated = False
        self.human = original_function(config['human_source'], 'compute_humanoid_observations_max')
        self.location = original_function(config['carry_source'], 'compute_location_observations')
        params = yaml.safe_load(Path(config['train_config']).read_text())['params']['network']
        info = dict(onehot_size=2, tota_subtask_obs_size=84, each_subtask_obs_size=[42, 42],
                    each_subtask_obs_mask=torch.zeros((2,84), dtype=torch.bool, device=task.device),
                    each_subtask_obs_indx=torch.tensor([0,42,84], device=task.device),
                    enable_task_mask_obs=False, each_subtask_name=['new_carry','old_carry'],
                    major_task_name='carry', has_extra=False)
        builder = AMPTransformerMultiTaskAdaptBuilder(); builder.load(params)
        self.net = builder.build('amp', actions_num=32, input_shape=(265,), num_seqs=task.num_envs,
            value_size=1, amp_input_shape=(1290,), self_obs_size=223, task_obs_size=42,
            multi_task_info=info, device=task.device, hrl_checkpoint=config['base_checkpoint']).to(task.device)
        checkpoint = torch.load(config['checkpoint'], map_location=task.device, weights_only=False)
        state = {k[len('a2c_network.'):]:v for k,v in checkpoint['model'].items()}
        self.net.load_state_dict(state, strict=True)
        self.net.requires_grad_(False); self.net.eval()
        self.rms = RunningMeanStd((265,)).to(task.device).eval()
        self.rms.load_state_dict(checkpoint['running_mean_std'], strict=True)
        assert all(torch.equal(v, self.net.state_dict()[k]) for k,v in state.items())
        assert all(torch.equal(v, self.rms.state_dict()[k]) for k,v in checkpoint['running_mean_std'].items())
        signs = torch.tensor([[1,1,-1],[-1,1,-1],[-1,-1,-1],[1,-1,-1],
                              [1,1,1],[-1,1,1],[-1,-1,1],[1,-1,1]],device=task.device)
        self.bps = task._box_size[:,0,None]*signs[None]/2
        self.metadata = dict(policy=config.get('policy_name', 'public_tokenhsi_chair_adapt'), source_epoch=int(checkpoint['epoch']),
            checkpoint=str(Path(config['checkpoint']).resolve()), checkpoint_sha256=sha256(config['checkpoint']),
            base_checkpoint_sha256=sha256(config['base_checkpoint']), strict_model_tensors=len(state),
            observation_size=265, self_observation_size=223, carry_observation_size=42,
            original_observation_functions=True, weights_frozen=True)
        task._public_policy_metadata = self.metadata
        print('PUBLIC_CARRY_POLICY', json.dumps(self.metadata), flush=True)

    @torch.no_grad()
    def action(self, reset_env_ids=None):
        t = self.task
        if reset_env_ids is not None:
            from evaluation.climb_paired.repeat_metrics import bodies_after_reset
            pos,rot,vel,ang=bodies_after_reset(
                [t._rigid_body_pos[:,0],t._rigid_body_rot[:,0],t._rigid_body_vel[:,0],t._rigid_body_ang_vel[:,0]],
                t._kinematic_humanoid_rigid_body_states[:,0],reset_env_ids)
            self.first=False
        elif self.first:
            bodies=t._kinematic_humanoid_rigid_body_states[:,0]
            pos,rot,vel,ang=(bodies[...,:3],bodies[...,3:7],bodies[...,7:10],bodies[...,10:13])
            self.first=False
        else:
            pos,rot,vel,ang=t._rigid_body_pos[:,0],t._rigid_body_rot[:,0],t._rigid_body_vel[:,0],t._rigid_body_ang_vel[:,0]
        human=self.human(pos,rot,vel,ang,False,False)
        carry=self.location(t._humanoid_root_states[:,0],t._box_states[:,0],self.bps,t._tar_pos[:,0],True)
        raw=torch.cat([human,carry],-1)
        if raw.shape!=(t.num_envs,265) or not torch.isfinite(raw).all():
            raise ValueError('Invalid original carry observation')
        mu,_=self.net.eval_actor(self.rms(raw),raw)
        if not torch.isfinite(mu).all():
            raise ValueError('Nonfinite public policy actions')
        if self.config.get('validate_cpu_action') and not self._cpu_validated:
            from types import SimpleNamespace
            count = min(8, t.num_envs)
            ref_task = SimpleNamespace(device='cpu', num_envs=count,
                                      _box_size=t._box_size[:count].detach().cpu())
            reference = PublicChairPolicy(ref_task, self.player,
                                         dict(self.config, validate_cpu_action=False))
            cpu_raw = raw[:count].detach().cpu()
            cpu_mu,_ = reference.net.eval_actor(reference.rms(cpu_raw), cpu_raw)
            torch.testing.assert_close(mu[:count].cpu(), cpu_mu, rtol=1e-4, atol=1e-4)
            error = float((mu[:count].cpu()-cpu_mu).abs().max())
            output = Path(t.cfg['args'].output_path)
            torch.save(dict(raw=cpu_raw, gpu_mu=mu[:count].cpu(), cpu_mu=cpu_mu),
                       output / 'public_policy_cpu_parity.pth')
            (output / 'public_policy_cpu_parity.json').write_text(json.dumps(dict(
                validated=True, samples=count, max_abs_error=error,
                flash_sdp_enabled=torch.backends.cuda.flash_sdp_enabled(),
                mem_efficient_sdp_enabled=torch.backends.cuda.mem_efficient_sdp_enabled()), indent=2))
            print('PUBLIC_POLICY_CPU_PARITY', error, flush=True)
            self._cpu_validated = True
        if self.player.clip_actions:
            return (self.player.actions_high-self.player.actions_low)/2*mu.clamp(-1,1)+(self.player.actions_high+self.player.actions_low)/2
        return mu

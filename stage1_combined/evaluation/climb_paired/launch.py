"""Same scene and first-episode evaluator; select only the action policy."""
from pathlib import Path
import importlib.util
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'tokenhsi'), str(ROOT)]
import isaacgym
import torch

# RTX5090/torch2.0 cannot use the bundled fused attention kernels.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
from evaluation.finetuned.backend import EvaluationTask, EvaluationPlayer
from evaluation.public_point.policy import PublicPolicy
import utils.parse_task as parser

if os.environ.get('CLIMB_PAIRED_ORIGINAL30D') == '1':
    from env.tasks.multi_agent.humanoid_ma_carry import HumanoidMACarry
    from learning.multi_agent.ma_players import MAPlayerContinuous

    class Original30Task(EvaluationTask):
        def get_obs_size(self):return HumanoidMACarry.get_obs_size(self)
        def get_scene_entity_sizes(self):return HumanoidMACarry.get_scene_entity_sizes(self)
        def _compute_clean_scene_obs(self,env_ids=None):return HumanoidMACarry._compute_clean_scene_obs(self,env_ids)

    EvaluationTask = Original30Task
    PlayerBase = MAPlayerContinuous
else:
    PlayerBase = EvaluationPlayer


class PairedPlayer(PlayerBase):
    def restore(self, path):
        result = super().restore(path)
        source = torch.load(path, map_location='cpu', weights_only=False)
        assert set(self.model.state_dict()) == set(source['model'])
        for key, value in self.model.state_dict().items():
            assert torch.equal(value.detach().cpu(), source['model'][key]), key
        for key, value in self.running_mean_std.state_dict().items():
            assert torch.equal(value.detach().cpu(), source['running_mean_std'][key]), key
        self._paired_epoch = int(source['epoch'])
        self._paired_model = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        self._paired_rms = {key: value.detach().cpu().clone() for key, value in self.running_mean_std.state_dict().items()}
        return result

    def run_eval(self):
        task = self.env.task
        assert abs(float(task.dt)-1/30) < 1e-7
        if os.environ['CLIMB_PAIRED_POLICY'] == 'tokenhsi':
            # Public network creation must not change reset sampling.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                self.public = PublicPolicy(task, self, task.cfg['env']['chairFT']['public_checkpoint'])
        if os.environ.get('CLIMB_PAIRED_CONTINUOUS3')=='1':
            from evaluation.climb_paired.continuous import run_continuous
            result=run_continuous(self)
        else:
            result = EvaluationPlayer.run_eval(self)
        for key, value in self.model.state_dict().items():
            assert torch.equal(value.detach().cpu(), self._paired_model[key]), key
        for key, value in self.running_mean_std.state_dict().items():
            assert torch.equal(value.detach().cpu(), self._paired_rms[key]), key
        path = Path(task.cfg['args'].output_path)/'results.json'
        record = json.loads(path.read_text())
        record['policy'] = self.public.metadata if hasattr(self, 'public') else dict(epoch=self._paired_epoch, model_keys_strict=True)
        record['scene_checkpoint_exact'] = True
        record['scene_model_and_rms_unchanged'] = True
        record['action_policy'] = os.environ['CLIMB_PAIRED_POLICY']
        path.write_text(json.dumps(record, indent=2))
        return result

    def get_action(self, obs_dict, is_determenistic=False):
        if hasattr(self, 'public'):
            return self.public.action(getattr(self,'_public_reset_env_ids',None))
        return super().get_action(obs_dict, is_determenistic)


parser.EvaluationTask = EvaluationTask
spec = importlib.util.spec_from_file_location('climb_paired_run', ROOT/'tokenhsi/run.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
original = run.build_alg_runner


def build(observer):
    runner = original(observer)
    runner.player_factory.register_builder('climb30_ft' if os.environ.get('CLIMB_PAIRED_ORIGINAL30D') == '1' else 'chair_ft', lambda **kwargs: PairedPlayer(**kwargs))
    return runner


run.build_alg_runner = build
run.main()

"""Run coupled_base with local PyTorch 2.0 / CUDA 11 Blackwell compatibility."""
import inspect
from pathlib import Path
import runpy
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT/'ksh_jointpush/coupled_base'


def _replace_function(owner, name, replacements):
    original = getattr(owner, name)
    if getattr(original, '_local_runtime_compat', False):
        return False
    source = textwrap.dedent(inspect.getsource(original))
    updated = source
    for before, after in replacements:
        if before not in updated:
            raise ValueError('Unexpected coupled compatibility target: '+name+' / '+before)
        updated = updated.replace(before, after)
    scope = dict(original.__globals__)
    exec(compile(updated, original.__code__.co_filename, 'exec'), scope)
    # Preserve aliases imported by tasks and samplers before this installation.
    original.__code__ = scope[name].__code__
    original._local_runtime_compat = True
    return True


def install():
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(PROJECT/'tokenhsi'), str(PROJECT), str(ROOT)]
    import isaacgym
    import torch
    from ksh_jointpush.ksh_torch_compat import install_attention_compat
    install_attention_compat()
    patched = []
    try:
        torch.zeros(1, 1, dtype=torch.bool).any((0, 1))
    except TypeError:
        from utils import coupled_push_scene
        if _replace_function(coupled_push_scene, 'scene_infeasible',
                             [('.any((1,2))', '.any(2).any(1)')]):
            patched.append('coupled_push_scene.scene_infeasible')
    if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
            and int(torch.version.cuda.split('.')[0]) < 12):
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        from utils import coupled_push_spec, size_rsi
        from env.tasks.multi_agent.humanoid_ma_coupled_push import HumanoidMACoupledPush
        targets = [
            (coupled_push_spec, 'asset_bank', ['sizes', 'sizes[:,16:]'], '.prod(-1)'),
            (HumanoidMACoupledPush, '_load_box_asset', ['self._box_size'], '.prod(-1)'),
            (size_rsi, 'task_probabilities', ['(bounds[:, 1] - bounds[:, 0] + 1)'], '.prod()'),
        ]
        for owner, name, expressions, reduction in targets:
            replacements = [(value+reduction,
                '('+'*'.join(value+'[..., '+str(axis)+']' for axis in range(3))+')')
                for value in expressions]
            if _replace_function(owner, name, replacements):
                patched.append(name)
    print('COUPLED_LOCAL_RUNTIME_COMPAT', dict(torch=torch.__version__, cuda=torch.version.cuda,
          functions=patched), flush=True)
    return patched


def install_training_audit():
    """Record the first two PPO updates and save a reviewable startup checkpoint."""
    import json
    import torch
    from ksh_jointpush.ksh_training_audit import digest
    from learning.multi_agent.ma_agent import MAAgent
    original = MAAgent.train_epoch

    def epoch(self):
        network = self.model.a2c_network
        if not hasattr(self, '_local_audit_epoch'):
            self._local_audit_epoch = self.epoch_num
            self._local_audit_before = {
                name: digest(module) for name, module in (
                    ('actor', network.actor_encoder), ('rms', self.running_mean_std),
                    ('coordination', network.coordination), ('head', network.action_head))}
        result = original(self)
        if self.epoch_num == self._local_audit_epoch + 1:
            task = self.vec_env.env.task
            before = self._local_audit_before
            report = dict(epoch=self.epoch_num, environments=task.num_envs,
                region_goals=bool(task._region_goals),
                actor_encoder_unchanged=digest(network.actor_encoder) == before['actor'],
                scene_normalizer_unchanged=digest(self.running_mean_std) == before['rms'],
                coordination_updated=digest(network.coordination) != before['coordination'],
                action_head_updated=digest(network.action_head) != before['head'],
                optimizer_states=len(self.optimizer.state),
                model_finite=all(torch.isfinite(v).all().item() for v in self.model.state_dict().values()
                    if v.is_floating_point()))
            assert all(report[k] for k in ('region_goals', 'actor_encoder_unchanged',
                'scene_normalizer_unchanged', 'coordination_updated', 'action_head_updated', 'model_finite')), report
            out = Path(self.experiment_dir)
            (out/'startup_verification.json').write_text(json.dumps(report, indent=2)+'\n')
            self.save(str(out/'nn/Stage2CoupledPushRegionGoals_startup'))
            print('COUPLED_REGION_GOALS_UPDATES_VERIFIED', json.dumps(report), flush=True)
        return result
    MAAgent.train_epoch = epoch


if __name__ == '__main__':
    install()
    if '--test' not in sys.argv:
        install_training_audit()
    runpy.run_path(str(PROJECT/'tokenhsi/run.py'), run_name='__main__')

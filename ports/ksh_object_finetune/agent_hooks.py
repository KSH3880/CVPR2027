"""Fresh optimizer, original object tokenizer, frozen actor/RMS, trainable critic."""
import json
from pathlib import Path

from finetune_contract import SCHEMA, freeze_actor, frozen_snapshot, verify_frozen


def install(output):
    import torch
    from learning.multi_agent.ma_agent import MAAgent
    old_init, old_train = MAAgent.__init__, MAAgent.set_train
    old_stats, old_save = MAAgent.get_stats_weights, MAAgent.save
    old_set_weights = MAAgent.set_weights

    def init(self, base_name, config):
        old_init(self, base_name, config)
        task = self.vec_env.env.task
        cfg = task.cfg['env']['objectFineTune']
        from native_hooks import PORT
        source = PORT / cfg['initial_checkpoint']
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        # A fresh adaptation run inherits tensors/statistics, never the source
        # optimizer, epoch/frame counters, env state or teacher distillation.
        mode = cfg.get('actor_mode', 'tokenizer_only')
        loaded = self.model.load_state_dict(checkpoint['model'], strict=False)
        expected = {n for n in self.model.state_dict() if n.startswith(('a2c_network.action_adapter.', 'a2c_network.actor_encoder.adapted_object_tokenizer.'))}
        if set(loaded.missing_keys) != expected or loaded.unexpected_keys:
            raise RuntimeError('Unexpected checkpoint architecture mismatch')
        if mode == "copied_adapter":
            from object_tokenizer_copy import sync_copy
            sync_copy(self.model.a2c_network.actor_encoder)
        self.set_stats_weights(checkpoint)
        selected = freeze_actor(self.model, self.vec_env.env.task.cfg['env']['objectFineTune'].get('actor_mode', 'tokenizer_only'))
        self.optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad],
            lr=float(self.last_lr), eps=1e-8, weight_decay=self.weight_decay)
        self.running_mean_std.eval()
        if self._normalize_amp_input: self._amp_input_mean_std.eval()
        self._ft_frozen = frozen_snapshot(self.model)
        self._ft_initial = {n: p.detach().cpu().clone() for n,p in self.model.named_parameters() if p.requires_grad}
        self._ft_obs_rms = {n:v.cpu().clone() for n,v in self.running_mean_std.state_dict().items()}
        self._ft_amp_rms = {n:v.cpu().clone() for n,v in self._amp_input_mean_std.state_dict().items()}
        self._ft_metadata = dict(schema=SCHEMA, source_checkpoint=str(source),
            variant=cfg['variant'],
            source_epoch=checkpoint['epoch'], actor_trainable=('actor_encoder.adapted_object_tokenizer' if mode == 'copied_adapter' else 'actor_encoder.tokenizers.1'), actor_mode=mode,
            critic_trainable=True, discriminator_trainable=True, observation_size=634,
            object_features=30, new_modules=(mode in ('adapter', 'copied_adapter')),
            original_object_tokenizer_frozen=(mode == 'copied_adapter'), observation_rms='frozen', amp_rms='frozen',
            native_assets=task._ft_profile_specs, rsi='native_exact_asset_screened')
        report = dict(self._ft_metadata, trainable_parameters=selected,
                      initial_model_exact_copy=all(torch.equal(v.cpu(),checkpoint['model'][n].cpu())
                        for n,v in self.model.state_dict().items() if n in checkpoint['model']))
        if not report['initial_model_exact_copy']:raise RuntimeError('Initialization did not preserve source tensors')
        Path(self.experiment_dir,'finetune_initialization.json').write_text(json.dumps(report,indent=2))

    def set_train(self):
        old_train(self)
        self.running_mean_std.eval()
        if self._normalize_amp_input: self._amp_input_mean_std.eval()

    def stats(self):
        state = old_stats(self)
        if hasattr(self, '_ft_metadata'): state['object_finetune'] = self._ft_metadata
        return state

    def check(self):
        frozen = verify_frozen(self.model, self._ft_frozen)
        params = dict(self.model.named_parameters())
        changed = [n for n,v in self._ft_initial.items() if not torch.equal(params[n].detach().cpu(),v)]
        for module, reference in ((self.running_mean_std,self._ft_obs_rms),
                                  (self._amp_input_mean_std,self._ft_amp_rms)):
            if any(not torch.equal(v.detach().cpu(), reference[n]) for n,v in module.state_dict().items()):
                raise RuntimeError('Frozen normalization statistics changed')
        report = dict(frozen_actor_parameters_unchanged=frozen, observation_rms_unchanged=True,
            amp_rms_unchanged=True, changed_trainable_parameters=changed, epoch=self.epoch_num,
            action_head_updated=any('action_head.' in n for n in changed),
            adapter_updated=any('action_adapter.' in n for n in changed),
            actor_object_updated=any('actor_encoder.tokenizers.1.' in n for n in changed),
            adapted_object_updated=any('actor_encoder.adapted_object_tokenizer.' in n for n in changed),
            critic_updated=any('critic_encoder.' in n or 'value_head.' in n for n in changed),
            discriminator_updated=any('_disc_' in n for n in changed))
        Path(self.experiment_dir,'finetune_update_audit.json').write_text(json.dumps(report,indent=2))
        return report

    def save(self, filename):
        check(self)
        old_save(self, filename)

    def set_weights(self, weights):
        if weights.get('object_finetune',{}).get('schema') != SCHEMA:
            raise ValueError('Use the private initialization path for base weights; resume requires native fine-tune metadata')
        if weights['object_finetune']['variant'] != self.vec_env.env.task.cfg['env']['objectFineTune']['variant']:
            raise ValueError('Fine-tune variant mismatch')
        old_set_weights(self, weights)

    def restore(self, filename):
        weights = torch.load(filename, map_location=self.ppo_device, weights_only=False)
        self.set_full_state_weights(weights)
        freeze_actor(self.model, self.vec_env.env.task.cfg['env']['objectFineTune'].get('actor_mode', 'tokenizer_only'))
        self._ft_frozen = frozen_snapshot(self.model)
        self._ft_initial = {n:p.detach().cpu().clone() for n,p in self.model.named_parameters() if p.requires_grad}

    MAAgent.__init__ = init
    MAAgent.set_train = set_train
    MAAgent.get_stats_weights = stats
    MAAgent.set_weights = set_weights
    MAAgent.restore = restore
    MAAgent.save = save

import torch

from depbias_ablation import DependencyBiasAblation


class TwoPersonShuffle(DependencyBiasAblation):
    def transform(self, bias, order, label):
        if self.mode != 'shuffled' or bias.shape[-2:] != (2, 2):
            raise ValueError('Identity control requires two-person shuffled mode')
        result = super().transform(bias, order, label)
        if not torch.equal(result, bias):
            raise ValueError('Two-person shuffle changed attention bias')
        return result

    def finish(self, steps):
        try:
            super().finish(steps)
        except ValueError as error:
            if str(error) != 'Intervention never changed bias':
                raise
        else:
            raise ValueError('Two-person shuffle unexpectedly changed bias')
        identity = torch.arange(2, device=self.permutation.device).expand_as(self.permutation)
        if not torch.equal(self.permutation, identity):
            raise ValueError('Two-person permutation is not identity')
        return dict(mode=self.mode, permutation_seed=self.seed, actor_and_critic=True,
            fixed_in_canonical_person_ids=True, calls=self.calls, checks=self.checks,
            degenerate_two_person_shuffle=True, bias_identity_verified_every_call=True)

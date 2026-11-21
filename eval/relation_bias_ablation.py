from types import MethodType

import torch


class RelationBiasAblation:
    def __init__(self, network, mode, seed=None):
        self.relation = {'enabling_off': 2, 'coupled_off': 3}[mode]
        self.mode = mode
        self.enabled = False
        self.permutation = None
        self.bindings = []
        self.calls = {}
        self.checks = {}
        for branch, coordination in (('actor', network.coordination),
                                     ('critic', network.critic_coordination)):
            for index, block in enumerate(coordination.blocks):
                self.bind(block, f'{branch}/{index}')

    def bind(self, block, label):
        original = block.effective_relation_bias

        def output(instance, embedding):
            bias = original(embedding)
            if not self.enabled:
                return bias
            result = bias.clone()
            result[:, self.relation] = 0
            self.calls[label] = self.calls.get(label, 0) + 1
            if label not in self.checks:
                others = [kind for kind in range(4) if kind != self.relation]
                check = dict(target_zero=not bool(result[:, self.relation].any()),
                             other_relations_exact=torch.equal(bias[:, others], result[:, others]),
                             original_target_max_abs=float(bias[:, self.relation].abs().max()))
                if not check['target_zero'] or not check['other_relations_exact']:
                    raise ValueError('Selective relation bias removal failed')
                self.checks[label] = check
            return result

        self.bindings.append((block, original))
        block.effective_relation_bias = MethodType(output, block)

    def reset_counts(self):
        self.calls = {}

    def finish(self, steps):
        for block, original in self.bindings:
            block.effective_relation_bias = original
        if len(self.calls) != len(self.bindings) or any(n != steps for n in self.calls.values()):
            raise ValueError('Selective bias removal missing on a policy frame')
        if not any(check['original_target_max_abs'] for check in self.checks.values()):
            raise ValueError('Selected bias was already zero')
        return dict(mode=self.mode, relation_id=self.relation, actor_and_critic=True,
                    inference_only=True, calls=self.calls, checks=self.checks,
                    relation_labels_and_fusion_unchanged=True)

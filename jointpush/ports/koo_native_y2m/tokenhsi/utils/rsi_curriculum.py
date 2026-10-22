"""Reset-skill probabilities indexed by the absolute PPO checkpoint epoch."""
import math


class SkillInitCurriculum:
    def __init__(self, skills, final_prob, config):
        self.skills = tuple(skills)
        self.final_prob = self._validate_prob(final_prob, 'skillInitProb')
        self.initial_prob = self._validate_prob(config['initialProb'], 'initialProb')
        self.warmup_epochs = config['warmupEpochs']
        self.end_epoch = config['endEpoch']
        if any(isinstance(v, bool) or not isinstance(v, int)
               for v in (self.warmup_epochs, self.end_epoch)):
            raise ValueError('RSI warmupEpochs and endEpoch must be integers')
        if not 0 <= self.warmup_epochs < self.end_epoch:
            raise ValueError('RSI requires 0 <= warmupEpochs < endEpoch')

    def _validate_prob(self, values, name):
        values = tuple(float(value) for value in values)
        if len(values) != len(self.skills) or not values:
            raise ValueError(name + ' must have one probability per skill')
        if (any(not math.isfinite(value) or value < 0 for value in values)
                or not math.isclose(sum(values), 1.0, abs_tol=1e-6)):
            raise ValueError(name + ' must be finite, nonnegative and sum to 1')
        return values

    def at_epoch(self, epoch):
        blend = min(1.0, max(0.0, (epoch - self.warmup_epochs)
                            / (self.end_epoch - self.warmup_epochs)))
        prob = tuple((1.0 - blend) * initial + blend * final
                     for initial, final in zip(self.initial_prob, self.final_prob))
        return prob, blend

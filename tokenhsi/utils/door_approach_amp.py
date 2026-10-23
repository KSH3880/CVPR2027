"""Optional per-owner locomotion reward gate; disabled means full AMP throughout."""
import math
import torch


def validate(spec):
    fields={'full_distance','interaction_distance','hand_distance','angle_threshold','spawn_distance'}
    if set(spec)-{'gate_enabled'}!=fields or any(isinstance(spec[k],bool) or not isinstance(spec[k],(int,float)) or not math.isfinite(spec[k]) or spec[k]<=0 for k in fields):
        raise ValueError('Invalid Door approach AMP thresholds')
    if 'gate_enabled' in spec and type(spec['gate_enabled']) is not bool:
        raise ValueError('Door AMP gate_enabled must be boolean')
    if not spec['interaction_distance'] < spec['full_distance'] <= spec['spawn_distance']:
        raise ValueError('Door AMP requires interaction < full <= spawn distance')


class DoorApproachAMP:
    def __init__(self, num_envs, num_agents, device, spec):
        validate(spec)
        self.spec=spec
        self.interacting=torch.zeros(num_envs,num_agents,dtype=torch.bool,device=device)

    def reset(self, env_ids):
        self.interacting[env_ids]=False

    def step(self, roots, hands, handles, angles):
        if not self.spec.get('gate_enabled', True):
            return torch.ones_like(angles)
        distance=(roots[...,:2]-handles[...,:2]).norm(dim=-1)
        hand_distance=(hands-handles.unsqueeze(-2)).norm(dim=-1).amin(-1)
        s=self.spec
        self.interacting |= ((distance<=s['interaction_distance']) |
            (hand_distance<=s['hand_distance']) | (angles.abs()>=s['angle_threshold']))
        weight=((distance-s['interaction_distance']) /
            (s['full_distance']-s['interaction_distance'])).clamp(0,1)
        return weight.masked_fill(self.interacting,0)


def apply_reward_gate(amp_rewards, weights):
    if amp_rewards['disc_rewards'].shape != weights.shape:
        raise ValueError('Door AMP gate must match every rollout sample')
    return dict(amp_rewards, disc_rewards=amp_rewards['disc_rewards']*weights)

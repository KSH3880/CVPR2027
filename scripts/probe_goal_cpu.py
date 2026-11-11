"""CPU-only counterfactual goal sensitivity; not a success-rate evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tokenhsi'), str(ROOT)]
import numpy as np
import torch
import yaml
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from learning.transformer.amp_network_builder_transformer import AMPTransformerMultiTaskBuilder
from learning.amp_models import ModelAMPContinuous
from learning.push_kd_losses import normalize_policy


def describe(x):
    x = x.detach().cpu().numpy()
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                p10=float(np.quantile(x, .1)), p90=float(np.quantile(x, .9)), max=float(x.max()))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    params = yaml.safe_load((ROOT/'tokenhsi/data/cfg/train/rlg/push_kd.yaml').read_text())['params']['network']
    builder = AMPTransformerMultiTaskBuilder(); builder.load(params)
    info = dict(onehot_size=5, tota_subtask_obs_size=169, each_subtask_obs_size=[20,38,42,27,42],
                each_subtask_obs_indx=[0,20,58,100,127,169],
                each_subtask_name=['traj','sit','carry','climb','push'], enable_task_mask_obs=True)
    model = ModelAMPContinuous.Network(builder.build('amp', actions_num=32, input_shape=(397,),
        num_seqs=1, amp_input_shape=(1340,), self_obs_size=223, task_obs_size=174,
        multi_task_info=info, device='cpu')).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'], strict=True)
    rms = RunningMeanStd((397,)).eval()
    rms.load_state_dict(checkpoint['running_mean_std'])
    initial_stats = {k:v.clone() for k,v in rms.state_dict().items()}
    trace_path = ROOT/'output_etc/check/backend_parity/trace_shared_fixed_rng.pt'
    records = torch.load(trace_path, map_location='cpu', weights_only=True)['records']
    # Actual environment observations after fixed-action physics steps, not synthetic bodies.
    x = torch.cat([r['obs'] for r in records if r['stage'].startswith('step_')])
    assert x.shape == (96,397) and (x[:,396] == 1).all()
    box = x[:,356:359]
    forward = torch.stack((-x[:,360], x[:,359]), -1)
    forward /= forward.norm(dim=-1, keepdim=True)
    variants = {'original':x.clone(), 'repeat':x.clone()}
    for degree in (-40, 0, 40):
        angle = np.deg2rad(degree)
        direction = torch.stack((np.cos(angle)*forward[:,0]-np.sin(angle)*forward[:,1],
                                 np.sin(angle)*forward[:,0]+np.cos(angle)*forward[:,1]), -1)
        for distance in (2., 5.):
            v = x.clone()
            v[:,389:391] = box[:,:2] + distance*direction
            variants[f'angle{degree}_distance{int(distance)}'] = v
    # Masked CARRY token should not affect a PUSH action.
    inactive = x.clone(); inactive[:,320:323] += 1.
    variants['inactive_carry_goal_control'] = inactive
    actions, sigmas, normalized, clipping = {}, {}, {}, {}
    for key,v in variants.items():
        if key.startswith('angle'):
            assert torch.equal(v[:,:389], x[:,:389]) and torch.equal(v[:,391:], x[:,391:])
        z = normalize_policy(v, rms, 5)
        mu, logstd = model.a2c_network.eval_actor(z,v)
        assert torch.isfinite(mu).all()
        actions[key], sigmas[key], normalized[key] = mu, logstd.exp().expand_as(mu), z
        before = (v[:,389:392]-rms.running_mean[389:392].float())/torch.sqrt(rms.running_var[389:392].float()+rms.epsilon)
        clipping[key] = dict(goal_component_fraction=float((before.abs()>5).float().mean()),
                             goal_xy_fraction=float((before[:,:2].abs()>5).float().mean()))
    comparisons = {}
    pairs = [('original','repeat'),('original','inactive_carry_goal_control')]
    for distance in (2,5):
        pairs.append((f'angle-40_distance{distance}', f'angle40_distance{distance}'))
    for angle in (-40,0,40):
        pairs.append((f'angle{angle}_distance2',f'angle{angle}_distance5'))
    for a,b in pairs:
        delta = actions[b]-actions[a]
        effective = actions[b].clamp(-1,1)-actions[a].clamp(-1,1)
        # Sigma is fixed; analytic KL avoids rl_games epsilon bias.
        assert torch.equal(sigmas[a],sigmas[b])
        comparisons[a+' -> '+b] = dict(action_rms=describe(delta.square().mean(-1).sqrt()),
            clipped_action_rms=describe(effective.square().mean(-1).sqrt()),
            rms_in_sigma=describe((delta/sigmas[a]).square().mean(-1).sqrt()),
            analytic_kl=describe(.5*(delta/sigmas[a]).square().sum(-1)),
            identical_actions=int((delta.abs().amax(-1)<1e-7).sum()),
            normalized_goal_distance=describe((normalized[b][:,389:392]-normalized[a][:,389:392]).norm(dim=-1)))
    assert torch.equal(actions['original'],actions['repeat'])
    assert all(torch.equal(v,rms.state_dict()[k]) for k,v in initial_stats.items())
    assert not torch.cuda.is_initialized()
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()), epoch=checkpoint['epoch'], device='cpu',
        cuda_initialized=False, observation_source=str(trace_path), observations=len(x), scenes=8,
        frames_per_scene=12, goal_indices=[389,390,391], changed_indices=[389,390],
        caveat='Correlated fixed-action trace states, not current-policy rollouts or success-rate evidence. Goal direction is within box local +Y +/-40 degrees; no new scene collision screening.',
        sigma=describe(sigmas['original']), clipping=clipping, comparisons=comparisons)
    (out/'result.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'counterfactuals.npz', observations=x.numpy(),
        **{'obs_'+k:v.numpy() for k,v in variants.items()}, **{'mu_'+k:v.numpy() for k,v in actions.items()})
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()

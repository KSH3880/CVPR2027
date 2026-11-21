from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tokenhsi'), str(ROOT/'eval'), str(ROOT/'FSM/dct'),
               str(ROOT.parent/'tokenhsi_push'), str(ROOT.parent/'tokenhsi_push/scripts')]
import isaacgym
import torch
from learning.multi_agent import formal_eval
from utils import coupled_push_spec as coupled, stage2_eval_sizes
from stage2_protocol_eval import capture_initial
import stage2_scalability_layout
import stage2_xy100
import stage2_full_observation
from common import MODELS, write
from policy import load, check_weights
from single_carry_sizes import CARRY_MODE, apply_original9

MODELS['tokenhsi'] = ROOT.parent/'tokenhsi_push/checkpoints/TokenHSIPushKD_00040500.pth'
VARIANT = os.environ['SINGLE_SIZE']
if VARIANT not in ('native', 'common'):
    raise ValueError('Expected native/common sizes')


def validate_single(env):
    if (env['numAgents'], env['numObjects']) != (1, 2):
        raise ValueError('Expected one person and two object tokens')
    equivalent = deepcopy(env)
    equivalent.update(numAgents=4, numObjects=8)
    equivalent['stage2JointPush']['physical_boxes'] = coupled.physical_box_count(4, 8)
    stage2_scalability_layout.original_validate(equivalent)


stage2_scalability_layout.validate_env = validate_single
stage2_scalability_layout.install()
stage2_xy100.install()
if os.environ.get('REFERENCE_ONLY') == '1':
    formal_eval.run_formal_eval = capture_initial
else:
    import stage2_distractor_physics
import stage2_protocol_sampling

original_bank = stage2_eval_sizes.prepare_intersection_bank


def bank(sizes, masses, env, preset):
    episode = original_bank(sizes, masses, env, preset)
    _, groups, _, templates = episode
    primary = coupled.physical_assignment(templates, groups)[:, 0]
    kinds = templates[:, 0]
    if VARIANT == 'common':
        # A separate generator preserves task assignment and the layout RNG stream.
        generator = torch.Generator(device=sizes.device).manual_seed(17000+int(os.environ['PYTHONHASHSEED']))
        common = torch.stack([torch.randint(lo, hi+1, (len(sizes),), device=sizes.device,
                              generator=generator)/20 for lo, hi in ((8, 12), (8, 12), (7, 10))], -1)
        rows = (kinds != 5).nonzero().flatten()
        slot = torch.where(kinds == 4, torch.full_like(primary, 3), primary)
        density = masses[rows, slot[rows]]/sizes[rows, slot[rows]].prod(-1)
        sizes[rows, slot[rows]] = common[rows]
        masses[rows, slot[rows]] = density*common[rows].prod(-1)
    if CARRY_MODE == 'original9':
        apply_original9(sizes, masses, kinds, primary, int(os.environ['PYTHONHASHSEED']))
    return episode


stage2_eval_sizes.prepare_intersection_bank = bank
original_eval = formal_eval.run_formal_eval


@torch.no_grad()
def evaluate(player):
    task = player.env.task
    path = Path(task.cfg['args'].output_path)
    model = os.environ['SINGLE_MODEL']
    task.cfg['env']['episodeLength'] = int(os.environ.get('SINGLE_STEPS', '900'))
    action = player.get_action
    native = saved = None
    calls = 0
    hooks = []
    encoder_calls = dict(actor=0, critic=0)

    def hook(name):
        def check(module, args, kwargs):
            if kwargs.get('object_valid') is not None:
                raise ValueError('Observed distractors were masked')
            encoder_calls[name] += 1
        return check

    def act(obs, deterministic=False):
        nonlocal native, saved, calls
        if calls == 0:
            expected = torch.load(Path(os.environ['STAGE2_REFERENCE_ROOT'])/
                f'general_seed{task.cfg["args"].seed}'/'observed_scene.pt', map_location='cpu', weights_only=False)
            if not torch.equal(obs['obs'].cpu(), expected):
                raise ValueError('Actual first policy input differs from shared scene')
            torch.save(obs['obs'].cpu(), path/'first_policy_input.pt')
            if model == 'tokenhsi':
                from tokenhsi_table_policy import TokenHSIPolicy
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    native = TokenHSIPolicy(MODELS[model], task.device)
                if native.saved['epoch'] != 40500:
                    raise ValueError('Wrong TokenHSI checkpoint')
            else:
                native, saved = load(player, model, obs['obs'])
                for name, encoder in (('actor', player.model.a2c_network.actor_encoder),
                                      ('critic', player.model.a2c_network.critic_encoder)):
                    hooks.append(encoder.register_forward_pre_hook(hook(name), with_kwargs=True))
            write(path/'first_policy_input.json', dict(exact=True, model=model, variant=VARIANT))
        calls += 1
        if model == 'tokenhsi':
            from tokenhsi_table_policy import project_scene
            raw = project_scene(obs['obs'], 1, 2)
            value = native.action(raw)
            if calls == 1:
                torch.save(dict(raw=raw.cpu(), normalized=native.normalize(raw).cpu(), action=value.cpu()),
                           path/'tokenhsi_first_input.pt')
            return value
        return action(obs, deterministic)

    player.get_action = act
    try:
        result = original_eval(player)
    finally:
        player.get_action = action
        for handle in hooks:
            handle.remove()
    if native is not None:
        proof = native.finish()
    else:
        check_weights(player, saved)
        proof = dict(epoch=saved['epoch'], weights_unchanged=True, normalization_unchanged=True)
    if calls != result['simulated_steps']:
        raise ValueError('Policy call count differs')
    if model != 'tokenhsi' and encoder_calls != dict(actor=calls, critic=calls):
        raise ValueError('Unexpected policy encoder calls')
    proof.update(model=model, actual_action_calls=calls, first_policy_input_exact=True)
    result.update(policy=model, size_variant=VARIANT, carry_sizes=CARRY_MODE, scene_mode='independent', xy_extent_percent=100,
        comparison_protocol='table1_independent_single_size', policy_verification=proof,
        encoder_attention_checks=encoder_calls,
        distractor_attention=('unsupported_fixed397D' if model == 'tokenhsi' else 'all_observed_tokens'))
    write(path/'policy_verification.json', proof)
    write(path/'result.json', result)
    return result


if os.environ.get('REFERENCE_ONLY') != '1':
    formal_eval.run_formal_eval = evaluate
runpy.run_path(str(ROOT/'tokenhsi/run.py'), run_name='__main__')

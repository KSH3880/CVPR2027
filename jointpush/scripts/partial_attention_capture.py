"""Finite, read-only Partial13000 actor attention probe; bootstrap stays intact."""
import os
import sys
import runpy
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'jointpush/ports/jhh_raw_cuboid'
sys.path[:0] = [str(SOURCE / 'tokenhsi'), str(SOURCE), str(ROOT)]
sys.dont_write_bytecode = True
assert not os.environ.get('CUDA_MPS_PIPE_DIRECTORY'), 'Probe must bypass MPS'
assert os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['PARTIAL_ATTN_GPU_UUID']
import isaacgym
import torch
import numpy as np
import torch.nn.functional as F


def digest(player):
    h = hashlib.sha256()
    for mapping in (player.model.state_dict(), player.running_mean_std.state_dict()):
        for key, value in sorted(mapping.items()):
            h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(player):
    task = player.env.task
    out = Path(task.cfg['args'].output_path)
    out.mkdir(parents=True, exist_ok=True)
    net = player.model.a2c_network
    encoder, coord = net.actor_encoder, net.coordination
    player.model.eval(); player.running_mean_std.eval()
    player._eval = False; player._reward_debug = False
    before = digest(player)
    obs = player.env_reset(); player.get_batch_size(obs['obs'], 1)
    n, m = task.num_envs, task.num_agents
    active = torch.ones(n, dtype=torch.bool, device=task.device)
    frames = {'stage1': [], 'stage2': []}; states = []; errors = []
    action_errors = []

    def probe(normalized):
        records = {'stage1': [], 'stage2': []}; hooks = []; pending = {}

        def stage1_pre(layer, args, kwargs):
            x, bias = args[:2]
            b, length, width = x.shape
            q, k, v = layer.qkv(x).view(b, length, 3, layer.num_heads, layer.head_dim).permute(2, 0, 3, 1, 4)
            g, gi = kwargs.get('gta_g'), kwargs.get('gta_ginv')
            if g is not None:
                q = apply_gta_transform(g.transpose(-1, -2), q)
                k = apply_gta_transform(gi, k); v = apply_gta_transform(gi, v)
            qk = (q @ k.transpose(-1, -2)) * layer.scale
            bias = torch.zeros_like(qk) if bias is None else bias.expand_as(qk)
            logits = qk + bias; attention = logits.softmax(-1)
            z = attention @ v
            if g is not None: z = apply_gta_transform(g, z)
            z = z.transpose(1, 2).reshape(b, length, width)
            fusion = kwargs.get('task_action_fusion')
            if fusion is not None: z = fusion(z, kwargs['task_condition'], False)
            z = layer.norm1(x + layer.proj(z))
            pending[id(layer)] = layer.norm2(z + layer.linear2(layer.act(layer.linear1(z))))
            records['stage1'].append([a.detach().cpu().numpy() for a in (qk, bias, logits, attention)])

        def post(layer, args, result):
            errors.append(float((result - pending.pop(id(layer))).abs().max()))

        def stage2_pre(block, args):
            humans, tokens, relations, padding = args
            assert tokens is None
            x = block.query_norm(humans); att = block.attention
            b, length, width = x.shape; heads = att.num_heads; dim = width // heads
            q, k, v = F.linear(x, att.in_proj_weight, att.in_proj_bias).chunk(3, -1)
            q, k, v = [t.view(b, length, heads, dim).transpose(1, 2) for t in (q, k, v)]
            qk = (q @ k.transpose(-1, -2)) * dim ** -.5
            bias = block.relation_bias[:, relations].permute(1, 0, 2, 3).masked_fill(padding, float('-inf'))
            logits = qk + bias; attention = logits.softmax(-1)
            z = (attention @ v).transpose(1, 2).reshape(b, length, width)
            z = humans + att.out_proj(z)
            pending[id(block)] = z + block.ffn(block.ffn_norm(z))
            records['stage2'].append([a.detach().cpu().numpy() for a in (qk, bias, logits, attention)])

        for layer in encoder.layers:
            hooks += [layer.register_forward_pre_hook(stage1_pre, with_kwargs=True), layer.register_forward_hook(post)]
        for block in coord.blocks:
            hooks += [block.register_forward_pre_hook(stage2_pre), block.register_forward_hook(post)]
        old_forward = encoder.forward; old_shuffle = coord.shuffle_human_order
        encoder.forward = lambda scene, **kw: old_forward(scene, token_order=torch.arange(sum(encoder.entity_counts), device=scene.device), **kw)
        coord.shuffle_human_order = False
        try: canonical, _ = net.eval_actor(normalized)
        finally:
            encoder.forward = old_forward; coord.shuffle_human_order = old_shuffle
            for hook in hooks: hook.remove()
        return records, canonical

    with torch.no_grad():
        for step in range(121):
            if step in (0, 30, 60, 120) and active.any():
                cpu_rng = torch.get_rng_state(); gpu_rng = torch.cuda.get_rng_state_all()
                normalized = player._preproc_obs(obs['obs'])
                records, canonical = probe(normalized)
                original, _ = net.eval_actor(normalized)
                action_errors.append(float((canonical - original).abs().max()))
                torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(gpu_rng)
                for branch in frames: frames[branch].append(np.stack(records[branch]))
                states.append(dict(step=step, active=active.cpu().tolist(), assignment=task._agent_box_assignment.cpu().tolist(), regions=task._agent_region_assignment.cpu().tolist(), humans=task._humanoid_root_states[..., :3].cpu().tolist(), goals=task._tar_pos.cpu().tolist()))
                print('ATTENTION_SAMPLE', step, 'active', int(active.sum()), flush=True)
            action = player.get_action(obs, True)
            assert torch.isfinite(action).all()
            obs, _, done, _ = player.env_step(player.env, action)
            if not isinstance(obs, dict): obs = player.obs_to_torch(obs)
            assert torch.isfinite(obs['obs']).all()
            active &= ~done.reshape(n, m).any(-1)
            if not active.any(): break
    torch.cuda.synchronize()
    assert digest(player) == before
    assert max(errors) < 2e-4, max(errors)
    assert max(action_errors) < 2e-4, max(action_errors)
    for branch, values in frames.items():
        array = np.stack(values)
        np.savez_compressed(out / (branch + '_attention.npz'), **{key: array[:, :, i].transpose(0, 2, 1, 3, 4, 5) for i, key in enumerate(('qk', 'bias', 'logits', 'attention'))})
    (out / 'states.json').write_text(json.dumps(states, indent=2))
    report = dict(checkpoint=task.cfg['args'].checkpoint, gpu_uuid=os.environ['PARTIAL_ATTN_GPU_UUID'], mps=False, agents=m, objects=task.num_objects, scenes=n, seed=task.cfg['args'].seed, frames=len(states), model_rms_unchanged=True, layer_reconstruction_max_error=max(errors), canonical_action_max_error=max(action_errors), layout='frame,scene,layer,head,query,key', components=['QK/sqrt(d) before bias', 'additive bias', 'QK/sqrt(d)+bias logits', 'softmax(logits) actual attention'], scope='Actor Stage1 frozen scene encoder and Stage2 human coordination within the same Partial13000 model. First termination censors the entire scene; no reset samples.', later_layer_note='Later-layer QK includes effects of previous biased layers.', stage1_labels=['H0','H1','H2','H3']+['B'+str(i) for i in range(8)]+['G0','G1','G2','G3'], stage2_labels=['H0','H1','H2','H3'])
    (out / 'verification.json').write_text(json.dumps(report, indent=2))
    print('PARTIAL_ATTENTION_VERIFIED', json.dumps(report), flush=True)


original_run_path = runpy.run_path
def bootstrap(path, *args, **kwargs):
    if Path(path) == SOURCE / 'tokenhsi/run.py':
        global apply_gta_transform
        from learning.multi_agent.ma_players import MAPlayerContinuous
        from learning.multi_agent.amp_network_builder_ma import apply_gta_transform
        MAPlayerContinuous.run = run
    return original_run_path(path, *args, **kwargs)
runpy.run_path = bootstrap
original_run_path(str(ROOT / 'jointpush/scripts/partial_run.py'), run_name='__main__')

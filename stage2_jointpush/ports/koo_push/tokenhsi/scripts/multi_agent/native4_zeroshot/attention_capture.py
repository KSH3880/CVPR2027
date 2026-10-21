"""Read-only actor attention capture in canonical human/door/box/goal order."""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


def transform(matrix, features):
    blocks = features.reshape(*features.shape[:-1], features.shape[-1] // 4, 4)
    return torch.einsum('blij,bhlrj->bhlri', matrix.to(features.dtype), blocks).reshape_as(features)


def attention_parts(qkv, heads, bias=None, gta_g=None, gta_ginv=None):
    batch, length, width = qkv.shape
    dim = width // 3
    q, k, _ = qkv.reshape(batch, length, 3, heads, dim // heads).permute(2, 0, 3, 1, 4)
    if gta_g is not None:
        q = transform(gta_g.transpose(-1, -2), q)
        k = transform(gta_ginv, k)
    scores = (q @ k.transpose(-1, -2)) * (1. / math.sqrt(dim // heads))
    semantic = torch.zeros_like(scores) if bias is None else torch.zeros_like(scores) + bias
    return scores, semantic, torch.softmax(scores + semantic, dim=-1)


class AttentionRecorder:
    def __init__(self, encoder):
        self.encoder = encoder
        self.original_forward = encoder.forward
        self.enabled = False
        self.parts = {}
        self.context = {}
        self.handles = []

        def forward(obs, return_all=False, token_order=None):
            length = sum(encoder.entity_counts)
            if token_order is None:
                token_order = (torch.randperm(length, device=obs.device)
                               if encoder.relation_graph_spec.get('shuffle_token_order', False)
                               else torch.arange(length, device=obs.device))
            self.inverse = token_order.argsort()
            return self.original_forward(obs, return_all=return_all, token_order=token_order)

        encoder.forward = forward
        for index, layer in enumerate(encoder.layers):
            def before(module, args, _index=index):
                if self.enabled:
                    self.context[_index] = (args[1], args[4], args[5])

            def qkv_hook(module, args, output, _index=index, _heads=layer.num_heads):
                if self.enabled:
                    bias, gta_g, gta_ginv = self.context[_index]
                    values = attention_parts(output.detach(), _heads, bias, gta_g, gta_ginv)
                    self.parts[_index] = tuple(value.index_select(-2, self.inverse)
                                              .index_select(-1, self.inverse).detach().cpu().numpy()
                                              for value in values)

            self.handles.append(layer.register_forward_pre_hook(before))
            self.handles.append(layer.qkv.register_forward_hook(qkv_hook))

    def close(self):
        self.encoder.forward = self.original_forward
        for handle in self.handles:
            handle.remove()


def render_images(output, frames, steps, states):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    humans = frames.shape[-1] // 4
    length = frames.shape[-1]
    labels = [f'{kind}{agent}' for kind in ('H', 'D', 'B', 'G') for agent in range(humans)]
    directory = output / 'attention_png'
    directory.mkdir(exist_ok=True)
    count = 0
    for frame, step in enumerate(steps):
        for scene in range(frames.shape[1]):
            if not states[frame]['active'][scene]:
                continue
            figure, axes = plt.subplots(4, 2, figsize=(11, 15) if humans == 2 else (15, 20), layout='constrained')
            for layer in range(4):
                for head in range(2):
                    ax = axes[layer, head]
                    values = frames[frame, scene, layer, head]
                    im = ax.imshow(values, vmin=0, vmax=1, cmap='magma', interpolation='nearest')
                    ax.set_xticks(range(length), labels, fontsize=8, rotation=45 if humans == 4 else 0)
                    ax.set_yticks(range(length), labels, fontsize=8)
                    ax.set_xlabel('Key (read from)', fontsize=9)
                    ax.set_ylabel('Query (updated token)', fontsize=9)
                    ax.set_title(f'Actor layer {layer + 1} / head {head + 1}', fontsize=10)
                    for row in range(length):
                        for col in range(length):
                            if values[row, col] >= .02:
                                ax.text(col, row, f'{values[row, col]:.2f}', ha='center', va='center',
                                        color='black' if values[row, col] > .65 else 'white', fontsize=7)
                    for boundary in (humans - .5, 2 * humans - .5, 3 * humans - .5):
                        ax.axhline(boundary, color='cyan', lw=.5, alpha=.5)
                        ax.axvline(boundary, color='cyan', lw=.5, alpha=.5)
            tasks = ['OPEN' if value == 0 else 'PUSH' for value in states[frame]['tasks'][scene]]
            task_title = '  '.join(f'H{i}:{task}' for i, task in enumerate(tasks))
            figure.suptitle(f'Original Koo Stage1 | scene {scene} | step {step} ({step / 30:.2f}s)\n'
                            f'{task_title} | H=Human D=Door B=Box G=Goal\n'
                            'Indices are canonical owner slots; physical lanes are in states.json', fontsize=11)
            figure.colorbar(im, ax=axes, shrink=.7, label='Softmax attention (row sums to 1)')
            figure.savefig(directory / f'scene_{scene:03d}_step_{step:06d}.png', dpi=140)
            plt.close(figure)
            count += 1
    return count


def capture_run(self):
    from attention_probe import tensor_digest

    task = self.env.task
    output = Path(os.environ['ATTENTION_OUT'])
    self.model.eval()
    self.running_mean_std.eval()
    self._eval = self._reward_debug = False
    humans = task.num_agents
    assert humans in (2, 4) and task.num_objects == 2 * humans
    encoder = self.model.a2c_network.actor_encoder
    recorder = AttentionRecorder(encoder)
    before = tensor_digest(*self.model.state_dict().values(), *self.running_mean_std.state_dict().values())
    obs = self.env_reset()
    self.get_batch_size(obs['obs'], 1)
    wanted = {int(value) for value in os.environ.get('ATTENTION_STEPS', '0,30,60,120,180,299').split(',')}
    horizon = int(os.environ.get('PROBE_STEPS', '300'))
    active = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
    first_end = [None] * task.num_envs
    frames, steps, states = [], [], []
    max_action_difference = None
    for step in range(horizon):
        recorder.enabled = step in wanted
        if recorder.enabled:
            states.append(dict(step=step, active=active.cpu().tolist(), tasks=task.relation_runtime.tasks.cpu().tolist(),
                               physical_door_lanes=task._agent_lanes.cpu().tolist(),
                               physical_box_binding=task._push_binding.cpu().tolist(),
                               roots=task._humanoid_root_states.cpu().tolist(),
                               logical_object_states=task._logical_box_values(task._box_states).cpu().tolist(),
                               goals=task._tar_pos.cpu().tolist()))
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        with torch.no_grad():
            action = self.get_action(obs, True)
        if step == 0:
            # Verify recording does not change the original policy's action.
            cpu_after, gpu_after = torch.get_rng_state(), torch.cuda.get_rng_state()
            recorder.enabled = False
            wrapped = encoder.forward
            encoder.forward = recorder.original_forward
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(gpu_rng)
            with torch.no_grad():
                reference = self.get_action(obs, True)
            max_action_difference = (reference - action).abs().max().item()
            assert max_action_difference == 0., 'Recorder changed original actor output'
            encoder.forward = wrapped
            torch.set_rng_state(cpu_after)
            torch.cuda.set_rng_state(gpu_after)
        if step in wanted:
            assert len(recorder.parts) == len(encoder.layers)
            frames.append(tuple(np.stack([recorder.parts[layer][part] for layer in range(4)], axis=1)
                                for part in range(3)))
            steps.append(step)
        obs, reward, done, info = self.env_step(self.env, action)
        if not isinstance(obs, dict):
            obs = self.obs_to_torch(obs)
        ended = done.reshape(task.num_envs, humans).any(-1).bool()
        for scene in (active & ended).nonzero().flatten().tolist():
            first_end[scene] = dict(step=step + 1, fallen=task._terminate_buf.reshape(task.num_envs, humans)[scene].bool().cpu().tolist())
        active &= ~ended
        if not active.any():
            break
    recorder.close()
    assert before == tensor_digest(*self.model.state_dict().values(), *self.running_mean_std.state_dict().values())
    qk, bias, attention = [np.stack([frame[part] for frame in frames]) for part in range(3)]
    assert np.isfinite(attention).all() and np.allclose(attention.sum(-1), 1., atol=1e-6)
    np.savez_compressed(output / 'attention_raw.npz', attention=attention, qk_scores=qk,
                        relation_bias=bias, steps=np.array(steps))
    (output / 'states.json').write_text(json.dumps(states, indent=2) + '\n')
    png_count = render_images(output, attention, steps, states)
    summary = dict(agents=humans, physical_boxes=humans, physical_doors=humans, layers=4, heads=2,
                   task_graph=task.cfg['args'].task_graph, capture_steps=steps,
                   attention_shape=list(attention.shape), axis_order=['step', 'scene', 'layer', 'head', 'query', 'key'],
                   token_labels=[f'{kind}{agent}' for kind in ('H', 'D', 'B', 'G') for agent in range(humans)],
                   png_images=png_count, original_action_max_difference=max_action_difference,
                   policy_unchanged=True, first_ends=first_end,
                   note='Attention weights describe information mixing, not causal importance; inactive scenes after first termination are not rendered')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('ATTENTION_CAPTURE_SUMMARY', json.dumps(summary), flush=True)

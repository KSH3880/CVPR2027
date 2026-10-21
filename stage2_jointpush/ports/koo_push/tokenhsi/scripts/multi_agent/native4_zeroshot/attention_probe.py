"""Full versus two-scene attention, on identical physical native4 worlds."""
import hashlib
import json
import os
from pathlib import Path

import torch

SCOPES = ('pair', 'none_zero', 'hide_humans', 'hide_doors', 'hide_boxes', 'hide_goals', 'hide_boxes_goals', 'hide_objects_goals', 'hide_box_first', 'hide_box_second', 'remove_doors',
          'keep_ordinary_0', 'keep_ordinary_1', 'keep_ordinary_2', 'keep_ordinary_3')


def tensor_digest(*values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def scene_groups(task):
    # Physical boxes 0,1 and 2,3 came from the two duplicated native scenes.
    # Logical door slots follow their own independently shuffled physical lanes.
    owner_scene = task._push_binding // 2
    return torch.cat((owner_scene, task._agent_lanes // 2,
                      owner_scene, owner_scene), dim=1)


def zero_none_bias(bias, suffix):
    """Keep valid task edges and SELF; zero only unconnected off-diagonal pairs."""
    from utils.edge_stage1_spec import parse_semantic_packet
    valid, src, dst, _, _ = parse_semantic_packet(suffix)
    length = bias.shape[-1]
    occupied = torch.zeros(len(suffix), length * length, dtype=torch.long,
                           device=bias.device).scatter_add(1, src * length + dst, valid.long())
    occupied = occupied.bool().reshape(-1, length, length)
    none = ~occupied & ~torch.eye(length, dtype=torch.bool, device=bias.device)[None]
    return bias.masked_fill(none[None, :, None], 0.)


def attention_allowed(task, order, scope):
    groups = scene_groups(task).index_select(1, order)
    same = groups[:, :, None] == groups[:, None, :]
    assert same.sum(-1).eq(8).all()
    if scope == 'pair':
        return same
    if scope in ('hide_box_first', 'hide_box_second'):
        # Physical IDs 0,1 / 2,3 define the duplicated pairs. Select one
        # physical rank per pair, regardless of shuffled logical owner slots.
        selected = torch.zeros_like(scene_groups(task), dtype=torch.bool)
        rank = 0 if scope == 'hide_box_first' else 1
        selected[:, 8:12] = task._push_binding.remainder(2) == rank
        selected = selected.index_select(1, order)
        allowed = same | ~selected[:, None, :]
        assert allowed.sum(-1).eq(15).all(), 'Exactly one foreign box key must be hidden'
        return allowed
    # Canonical order: 4 humans, 4 doors, 4 boxes, 4 goals.
    kinds = torch.arange(4, device=order.device).repeat_interleave(4).index_select(0, order)
    key_selected = (kinds == 0 if scope == 'hide_humans' else
                    kinds == 1 if scope == 'hide_doors' else
                    kinds == 2 if scope == 'hide_boxes' else
                    kinds == 3 if scope == 'hide_goals' else
                    kinds >= 1 if scope == 'hide_objects_goals' else kinds >= 2)
    return same | ~key_selected[None, None, :]


def install_attention_scope(encoder, task, control):
    original_encoder = encoder.forward
    original_fusion = encoder.context_fusion.forward

    def fusion_forward(suffix, *args, **kwargs):
        bias = original_fusion(suffix, *args, **kwargs)
        return zero_none_bias(bias, suffix) if control['scope'] == 'none_zero' else bias

    encoder.context_fusion.forward = fusion_forward
    for layer in encoder.layers:
        original_layer = layer.forward

        def layer_forward(x, rel_bias=None, *args, _original=original_layer,
                          _layer=layer, **kwargs):
            keep = getattr(_layer, '_probe_keep', None)
            if keep is not None:
                # Remove all door queries/keys/values, not just foreign-door edges.
                # Scatter back only to preserve the surrounding encoder's readout
                # indices; the door slots never participate in any actor layer.
                reduced_bias = (None if rel_bias is None else
                                rel_bias.index_select(-2, keep).index_select(-1, keep))
                reduced_args = list(args)
                for index in (0, 1):  # Optional pairwise geometry score/message.
                    if len(reduced_args) > index and reduced_args[index] is not None:
                        reduced_args[index] = reduced_args[index].index_select(2, keep).index_select(3, keep)
                for index in (2, 3):  # GTA transforms.
                    if len(reduced_args) > index and reduced_args[index] is not None:
                        reduced_args[index] = reduced_args[index].index_select(1, keep)
                reduced = _original(x.index_select(1, keep), reduced_bias,
                                    *reduced_args, **kwargs)
                return x.index_copy(1, keep, reduced)
            allowed = getattr(_layer, '_probe_allowed', None)
            if allowed is not None:
                mask = x.new_zeros(x.shape[0], 1, x.shape[1], x.shape[1])
                mask.masked_fill_(~allowed[:, None], float('-inf'))
                rel_bias = mask if rel_bias is None else rel_bias + mask
            return _original(x, rel_bias, *args, **kwargs)

        layer.forward = layer_forward

    def encoder_forward(obs, return_all=False, token_order=None):
        length = sum(encoder.entity_counts)
        if token_order is None:
            token_order = (torch.randperm(length, device=obs.device)
                           if encoder.relation_graph_spec.get('shuffle_token_order', False)
                           else torch.arange(length, device=obs.device))
        partition_scope = control['scope'].startswith('keep_ordinary_')
        active = control['scope'] not in ('full', 'none_zero', 'remove_doors') and not partition_scope
        allowed = attention_allowed(task, token_order, control['scope']) if active else None
        humans = encoder.entity_counts[0]
        keep = ((token_order < humans) | (token_order >= 2 * humans)).nonzero().flatten()
        assert len(keep) == length - humans
        if partition_scope:
            ordinary = int(control['scope'].rsplit('_', 1)[1])
            assert encoder.entity_counts == [4, 8, 4]
            keep = ((token_order < 8 + ordinary) | (token_order >= 12)).nonzero().flatten()
            assert len(keep) == 12 + ordinary
        for layer in encoder.layers:
            layer._probe_allowed = allowed
            layer._probe_keep = keep if control['scope'] == 'remove_doors' or partition_scope else None
        return original_encoder(obs, return_all=return_all, token_order=token_order)

    encoder.forward = encoder_forward


def evaluate_attention(self):
    scope = os.environ['NATIVE4_ATTENTION_SCOPE']
    comparison = os.environ.get('NATIVE4_ATTENTION_COMPARE', 'pair')
    if comparison not in SCOPES or scope not in ('full', comparison):
        raise ValueError('Unknown attention comparison scope')
    task = self.env.task
    self.model.eval()
    self.running_mean_std.eval()
    self._eval = self._reward_debug = False
    encoder = self.model.a2c_network.actor_encoder
    control = {'scope': scope}
    install_attention_scope(encoder, task, control)
    before = tensor_digest(*self.model.state_dict().values(),
                           *self.running_mean_std.state_dict().values())
    obs = self.env_reset()
    self.get_batch_size(obs['obs'], 1)
    initial = tensor_digest(task._humanoid_root_states, task._dof_pos,
                            task._dof_vel, task._box_states, task._tar_pos,
                            getattr(task, '_agent_lanes', task._agent_box_assignment),
                            getattr(task, '_push_binding', task._agent_box_assignment))
    n, m = task.num_envs, task.num_agents
    active = torch.ones(n, dtype=torch.bool, device=task.device)
    seen_proximity = torch.zeros(n, m, dtype=torch.bool, device=task.device)
    rows = torch.arange(n, device=task.device)[:, None]
    trials, action_rows = [], []
    horizon = int(os.environ.get('PROBE_STEPS', '300'))
    from utils.door_reference import rotate

    for step in range(1, horizon + 1):
        # Shadow inference restores RNG so each rollout consumes exactly one action
        # call's randomness. Both policies see the same observation for this comparison.
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        actions = {}
        for candidate in ('full', comparison):
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(gpu_rng)
            control['scope'] = candidate
            with torch.no_grad():
                actions[candidate] = self.get_action(obs, True)
        delta = (actions['full'] - actions[comparison]).reshape(n, m, -1)
        selected = delta[active]
        action_rows.append(dict(step=step, rms=selected.square().mean().sqrt().item(),
                                max_abs=selected.abs().max().item(),
                                per_agent_rms=selected.square().mean((0, 2)).sqrt().tolist()))
        control['scope'] = scope
        obs, reward, done, info = self.env_step(self.env, actions[scope])
        if not isinstance(obs, dict):
            obs = self.obs_to_torch(obs)
        boxes = task._box_states[rows, task._agent_box_assignment]
        inverse = boxes[..., 3:7].clone()
        inverse[..., :3] *= -1
        bodies = task._rigid_body_pos
        local = rotate(inverse[:, :, None].expand(n, m, bodies.shape[2], 4),
                       bodies - boxes[:, :, None, :3])
        sizes = task._box_size[rows, task._agent_box_assignment]
        clearance = (local.abs() - sizes[:, :, None] / 2).clamp_min(0).norm(dim=-1).amin(-1)
        seen_proximity |= (clearance < .2) & active[:, None]
        fall = task._terminate_buf.reshape(n, m).bool()
        finished = active & (done.reshape(n, m).any(-1) | (step == horizon))
        for env in finished.nonzero().flatten().tolist():
            trials.append(dict(trial=env, steps=step, scene_fall=bool(fall[env].any()),
                               fallen_agents=fall[env].nonzero().flatten().tolist(),
                               fallen_before_assigned_box_proximity=(fall[env] & ~seen_proximity[env]).nonzero().flatten().tolist(),
                               assigned_box_proximity_seen=seen_proximity[env].tolist(),
                               root_height=task._humanoid_root_states[env, :, 2].tolist()))
        active &= ~finished
        if not active.any():
            break
    assert len(trials) == n
    assert before == tensor_digest(*self.model.state_dict().values(),
                                    *self.running_mean_std.state_dict().values())
    result = dict(scope=scope, comparison=comparison, agents=m, scenes=n, horizon_steps=horizon,
                  initial_state_sha256=initial, policy_unchanged=True,
                  scene_fall_rate=sum(t['scene_fall'] for t in trials) / n,
                  scenes_with_preproximity_fall=sum(bool(t['fallen_before_assigned_box_proximity']) for t in trials),
                  mean_first_end_steps=sum(t['steps'] for t in trials) / n,
                  first_step_action=action_rows[0], action_comparison=action_rows, trials=trials,
                  mask=('Partition task region tokens retained; ordinary box tokens kept: ' + comparison.rsplit('_', 1)[1] + '; no physical doors; critic and physical boxes retained'
                        if comparison.startswith('keep_ordinary_') else
                        f'All {m} door tokens removed from actor Q/K/V, GTA and bias matrices: {4*m} to {3*m} attention tokens; physical doors and critic retained'
                        if comparison == 'remove_doors' else
                        'All tokens/QKV retained; only actor NONE bias zeroed; task/SELF and critic unchanged'
                        if comparison == 'none_zero' else
                        'All actor layers block cross-scene attention to selected key types: ' + comparison +
                        '; unselected types can still carry indirect information; critic unchanged'),
                  proximity='Any body point within 0.2m of its assigned box; not a contact sensor')
    Path(os.environ['PROBE_OUT']).write_text(json.dumps(result, indent=2) + '\n')
    print('ATTENTION_PROBE_SUMMARY', json.dumps({k: v for k, v in result.items()
                                                if k not in ('trials', 'action_comparison')}), flush=True)

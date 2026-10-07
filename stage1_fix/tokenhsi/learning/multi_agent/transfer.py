"""Strict, reported carry -> OnTop weight transfer without optimizer/history restore."""
from utils.relation_task_spec import STATE_MODE


def transfer_carry_weights(model, checkpoint, target_reward):
    saved = checkpoint.get('relation_metadata', {})
    if saved.get('reward_mode') != STATE_MODE:
        raise ValueError('OnTop transfer requires a state_relation_v0 carry checkpoint')
    source_reward = saved.get('relation_reward_config', {})
    expected = {k: v for k, v in target_reward.items()
                if k not in ('mode', 'ontop', 'prerequisite', 'diagnostics')}
    actual = {k: v for k, v in source_reward.items() if k not in ('mode', 'diagnostics')}
    if actual != expected:
        raise ValueError('Carry transfer reward mismatch; Holding k and base reward must match')
    source, target = checkpoint['model'], model.state_dict()
    if set(source) != set(target):
        raise ValueError('Transfer model keys differ: ' + str(sorted(set(source) ^ set(target))))
    copied, expanded = [], []
    prepared = {}
    for key, value in target.items():
        old = source[key]
        if old.shape == value.shape:
            prepared[key] = old
            copied.append(key)
        elif (key.endswith('edge_encoder.relation_embed.weight')
              and tuple(old.shape) == (8, 32) and tuple(value.shape) == (9, 32)):
            new = value.clone()
            new[:8].copy_(old)
            prepared[key] = new
            expanded.append(key)
        else:
            raise ValueError('Unexpected transfer shape mismatch: ' + key)
    if len(expanded) != 2:
        raise ValueError('Expected exactly two expanded relation embeddings (actor and critic)')
    model.load_state_dict(prepared, strict=True)
    return dict(source_epoch=checkpoint.get('epoch'), copied_tensors=copied,
                expanded_embeddings=expanded, new_relation_rows=[8],
                optimizer_restored=False, counters_restored=False, env_state_restored=False,
                additional_freeze=False)

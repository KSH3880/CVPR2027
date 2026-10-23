"""Keep the source tokenizer and route explicitly to a separately trained copy."""
import copy


def install_copy(encoder):
    encoder.adapted_object_tokenizer = copy.deepcopy(encoder.tokenizers[1])
    encoder.use_adapted_object = True


def sync_copy(encoder):
    encoder.adapted_object_tokenizer.load_state_dict(encoder.tokenizers[1].state_dict(), strict=True)


def select_tokenizer(encoder, index, original):
    if index == 1 and getattr(encoder, 'use_adapted_object', False):
        return encoder.adapted_object_tokenizer
    return original


def set_branch(network, branch):
    if branch not in ('base', 'adapt'): raise ValueError(branch)
    if not hasattr(network.actor_encoder, 'adapted_object_tokenizer'):
        raise ValueError('Explicit branch selection requires a copied tokenizer')
    network.actor_encoder.use_adapted_object = branch == 'adapt'
    network.use_action_adapter = branch == 'adapt'

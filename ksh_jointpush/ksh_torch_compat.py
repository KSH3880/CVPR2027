"""Keep frozen KSH boolean reductions compatible with PyTorch 2.0."""
import inspect
import textwrap
import torch


def install_attention_compat():
    """PyTorch2.0 native MHA mishandles per-head floating attention biases."""
    if torch.__version__.split('.')[:2] != ['2', '0']:
        return
    original = torch.nn.MultiheadAttention.forward
    if getattr(original, '_ksh_mask_compat', False):
        return
    def forward(self, query, key, value, *args, **kwargs):
        mask = kwargs.get('attn_mask', args[2] if len(args) > 2 else None)
        if (not self.training and query is key and key is value
                and mask is not None and mask.is_floating_point()):
            # An identity view selects the ordinary attention implementation;
            # training mode, dropout, weights, mask and gradients are unchanged.
            query = query.view_as(query)
        return original(self, query, key, value, *args, **kwargs)
    forward._ksh_mask_compat = True
    torch.nn.MultiheadAttention.forward = forward


def _single_dim_any(module, name):
    original = getattr(module, name)
    source = inspect.getsource(original)
    # These immutable/local functions reduce boolean tensors over axes1/2.
    # Reducing higher axes first preserves the original axis numbering.
    updated = source.replace('.any((1,2))', '.any(2).any(1)')
    updated = updated.replace('.any((1, 2))', '.any(2).any(1)')
    if updated == source:
        return False
    namespace = original.__globals__
    exec(compile(updated, original.__code__.co_filename, 'exec'), namespace)
    setattr(module, name, namespace[name])
    return True


def install():
    install_attention_compat()
    try:
        torch.zeros(1, 1, dtype=torch.bool).any((0, 1))
    except TypeError:
        from ksh_jointpush import ksh_agent_rsi, ksh_initialization_safety
        from jointpush import ksh_agent_rsi as legacy_agent_rsi
        from utils import stage2_jointpush_scene, stage2_jointpush_rsi
        targets = [(ksh_agent_rsi, 'mixed_body_overlap'),
                   (legacy_agent_rsi, 'mixed_body_overlap'),
                   (ksh_initialization_safety, 'invalid_initialization'),
                   (stage2_jointpush_scene, 'scene_infeasible'),
                   (stage2_jointpush_rsi, 'coupled_body_overlap')]
        patched = [module.__name__ + '.' + name for module, name in targets
                   if not getattr(getattr(module, name), '_ksh_any_compat', False)
                   and _single_dim_any(module, name)]
        for module, name in targets:
            getattr(module, name)._ksh_any_compat = True
        print('KSH_TORCH_COMPAT', dict(torch_version=torch.__version__, functions=patched), flush=True)
        return patched
    return []


def install_cuda_volume_compat():
    """CUDA11 NVRTC cannot compile prod kernels for Blackwell's sm_120.

    The Stage2 volumes and three-axis grid count use explicit multiplication,
    preserving values while avoiding NVRTC-generated reduction kernels.
    """
    if (not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10
            or int(torch.version.cuda.split('.')[0]) >= 12):
        return []
    # Scripted quaternion interpolation otherwise asks old NVRTC to compile
    # sm_120 fusion kernels. Eager/precompiled kernels implement the same ops.
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_nvfuser_enabled(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    from utils import stage2_jointpush, size_rsi
    from env.tasks.multi_agent.humanoid_ma_stage2_jointpush import HumanoidMAStage2JointPush
    targets = [(stage2_jointpush, 'asset_bank', 'sizes', '.prod(-1)'),
               (HumanoidMAStage2JointPush, '_load_box_asset', 'self._box_size', '.prod(-1)'),
               (size_rsi, 'task_probabilities', '(bounds[:, 1] - bounds[:, 0] + 1)', '.prod()')]
    patched = []
    for owner, name, expression, reduction in targets:
        original = getattr(owner, name)
        if getattr(original, '_ksh_volume_compat', False):
            continue
        source = textwrap.dedent(inspect.getsource(original))
        before = expression + reduction
        after = '(' + '*'.join(expression + '[..., ' + str(axis) + ']' for axis in range(3)) + ')'
        if source.count(before) != 1:
            raise ValueError('Unexpected Stage2 volume expression: ' + name)
        namespace = dict(original.__globals__)
        exec(compile(source.replace(before, after), original.__code__.co_filename, 'exec'), namespace)
        replacement = namespace[name]
        # Preserve aliases already imported by task_role_spec and base tasks.
        original.__code__ = replacement.__code__
        original._ksh_volume_compat = True
        patched.append(name)
    print('KSH_CUDA_VOLUME_COMPAT', dict(cuda=torch.version.cuda, functions=patched), flush=True)
    return patched

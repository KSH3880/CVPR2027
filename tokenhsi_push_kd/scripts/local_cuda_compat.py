"""Opt-in CUDA 11/Blackwell compatibility shared by local simulator processes."""

def install():
    import isaacgym
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10 and int(torch.version.cuda.split('.')[0]) < 12:
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        # CUDA 11 NVRTC cannot compile prod reductions for Blackwell. This
        # three-axis integer product has an equivalent precompiled-kernel form.
        import inspect
        from push_reference.utils import size_rsi
        original = size_rsi.task_probabilities
        source = inspect.getsource(original)
        before = '(bounds[:, 1] - bounds[:, 0] + 1).prod()'
        assert source.count(before) == 1
        after = '((bounds[0, 1] - bounds[0, 0] + 1) * (bounds[1, 1] - bounds[1, 0] + 1) * (bounds[2, 1] - bounds[2, 0] + 1))'
        namespace = dict(original.__globals__)
        exec(compile(source.replace(before, after), original.__code__.co_filename, 'exec'), namespace)
        original.__code__ = namespace['task_probabilities'].__code__

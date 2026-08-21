"""Minimal horovod.torch stand-in backed by torch.distributed.

rl_games 1.1.4 drives multi-GPU training through horovod, which cannot be built
against the locally compiled torch. Only the handful of calls in
rl_games/distributed/hvd_wrapper.py are needed, so they are implemented here.
Put this directory on PYTHONPATH; launch with torchrun.

Collectives run over NCCL on the GPU. Set TOKENHSI_DDP_BACKEND=gloo to fall back
to CPU-side collectives (needed only if NCCL is unavailable — note that the NCCL
bundled inside torch 2.4 hangs on sm_120, so torch here links NCCL 2.27).
"""
import os

import torch
import torch.distributed as dist

_BACKEND = os.environ.get("TOKENHSI_DDP_BACKEND", "nccl")


def init(*args, **kwargs):
    if dist.is_initialized():
        return
    r = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", 0)))
    w = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", 1)))
    l = int(os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", r)))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(l)
    dist.init_process_group(_BACKEND, rank=r, world_size=w)
    _patch_rl_games()


def _patch_rl_games():
    """rl_games' sync_stats assumes every stat is a tensor, which breaks under
    mixed_precision (the GradScaler contributes plain floats). Skip non-tensors;
    the scaler's scale is a per-rank numerical detail and need not be shared."""
    try:
        from rl_games.distributed import hvd_wrapper
    except ImportError:
        return

    def sync_stats(self, algo):
        for key, group in algo.get_stats_weights().items():
            for name, val in group.items():
                if isinstance(val, torch.Tensor):
                    val.data = allreduce(val, name=key + name)
        algo.curr_frames = allreduce(torch.tensor(algo.curr_frames), average=False).item()

    hvd_wrapper.HorovodWrapper.sync_stats = sync_stats


def _ensure():
    if not dist.is_initialized():
        init()


def _to_comm(t):
    """NCCL needs CUDA tensors, gloo needs CPU ones."""
    return t.cuda() if _BACKEND == "nccl" else t.cpu()


def rank():
    _ensure()
    return dist.get_rank()


def size():
    _ensure()
    return dist.get_world_size()


def local_rank():
    return int(os.environ.get("LOCAL_RANK", rank()))


def allreduce(tensor, name=None, average=True, op=None):
    _ensure()
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    buf = _to_comm(tensor.detach()).clone()
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    if average:
        buf = buf / size()
    return buf.to(tensor.device)


def broadcast_parameters(params, root_rank=0):
    _ensure()
    items = params.items() if isinstance(params, dict) else params
    for _, p in items:
        if not isinstance(p, torch.Tensor):
            continue
        buf = _to_comm(p.data)
        dist.broadcast(buf, src=root_rank)
        p.data.copy_(buf)


def broadcast_optimizer_state(optimizer, root_rank=0):
    _ensure()
    for group in optimizer.param_groups:
        for key, val in group.items():
            if key == "params" or not isinstance(val, (int, float)):
                continue
            buf = _to_comm(torch.tensor([float(val)]))
            dist.broadcast(buf, src=root_rank)
            group[key] = type(val)(buf.item())
    for state in optimizer.state.values():
        for val in state.values():
            if isinstance(val, torch.Tensor):
                buf = _to_comm(val.data)
                dist.broadcast(buf, src=root_rank)
                val.data.copy_(buf)


class _DistributedOptimizer:
    """Averages gradients across ranks before delegating to the real optimizer."""

    def __init__(self, optimizer, named_parameters=None, **kwargs):
        self._opt = optimizer

    def step(self, closure=None):
        world = size()
        if world > 1:
            grads = [p.grad for g in self._opt.param_groups for p in g["params"] if p.grad is not None]
            if grads:
                flat = torch._utils._flatten_dense_tensors(grads)
                buf = _to_comm(flat)
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                buf /= world
                for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(buf.to(flat.device), grads)):
                    g.copy_(synced)
        return self._opt.step(closure)

    def __getattr__(self, name):
        return getattr(self.__dict__["_opt"], name)

    def __setattr__(self, name, value):
        if name == "_opt":
            self.__dict__[name] = value
        else:
            setattr(self.__dict__["_opt"], name, value)


def DistributedOptimizer(optimizer, named_parameters=None, **kwargs):
    return _DistributedOptimizer(optimizer, named_parameters, **kwargs)

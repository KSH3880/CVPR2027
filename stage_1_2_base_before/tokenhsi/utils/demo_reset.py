"""Bound placement work only for explicitly requested demo searches."""
import os


class DemoResetRejected(RuntimeError):
    pass


def bounded_demo_reset(task):
    args=task.cfg['args']
    return (os.environ.get('RANDOM_DEMO_SEARCH')=='1'
            and os.environ.get('RANDOM_DEMO_BOUNDED_RESET')=='1'
            and bool(args.test and args.eval and args.headless)
            and not getattr(args,'eval_protocol',''))


def task_only_demo_boxes(task):
    args=task.cfg['args']
    return (os.environ.get('RANDOM_DEMO_SEARCH')=='1'
            and os.environ.get('RANDOM_DEMO_TASK_BOXES_ONLY')=='1'
            and bool(args.test and args.eval and args.headless)
            and not getattr(args,'eval_protocol',''))


def task_physical_mask(mapping, task_counts, physical_count):
    import torch
    used=torch.arange(mapping.shape[1],device=mapping.device)[None]<task_counts[:,None]
    counts=torch.zeros(len(mapping),physical_count,dtype=torch.long,device=mapping.device)
    counts.scatter_add_(1,mapping,used.long())
    return counts>0

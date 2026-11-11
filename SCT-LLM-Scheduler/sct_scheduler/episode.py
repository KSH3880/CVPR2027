"""A bounded episode bridge for an already-created, already-reset Stage1 task."""
from pathlib import Path
import json
from .core import require


def run_episode(task, requests, agent_order, scheduler, output_dir, *, physical_gpu,
                observe, policy, step, max_steps, record_step=None):
    """Callbacks: observe()->obs, policy(obs)->actions, step(actions)->done(bool).

    requests must be captured from this task's reset state. One request per env.
    policy must be the frozen Stage1 SCT (including its original normalization).
    step MUST NOT auto-reset individual environments. End the batch on any reset.
    No simulator step happens during LLM inference. No process is spawned here.
    """
    from .isaac_adapter import InitialReleaseFreeze
    require(type(max_steps) is int and max_steps > 0, 'Bounded positive max_steps required')
    require(len(requests) == task.num_envs, 'One request per environment required')
    root = Path(__file__).resolve().parents[1] / 'output_etc/check'
    output = Path(output_dir).resolve()
    require(root.resolve() in output.parents, 'Episode outputs must stay under local output_etc/check/')
    output.mkdir(parents=True, exist_ok=False)
    answers = [scheduler.schedule(request, output / f'env_{i:04d}') for i, request in enumerate(requests)]
    controller = InitialReleaseFreeze(task, requests, answers, agent_order, physical_gpu=physical_gpu)
    import torch
    try:
        with torch.no_grad():
            for frame in range(max_steps):
                controller.before_action(frame)
                actions = controller.mask_actions(policy(observe()))
                done = step(actions)
                require(type(done) is bool, 'step must return a boolean batch termination flag')
                if record_step is not None:
                    record_step((frame + 1) * task.dt, task, controller)
                if done:
                    break
    finally:
        # Normal GPU synchronization; never send signals or MPS commands.
        try:
            torch.cuda.synchronize(task.device)
        finally:
            proof = controller.close()
            (output / 'release_audit.json').write_text(json.dumps(proof, indent=2))
    return answers

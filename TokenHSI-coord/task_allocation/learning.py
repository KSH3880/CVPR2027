"""Testable rollout boundaries and PPO update, independent of Isaac Gym."""
import torch
from task_allocation.core import step_reward

@torch.no_grad()
def macro_step(player, steps, gamma, time_coef, delivery_coef, failure_coef):
    task = player.env.task
    n, device = task.num_envs, task.device
    active = torch.ones(n, dtype=torch.bool, device=device)
    done = torch.zeros_like(active)
    duration = torch.zeros(n, device=device)
    reward = torch.zeros(n, device=device)
    diag = dict(delivered=0, success=0, failure=0, executed_steps=0, held_box_steps=0)
    task._compute_observations()
    obs = torch.clamp(task.obs_buf, -player.env.clip_obs, player.env.clip_obs).to(player.device)
    player.get_batch_size(obs, 1)
    for tick in range(steps):
        action = player.get_action({'obs': obs}, is_determenistic=True)
        obs, _, done_rows, _ = player.env.step(action)
        newly, success = task.update_delivery()
        native_done = done_rows.reshape(n, 2).any(-1)
        terminal = native_done | success
        failure = terminal & ~success
        r = step_reward(newly, failure, task.dt, time_coef, delivery_coef, failure_coef)
        reward += (gamma ** tick) * r * active.float()
        duration += active.float()
        diag['delivered'] += int(newly[active].sum())
        diag['success'] += int((success & active).sum())
        diag['failure'] += int((failure & active).sum())
        diag['executed_steps'] += int(active.sum())
        if hasattr(task,'allocation_locked'):
            diag['held_box_steps'] += int(task.allocation_locked[active].sum())
        done |= terminal & active
        active &= ~terminal
        if not active.any():
            break
    # Do not attribute subsequent episodes to the previous assignment action.
    # Reset only at the macro boundary; finished envs are excluded above.
    ids = done.nonzero(as_tuple=False).flatten()
    if len(ids):
        task.reset(task.agent_rows(ids))
    return reward, done, duration, diag


def ppo_update(policy, optimizer, obs, action, old_logp, old_value, advantages, returns, epochs=4, batch=256):
    flat = {k: torch.cat([x[k] for x in obs]) for k in obs[0]}
    action, old_logp, old_value = (x.flatten() for x in (action, old_logp, old_value))
    advantages, returns = advantages.flatten(), returns.flatten()
    advantages = (advantages-advantages.mean())/advantages.std(unbiased=False).clamp(min=1e-6)
    losses=[]
    for _ in range(epochs):
        for ids in torch.randperm(len(action), device=action.device).split(batch):
            dist, value = policy({k: v[ids] for k,v in flat.items()})
            ratio = (dist.log_prob(action[ids])-old_logp[ids]).exp()
            surrogate = torch.minimum(ratio*advantages[ids], ratio.clamp(.8,1.2)*advantages[ids])
            loss = -surrogate.mean()+.5*(value-returns[ids]).square().mean()-.01*dist.entropy().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite allocation PPO loss')
            optimizer.zero_grad()
            loss.backward()
            nnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            if not torch.isfinite(nnorm):
                raise RuntimeError('nonfinite allocation PPO gradient')
            optimizer.step()
            losses.append(float(loss.detach()))
    return sum(losses)/len(losses)


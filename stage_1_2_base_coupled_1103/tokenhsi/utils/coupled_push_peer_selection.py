import torch


MODES = ('all', 'keep3', 'shrink3')


def validate_mode(mode, task, evaluation, protocol='', resume=0):
    if mode not in MODES:
        raise ValueError('Unknown coupled peer selection: '+mode)
    if mode != 'all' and (task != 'HumanoidMACoupledPush' or not evaluation or protocol or resume):
        raise ValueError('--coupled_peers requires coupled viewer/test inference without eval_protocol or resume')


class CoupledPeerSelection:
    def __init__(self, mode, device, seed):
        if mode not in ('keep3', 'shrink3'):
            raise ValueError('Expected keep3 or shrink3 peer selection')
        self.mode = mode
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.selected = None

    def reset(self, env_ids=None):
        if env_ids is None:
            self.selected = None
        elif self.selected is not None:
            self.selected[env_ids] = False

    def _draw(self, candidates, count):
        chosen = torch.zeros_like(candidates)
        if not (count > 0).any():
            return chosen
        scores = torch.rand(candidates.shape, device=candidates.device, generator=self.generator)
        scores.masked_fill_(~candidates, -1.)
        indices = scores.topk(min(3, candidates.shape[-1]), dim=-1).indices
        slots = torch.arange(indices.shape[-1], device=candidates.device)
        chosen.scatter_(-1, indices, slots < count[..., None])
        return chosen

    def update(self, groups, on_face):
        people = groups.shape[1]
        peers = (groups[:, :, None] == groups[:, None, :])
        peers &= ~torch.eye(people, device=groups.device, dtype=torch.bool)
        limited = peers.sum(-1) > 3
        if self.selected is None:
            self.selected = torch.zeros_like(peers)
        selected = self.selected & peers
        unready = peers & ~on_face[:, None, :]
        if self.mode == 'shrink3':
            selected &= unready
        selected_unready = selected & unready
        needed = (unready.sum(-1).clamp_max(3)-selected_unready.sum(-1)).clamp_min(0)
        needed = torch.where(limited, needed, 0)
        ready_selected = selected & ~unready
        drop = (selected.sum(-1)+needed-3).clamp_min(0)
        selected &= ~(ready_selected & (ready_selected.long().cumsum(-1) <= drop[..., None]))
        selected |= self._draw(unready & ~selected, needed)
        if self.mode == 'keep3':
            empty = torch.where(limited, (3-selected.sum(-1)).clamp_min(0), 0)
            selected |= self._draw(peers & ~selected, empty)
        self.selected = torch.where(limited[..., None], selected, peers)
        return self.selected


class SingleCoupledPeerSelection:
    def __init__(self, mode, groups, seed):
        if mode not in ('keep1', 'shrink1'):
            raise ValueError('Expected keep1 or shrink1 peer selection')
        self.mode = mode
        self.peers = torch.full_like(groups, -1)
        self.generator = torch.Generator(device=groups.device).manual_seed(seed)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.peers.fill_(-1)
        else:
            self.peers[env_ids] = -1

    def update(self, groups, on_face, env_ids=None):
        selected = self.peers.clone() if env_ids is None else self.peers[env_ids]
        same = groups[:, :, None] == groups[:, None, :]
        same &= ~torch.eye(groups.shape[1], device=groups.device, dtype=torch.bool)
        unready = same & ~on_face[:, None, :]
        valid = (selected >= 0) & same.gather(-1, selected.clamp_min(0)[..., None]).squeeze(-1)
        pending = valid & unready.gather(-1, selected.clamp_min(0)[..., None]).squeeze(-1)
        need = unready.any(-1) & ~pending
        if self.mode == 'keep1':
            need |= ~valid & same.any(-1)
            candidates = torch.where(unready.any(-1, keepdim=True), unready, same)
            selected = torch.where(valid, selected, -1)
        else:
            candidates = unready
            selected = torch.where(pending, selected, -1)
        if need.any():
            scores = torch.rand(same.shape, device=groups.device, generator=self.generator)
            choice = scores.masked_fill(~candidates, -1.).argmax(-1)
            selected = torch.where(need, choice, selected)
        if env_ids is None:
            self.peers.copy_(selected)
        else:
            self.peers[env_ids] = selected
        return selected

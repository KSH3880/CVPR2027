"""Physical object activation, distractors and loco reference sampling."""
import torch

class LocoRsi:
    def __init__(self, task):
        self.task = task

    def availability(self, rows, owners, templates):
        valid = torch.zeros(len(rows), len(self.task._skill), dtype=torch.bool, device=rows.device)
        valid[:, self.task._skill.index('loco')] = True
        return valid

    def sample(self, skill, rows, owners, templates):
        if skill != 'loco':
            raise ValueError('Shared support initialization is loco-only')
        lib = self.task._motion_lib['loco']
        clips = lib.sample_motions(len(rows))
        return clips, lib.sample_time_rsi(clips)

class SceneLayoutMixin:
    def _active_physical(self, ids):
        active = torch.zeros(len(ids), self.num_physical_objects, dtype=torch.bool, device=self.device)
        active.scatter_(1, self._logical_box_order[ids], True)
        return active

    def _place_distractors(self, ids):
        mapping = self._logical_box_order[ids]
        task_slots = torch.arange(self.num_objects, device=self.device)[None] < self._task_token_count[ids, None]
        occupied = torch.zeros(len(ids), self.num_physical_objects, dtype=torch.long, device=self.device)
        occupied.scatter_add_(1, mapping, task_slots.long())
        occupied = occupied > 0
        for slot in range(self.num_objects):
            rows = (~task_slots[:, slot]).nonzero().flatten()
            if not len(rows):
                continue
            es, physical = ids[rows], mapping[rows, slot]
            candidate = self._box_states[es, physical, :2].clone()
            sizes = self._box_size[es]
            clearance = sizes[..., :2].norm(dim=-1)/2 + self._box_size[es, physical, :2].norm(dim=-1)[:, None]/2 + .05
            for _ in range(64):
                distance = (candidate[:, None]-self._box_states[es, :, :2]).norm(dim=-1)
                blocked = ((distance < clearance) & occupied[rows]).any(-1)
                human_distance = (candidate[:, None]-self._humanoid_root_states[es, :, :2]).norm(dim=-1)
                blocked |= (human_distance < self._box_min_agent_dist).any(-1)
                if not blocked.any():
                    break
                candidate[blocked] = self._sample_arena_xy(self._env_origins[es[blocked], :2])
            self._box_states[es, physical, :2] = candidate
            occupied[rows, physical] = True


import torch


class CoordinationState:
    def __init__(self, parent, groups):
        self.parent = parent
        self.same = groups[:, :, None] == groups[:, None, :]
        follower = parent >= 0
        enabling = follower.clone()
        rows, people = follower.nonzero(as_tuple=True)
        enabling[rows, parent[rows, people]] = True
        self.coupled = (self.same.sum(-1)>1) & ~enabling
        self.arrived = torch.zeros_like(follower)
        self.released = torch.zeros_like(follower)
        self.frozen = follower.clone()

    def update(self, leader_success, touching, alive):
        predecessor_success = leader_success.gather(1, self.parent.clamp_min(0))
        self.released |= (self.parent>=0) & predecessor_success
        touching = touching & alive
        self.arrived |= self.coupled & touching
        together = (self.arrived[:, None, :] | ~self.same).all(-1)
        self.released |= self.coupled & together
        self.frozen = (((self.parent>=0) | self.arrived) & ~self.released) & alive
        return self.frozen

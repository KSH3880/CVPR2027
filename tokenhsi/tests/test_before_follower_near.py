from types import SimpleNamespace

import torch

from env.tasks.multi_agent.before_follower_near import apply_layout, pair_invalid


def fixture():
    n = 256
    roots = torch.zeros(n, 4, 13)
    roots[..., 0] = torch.tensor([-1.5, 3., -1.5, 3.])
    roots[..., 2] = .9
    roots[..., 4], roots[..., 6] = .1, .99**.5
    bodies = roots[:, :, None].repeat(1, 1, 3, 1)
    bodies[:, :, 1, 2] = .1
    bodies[:, :, 2, 0] += .2
    boxes = torch.zeros(n, 8, 13)
    boxes[..., 2], boxes[..., 6] = .2, 1.
    boxes[:, [1, 3], 0] = 1.5
    boxes[:, 3, 0] = 12.  # ON_TOP starts 9 m away; near mode must resample it.
    before = torch.tensor([-1, 0, -1, 2]).expand(n, -1).clone()
    before[::3, 3] = -1
    return SimpleNamespace(device='cpu',
        _before_predecessor=before,
        _templates=torch.tensor([3, 1, 3, 4]).expand(n, -1).clone(),
        _humanoid_root_states=roots,
        _kinematic_humanoid_rigid_body_states=bodies,
        _box_states=boxes, _box_size=torch.full((n, 8, 3), .4),
        )


def test_random_distance_clearance_and_partial_reset():
    torch.manual_seed(5125)
    task = fixture()
    original = {k: v.clone() for k, v in vars(task).items() if torch.is_tensor(v)}
    ids = torch.arange(0, 256, 2)
    apply_layout(task, ids)
    row, follower = (task._before_predecessor[ids] >= 0).nonzero(as_tuple=True)
    es = ids[row]
    first = task._before_predecessor[es, follower]
    slots = (es[:, None], torch.stack((first, follower), -1))
    roots = task._humanoid_root_states[slots]
    offset = roots[..., :2]-task._box_states[es, first, None, :2]
    distance = offset.norm(dim=-1)
    assert ((distance >= 1.-1e-6) & (distance <= 5.+1e-6)).all()
    assert (distance[:, 1] > 2.).any()
    assert (distance[:, 1] < distance[:, 0]).all()
    assert distance[:, 1].std() > .1
    assert torch.atan2(offset[:, 1, 1], offset[:, 1, 0]).std() > 1.
    assert not pair_invalid(roots, task._kinematic_humanoid_rigid_body_states[slots],
        task._box_states[slots], task._box_size[slots], task._templates[es, follower] == 4).any()
    stack = task._templates[es, follower] == 4
    payload_offset = (task._box_states[es[stack], follower[stack], :2]
                      - roots[stack, 1, :2])
    payload_distance = payload_offset.norm(dim=-1)
    assert ((payload_distance >= 1.-1e-6) & (payload_distance <= 3.+1e-6)).all()
    assert payload_distance.std() > .1
    assert torch.atan2(payload_offset[:, 1], payload_offset[:, 0]).std() > 1.
    for key, value in original.items():
        torch.testing.assert_close(getattr(task, key)[1::2], value[1::2], atol=0, rtol=0)
    solo = ids[ids % 3 == 0]
    torch.testing.assert_close(task._humanoid_root_states[solo, 3], original['_humanoid_root_states'][solo, 3])
    torch.testing.assert_close(task._box_states[solo, 3], original['_box_states'][solo, 3])
    torch.testing.assert_close(task._box_states[:, 1], original['_box_states'][:, 1])
    torch.testing.assert_close(task._box_states[:, [0, 2]], original['_box_states'][:, [0, 2]])
    torch.testing.assert_close(task._box_states[..., 2], original['_box_states'][..., 2], atol=0, rtol=0)
    torch.testing.assert_close(task._kinematic_humanoid_rigid_body_states[..., 2],
                               original['_kinematic_humanoid_rigid_body_states'][..., 2], atol=0, rtol=0)
    old = task._humanoid_root_states[ids].clone()
    apply_layout(task, ids)
    assert not torch.equal(task._humanoid_root_states[ids], old)


def test_seed_reproducibility_and_no_pair_noop():
    a, b = fixture(), fixture()
    ids = torch.arange(256)
    torch.manual_seed(1)
    apply_layout(a, ids)
    torch.manual_seed(1)
    apply_layout(b, ids)
    torch.testing.assert_close(a._humanoid_root_states, b._humanoid_root_states, atol=0, rtol=0)
    torch.testing.assert_close(a._box_states, b._box_states, atol=0, rtol=0)
    a._before_predecessor[:] = -1
    state = torch.get_rng_state()
    old = a._humanoid_root_states.clone()
    apply_layout(a, ids)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(a._humanoid_root_states, old)

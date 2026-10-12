from copy import deepcopy
from types import SimpleNamespace

import torch

from env.tasks.multi_agent.before_reward import gated_success
from env.tasks.multi_agent.before_viewer import apply_layout, BeforeOrder
from utils.before_spec import sample_graph
from utils.edge_stage2_spec import classify_family
from test_before import SPEC


def test_near_layout_uses_physical_binding_and_preserves_other_environments():
    t = SimpleNamespace(_before_families=torch.tensor([1, 2, 3, 0]),
        _env_origins=torch.tensor([[0., 0., 0.], [10., 0., 0.], [0., 10., 0.], [10., 10., 0.]]),
        _logical_box_order=torch.tensor([[3, 1, 0, 2]]).expand(4, -1),
        _box_states=torch.zeros(4, 4, 13), _box_size=torch.full((4, 4, 3), .5),
        _tar_pos=torch.zeros(4, 2, 3), _humanoid_root_states=torch.zeros(4, 2, 13),
        _kinematic_humanoid_rigid_body_states=torch.zeros(4, 2, 3, 13),
        _dof_vel=torch.ones(4, 2, 32), _before_amp_rotation=torch.zeros(4, 2, 4),
        _before_amp_translation=torch.ones(4, 2, 3))
    t._humanoid_root_states[..., 2] = 1.
    t._humanoid_root_states[..., 6] = 1.
    t._kinematic_humanoid_rigid_body_states[:] = t._humanoid_root_states[:, :, None]
    t._kinematic_humanoid_rigid_body_states[:, :, 1, 0] = .1
    t._before_amp_rotation[..., 3] = 1.
    old = deepcopy(t)
    apply_layout(t, torch.tensor([0, 2, 3]))
    for name, value in vars(t).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value[[1, 3]], getattr(old, name)[[1, 3]])
    for row in (0, 2):
        shared = t._box_states[row, t._logical_box_order[row, 0]]
        roots = t._humanoid_root_states[row]
        distances = (roots[:, :2] - shared[:2]).norm(dim=-1)
        assert distances[1] < distances[0]
        torch.testing.assert_close(distances[0], torch.tensor(2.))
        torch.testing.assert_close((t._tar_pos[row, 0, :2]-shared[:2]).norm(), torch.tensor(1.5))
        assert (t._box_states[row, :, 2] == .25).all()
        assert not t._box_states[row, :, 7:].any()
        assert not roots[:, 7:].any() and not t._dof_vel[row].any()
        relative = t._kinematic_humanoid_rigid_body_states[row, :, 1, :3] - roots[:, :3]
        torch.testing.assert_close(relative.norm(dim=-1), torch.full((2,), .1))
    payload = t._box_states[2, t._logical_box_order[2, 1]]
    torch.testing.assert_close((t._humanoid_root_states[2, 1, :2]-payload[:2]).norm(), torch.tensor(.8))
    assert (payload[:2]-t._box_states[2, 3, :2]).norm() > .7


def test_order_distinguishes_early_after_simultaneous_and_gate_reclosure():
    families = torch.tensor([1, 2, 3, 1, 2, 0])
    graph = sample_graph(6, SPEC, families=families)
    tracker = BeforeOrder(graph, families)
    alive = torch.ones(6, dtype=torch.bool)
    def update(seconds, at_rows, b_rows):
        raw = torch.zeros(6, 4, dtype=torch.bool)
        raw[at_rows] |= tracker.at[at_rows]
        raw[b_rows] |= tracker.dependent[b_rows]
        success, gate = gated_success(raw, graph)
        tracker.update(dict(raw_own_success=raw, own_success=success,
                            prerequisite_gate=gate), seconds, alive)
    update(.1, [1, 4], [0])
    update(.2, [0, 1, 3], [0, 1, 3, 4])
    rows = tracker.report()
    assert [r['order'] for r in rows] == ['early', 'after_at', 'incomplete', 'same_step', 'after_at']
    assert rows[0]['raw_b_seconds'] < rows[0]['at_seconds']
    assert rows[0]['gated_b_seconds'] == rows[0]['at_seconds']
    assert rows[0]['raw_b_while_gate_closed']
    assert not rows[1]['raw_b_while_gate_closed']
    assert rows[4]['raw_b_while_gate_closed'] and rows[4]['gated_b_seconds'] is None
    old = deepcopy(rows)
    alive[:] = False
    update(.3, list(range(6)), list(range(6)))
    assert tracker.report() == old
    torch.testing.assert_close(classify_family(graph), families)

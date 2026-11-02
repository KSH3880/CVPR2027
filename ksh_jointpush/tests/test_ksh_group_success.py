import numpy as np
from ksh_jointpush.ksh_group_success import category, complete_success


def test_group_compositions_and_both_groups_must_arrive():
    assert category([2, 1, 2, 2]) == '1+3'
    assert category([2, 1, 2, 1]) == '2+2'
    assert category([3, 3, 3, 3]) == '4'
    flags = np.zeros((5, 4), bool)
    flags[1, :2] = True
    assert not complete_success(flags, [0, 0, 2, 2])
    flags[3, 2:] = True
    assert complete_success(flags, [0, 0, 2, 2])
    assert not complete_success(flags, [0, 1, 2, 3])
    flags[0, 3] = True
    assert not complete_success(flags, [0, 0, 2, 2])


def test_saved_scene_is_independent_and_does_not_keep_other_scene_storage():
    import torch
    from ksh_jointpush.ksh_group_success import scene_snapshot
    state = dict(tensors={'groups': torch.arange(12).reshape(3, 4)},
                 nested={'graph': {'edge_owner': torch.zeros(3, 8)}},
                 roots=torch.zeros(3, 36, 13), dofs=torch.zeros(3, 128, 2),
                 density=torch.ones(3, 24))
    saved = scene_snapshot(state, 1)
    assert saved['tensors']['groups'].tolist() == [[4, 5, 6, 7]]
    state['tensors']['groups'][1] = -1
    assert saved['tensors']['groups'].tolist() == [[4, 5, 6, 7]]
    assert saved['roots'].storage().size() == 36 * 13


def test_center_scene_moves_positions_only_and_preserves_relative_goal():
    import torch
    from ksh_jointpush.ksh_group_success import center_scene
    roots = torch.zeros(36, 13)
    roots[:, :3] = torch.tensor([448., 32., 1.])
    roots[:, 7] = 2.
    saved = dict(roots=roots.clone(), tensors={
        '_env_origins': torch.tensor([[448., 32., 0.]]),
        '_humanoid_root_states': roots[:4][None].clone(),
        '_tar_pos': torch.tensor([[[450., 34., .5]]]),
        '_box_bps': torch.ones(1, 24, 8, 3)})
    centered = center_scene(saved)
    assert centered['roots'][0, :3].tolist() == [0., 0., 1.]
    assert centered['roots'][0, 7] == 2
    assert centered['tensors']['_tar_pos'].tolist() == [[[2., 2., .5]]]
    assert centered['tensors']['_box_bps'].eq(1).all()
    assert centered['tensors']['_env_origins'].eq(0).all()

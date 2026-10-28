import torch

from env.tasks.multi_agent.scene_features import (
    build_env_local_position_features,
    build_gta_pose_records,
)


def test_scene_positions_share_one_coordinate_convention_and_origin_invariance():
    origins = torch.tensor([[10.0, -20.0, 0.0], [-7.0, 4.0, 0.5]])
    local = torch.tensor([
        [[2.0, -1.0, 0.9], [2.0, -1.0, 0.9], [-3.0, 4.0, 0.2]],
        [[1.0, 3.0, 1.1], [1.0, 3.0, 1.1], [4.0, -2.0, 0.4]],
    ])
    world = local + origins.unsqueeze(1)
    features = build_env_local_position_features(world, origins, arena_scale=5.0)

    expected = local.clone()
    expected[..., :2] /= 5.0
    assert torch.allclose(features, expected)
    # The first two slots model different entity types at the same physical point.
    assert torch.equal(features[:, 0], features[:, 1])

    translation = torch.tensor([[31.0, -8.0, 2.0], [-13.0, 9.0, -1.0]])
    shifted = build_env_local_position_features(
        world + translation.unsqueeze(1), origins + translation, arena_scale=5.0)
    assert torch.allclose(shifted, features)


def test_scene_position_feature_shape_validation():
    try:
        build_env_local_position_features(torch.zeros(2, 3), torch.zeros(2, 3), 5.0)
    except ValueError as exc:
        assert "world_positions" in str(exc)
    else:
        raise AssertionError("invalid position shape should be rejected")


def test_gta_pose_records_are_env_local_canonical_and_targets_use_owner_heading():
    origins = torch.tensor([[10.0, -20.0, 0.0]])
    humans = torch.tensor([[[11.0, -18.0, 0.9], [8.0, -19.0, 1.1]]])
    objects = torch.tensor([[[12.0, -22.0, 0.3], [7.0, -17.0, 0.4],
                             [10.5, -20.5, 0.2]]])
    targets = torch.tensor([[[13.0, -16.0, 0.7], [6.0, -21.0, 0.8]]])
    human_heading = torch.tensor([[
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.70710677, 0.70710677],
    ]])
    object_rotation = torch.tensor([[
        [0.1, 0.2, 0.3, 0.9],
        [0.0, 0.0, 0.0, 1.0],
        [0.4, 0.0, 0.0, 0.8],
    ]])

    poses = build_gta_pose_records(
        humans, human_heading, objects, object_rotation, targets, origins)
    assert poses.shape == (1, 7, 7)
    assert torch.equal(poses[0, 0:2, 0:3], humans[0] - origins[0])
    assert torch.equal(poses[0, 2:5, 0:3], objects[0] - origins[0])
    assert torch.equal(poses[0, 5:7, 0:3], targets[0] - origins[0])
    assert torch.equal(poses[0, 0:2, 3:7], human_heading[0])
    assert torch.equal(poses[0, 2:5, 3:7], object_rotation[0])
    assert torch.equal(poses[0, 5:7, 3:7], human_heading[0])

    shift = torch.tensor([[31.0, -8.0, 2.0]])
    shifted = build_gta_pose_records(
        humans + shift.unsqueeze(1), human_heading,
        objects + shift.unsqueeze(1), object_rotation,
        targets + shift.unsqueeze(1), origins + shift)
    assert torch.allclose(shifted, poses, atol=2e-6)

from unittest.mock import patch

from env.tasks.multi_agent import edge_ontop_reward
from utils import evaluation_errors, evaluation_protocol


def install():
    original = evaluation_protocol._spatial_distances
    geometry = edge_ontop_reward.inner_xy_region_error

    def full_surface(point, support, size, margin):
        return geometry(point, support, size, 0.)

    def distances(*args, **kwargs):
        with patch.object(edge_ontop_reward, 'inner_xy_region_error', full_surface):
            return original(*args, **kwargs)

    evaluation_protocol._spatial_distances = distances
    evaluation_errors._spatial_distances = distances

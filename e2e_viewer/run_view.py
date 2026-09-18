"""Native Isaac Gym viewer entry for the external e2e stack planner checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np


# Isaac Gym must be imported before torch, including transitive planner imports.
from isaacgym import gymapi as _gymapi  # noqa: F401

SOURCE_ROOT = Path(os.environ["E2E_SOURCE_ROOT"]).expanduser().resolve()
MASTEER = SOURCE_ROOT / "TokenHSI-masteer"
COORD = SOURCE_ROOT / "TokenHSI-coord"
if not (MASTEER / "tokenhsi" / "run.py").is_file():
    raise FileNotFoundError(f"invalid E2E_SOURCE_ROOT (masteer entry missing): {SOURCE_ROOT}")
if not (COORD / "e2e_planner").is_dir():
    raise FileNotFoundError(f"invalid E2E_SOURCE_ROOT (e2e_planner missing): {SOURCE_ROOT}")

# Match run_e2e_stack.py's import contract. Masteer's tokenhsi tree must win;
# coord is appended only for the otherwise-absent e2e_planner package.
sys.path.insert(0, str(MASTEER / "tokenhsi"))
sys.path.insert(1, str(MASTEER))
sys.path.append(str(COORD))

import utils.parse_task as task_registry  # noqa: E402
from env.tasks.adapt_interaction_skills.humanoid_ma_e2e_stack import (  # noqa: E402
    HumanoidMAE2EStack,
)


class HumanoidMAE2EStackView(HumanoidMAE2EStack):
    """E2E planner runtime with only the installed-command path overlays."""

    def _load_object_asset(self, object_urdfs):
        """Load absolute dataset URDFs without prepending a bogus ``./``."""
        asset_options = _gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = _gymapi.DOF_MODE_NONE
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = _gymapi.COMPUTE_PER_VERTEX
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = _gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 100000
        asset_options.vhacd_params.max_convex_hulls = 128
        asset_options.vhacd_params.max_num_vertices_per_ch = 64
        asset_options.replace_cylinder_with_capsule = False

        object_assets = []
        for urdf in object_urdfs:
            urdf_path = Path(urdf).expanduser().resolve()
            asset = self.gym.load_asset(
                self.sim, str(urdf_path.parent), urdf_path.name, asset_options
            )
            if asset is None:
                raise RuntimeError(f"failed to load object asset: {urdf_path}")
            object_assets.append(asset)
        return object_assets

    def _update_marker(self):
        # The inherited multi-task marker updater submits sit/climb actors that
        # this carry-only stack scene does not use. Physics already owns boxes
        # and targets; path lines need no actor-state write.
        return

    def _draw_task(self):
        """Draw inherited ribbons plus an always-visible route centerline."""
        super()._draw_task()
        if self.viewer is None or not hasattr(self, "_gt_path"):
            return

        paths = self._gt_path.detach().cpu().numpy()
        ends = self._s_end.detach().cpu().numpy()
        colors = np.asarray(
            [[1.0, 0.05, 0.55], [1.0, 0.45, 0.0], [0.65, 0.2, 1.0], [0.1, 0.9, 0.2]],
            dtype=np.float32,
        )
        agents = self.num_agents
        for env_id, env_ptr in enumerate(self.envs):
            for agent in range(agents):
                row = env_id * agents + agent
                end = max(int(ends[row] / 0.1) + 1, 2)
                end = min(end, paths.shape[1])
                points = paths[row, :end]
                vertices = np.concatenate(
                    (points[:-1], np.full((len(points) - 1, 1), 0.09, np.float32),
                     points[1:], np.full((len(points) - 1, 1), 0.09, np.float32)),
                    axis=1,
                ).astype(np.float32)
                line_colors = np.repeat(colors[agent % len(colors)][None], len(vertices), axis=0)
                self.gym.add_lines(
                    self.viewer, env_ptr, len(vertices), vertices, line_colors
                )
        return

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(
            "[e2e-view] pink/orange=installed full routes; "
            "cyan/green=policy steering windows; yellow=aim points",
            flush=True,
        )


task_registry.HumanoidMAE2EStackView = HumanoidMAE2EStackView

from run import main  # noqa: E402

foreign = sorted(
    name
    for name, module in list(sys.modules.items())
    if (name == "tokenhsi" or name.startswith("tokenhsi."))
    and getattr(module, "__file__", None)
    and not os.path.realpath(module.__file__).startswith(str(MASTEER) + os.sep)
)
if foreign:
    raise ImportError(
        "tokenhsi modules loaded outside the selected TokenHSI-masteer: "
        + ", ".join(f"{name}={sys.modules[name].__file__}" for name in foreign)
    )


if __name__ == "__main__":
    main()

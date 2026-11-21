import argparse
import math
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[2]


def install_top_view(task_class):
    from isaacgym import gymapi
    from utils.sensor_stage2_spec import WAREHOUSE
    original = task_class._update_camera

    def update_camera(task):
        if task.viewer is None:
            return
        if not hasattr(task, '_structure_yaw'):
            return original(task)
        yaw = float(task._structure_yaw[0, WAREHOUSE])
        c, s = math.cos(yaw), math.sin(yaw)
        center = task._structure_shift[0, WAREHOUSE].tolist()
        origin = task.gym.get_env_origin(task.envs[0])
        offset = task._env_origins[0].tolist()
        x, y = center[0]+offset[0]-origin.x-2*c, center[1]+offset[1]-origin.y-2*s
        z = offset[2]-origin.z+task._warehouse['floor_height']
        task.gym.viewer_camera_look_at(task.viewer, task.envs[0],
            gymapi.Vec3(x-.01*c, y-.01*s, z+18.), gymapi.Vec3(x, y, z))

    task_class._update_camera = update_camera


def install_sides(warehouse_before, warehouse_coop, box_side, leader_side):
    original = warehouse_coop._candidates

    def sample(task, ids, warehouse):
        task._warehouse_before_pairs[ids] = False
        task._warehouse_variant_samples[0] += warehouse.sum()

    def candidates(tasks, pre, roles, sizes, supports, near, counts, directions, modes, cfg, straight_paths=False):
        cfg = dict(cfg, viewer_sides=dict(box=box_side, leader=leader_side))
        return original(tasks, pre, roles, sizes, supports, near, counts, directions, modes, cfg, straight_paths)

    warehouse_before.sample = sample
    warehouse_coop._candidates = candidates


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False,
        description='Warehouse viewer options; left/right are viewed from outside facing into the room.')
    parser.add_argument('--view', choices=('default', 'top'), default='default')
    parser.add_argument('--box-side', choices=('left', 'right', 'random'))
    parser.add_argument('--leader-side', choices=('left', 'right', 'random'))
    parser.add_argument('--pushers', type=int, choices=(2, 3, 4))
    parser.add_argument('--before-pairs', type=int)
    parser.add_argument('--independent-at', type=int)
    args, command = parser.parse_known_args()
    if '--test' not in command or '--eval' not in command:
        parser.error('Viewer options require test/eval mode.')
    if '--eval_protocol' in command:
        parser.error('These scene overrides are for the viewer, not formal evaluation.')
    sys.path.insert(0, str(ROOT / 'tokenhsi'))
    import isaacgym
    if args.pushers is not None or args.before_pairs is not None or args.independent_at is not None:
        from qualitative.warehouse_expansion import install, command_counts
        pushers = args.pushers if args.pushers is not None else 2
        pairs = args.before_pairs if args.before_pairs is not None else 1
        independent_at = args.independent_at if args.independent_at is not None else 0
        if pairs < 1:
            parser.error('--before-pairs must be at least 1')
        if independent_at < 0:
            parser.error('--independent-at must be nonnegative')
        install(pushers, pairs, independent_at)
        command = command_counts(command, pushers, pairs, independent_at)
    if args.box_side is not None or args.leader_side is not None:
        from env.tasks.multi_agent import warehouse_before
        from utils import warehouse_coop
        install_sides(warehouse_before, warehouse_coop, args.box_side, args.leader_side)
    if args.view == 'top':
        from env.tasks.multi_agent.humanoid_ma_sensor_stage2 import HumanoidMASensorStage2
        install_top_view(HumanoidMASensorStage2)
    sys.argv = [str(ROOT / 'tokenhsi/run.py'), *command]
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()

import isaacgym
import torch
import run
import utils.parse_task as task_registry

from env.tasks.multi_agent.box_cleanup_demo import BoxCleanupDemo
from learning.multi_agent.box_cleanup_player import BoxCleanupPlayer


def build_demo_runner(observer):
    runner = build_runner(observer)
    runner.player_factory.register_builder('ma', lambda **kwargs: BoxCleanupPlayer(**kwargs))
    return runner


if __name__ == '__main__':
    torch.set_num_threads(1)
    task_registry.HumanoidMACarry = BoxCleanupDemo
    build_runner = run.build_alg_runner
    run.build_alg_runner = build_demo_runner
    run.main()

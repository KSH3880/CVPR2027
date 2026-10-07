import isaacgym
import torch
import run
import utils.parse_task as task_registry

from env.tasks.multi_agent.at_goal_demo import AtGoalDemo
from learning.multi_agent.at_goal_player import AtGoalPlayer


def build_demo_runner(observer):
    runner = build_runner(observer)
    runner.player_factory.register_builder('ma', lambda **kwargs: AtGoalPlayer(**kwargs))
    return runner


if __name__ == '__main__':
    torch.set_num_threads(1)
    task_registry.HumanoidMACarry = AtGoalDemo
    build_runner = run.build_alg_runner
    run.build_alg_runner = build_demo_runner
    run.main()

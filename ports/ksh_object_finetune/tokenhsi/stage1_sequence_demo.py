import isaacgym
import torch
import run
import utils.parse_task as task_registry

from env.tasks.multi_agent.stage1_sequence_demo import Stage1SequenceDemo
from learning.multi_agent.stage1_sequence_player import Stage1SequencePlayer


def build_sequence_runner(observer):
    runner = build_runner(observer)
    runner.player_factory.register_builder('ma', lambda **kwargs: Stage1SequencePlayer(**kwargs))
    return runner


if __name__ == '__main__':
    torch.set_num_threads(1)
    task_registry.HumanoidMACarry = Stage1SequenceDemo
    build_runner = run.build_alg_runner
    run.build_alg_runner = build_sequence_runner
    run.main()

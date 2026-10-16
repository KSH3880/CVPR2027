import isaacgym
import torch
import run
import yaml

from learning.multi_agent.stage1_grasp_shift_player import Stage1GraspShiftPlayer


def build_shift_runner(observer):
    runner = build_runner(observer)
    runner.player_factory.register_builder('ma', lambda **kwargs: Stage1GraspShiftPlayer(**kwargs))
    return runner


def load_shared_config(args):
    if not args.test or not args.eval:
        raise ValueError('Shared grasp shift is a Stage1 inference toy, not a training entry')
    cfg, train, logdir = load_config(args)
    with open('tokenhsi/data/cfg/train/rlg/amp_ma_carry_relation_unified_size_rsi_task_embedding_carry_distill.yaml') as handle:
        source = yaml.safe_load(handle)
    train['params']['network'] = source['params']['network']
    return cfg, train, logdir


if __name__ == '__main__':
    torch.set_num_threads(1)
    build_runner = run.build_alg_runner
    run.build_alg_runner = build_shift_runner
    load_config = run.load_cfg
    run.load_cfg = load_shared_config
    run.main()

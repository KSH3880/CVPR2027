"""Entry point for an explicitly configured low-LR CLIMB continuation."""
import isaacgym
import run as training
from learning.multi_agent.climb_continue_agent import ClimbContinueAgent
original=training.build_alg_runner

def build(observer):
    runner=original(observer)
    runner.algo_factory.register_builder('chair_ft',lambda **kwargs:ClimbContinueAgent(**kwargs))
    return runner

training.build_alg_runner=build
if __name__=='__main__':training.main()

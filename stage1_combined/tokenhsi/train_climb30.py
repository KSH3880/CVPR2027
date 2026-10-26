import isaacgym
import run as training
import utils.parse_task as parser
from env.tasks.multi_agent.humanoid_tokenhsi_climb_30d import HumanoidTokenHSIClimb30FT
from learning.multi_agent.climb_30d_agent import Climb30Agent
from learning.multi_agent.ma_players import MAPlayerContinuous
parser.HumanoidTokenHSIClimb30FT=HumanoidTokenHSIClimb30FT
original=training.build_alg_runner
def build(observer):
    runner=original(observer);runner.algo_factory.register_builder('climb30_ft',lambda **kw:Climb30Agent(**kw));runner.player_factory.register_builder('climb30_ft',lambda **kw:MAPlayerContinuous(**kw));return runner
training.build_alg_runner=build
if __name__=='__main__':training.main()

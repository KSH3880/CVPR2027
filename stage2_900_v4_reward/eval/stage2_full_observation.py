from learning import amp_players
from learning.multi_agent.ma_players import MAPlayerContinuous


def full_observation_action(self, obs, is_determenistic=False):
    self.model.a2c_network.object_valid = None
    return amp_players.AMPPlayerContinuous.get_action(self, obs, is_determenistic)


MAPlayerContinuous.get_action = full_observation_action

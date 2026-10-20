from learning.multi_agent.stage2_network_builder import Stage2Builder


class BeforeBuilder(Stage2Builder):
    class Network(Stage2Builder.Network):
        def _humans(self, encoder, obs):
            return encoder(obs)

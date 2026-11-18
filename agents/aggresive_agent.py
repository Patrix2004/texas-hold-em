from .base import Agent, aggress


class AggressiveAgent(Agent):
    name = "aggressive"

    def act(self, obs, mask) -> int:
        return aggress(mask)
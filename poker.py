from pettingzoo import make
import time
import torch
import torch.nn as nn
import numpy as np
import random


def choose_action(model, state, mask, epsilon = 0.1):

    legal_actions = np.flatnonzero(mask)
    
    
    if random.random() < epsilon:
        return random.choice(legal_actions)

    state = torch.tensor(state, dtype=torch.float32)

    output = model.forward(state)

    output[torch.tensor(mask) == 0] = -float("inf")

    return int(torch.argmax(output).item())

    

    

class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(72, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)
        

actions = {
    0: "Call",
    1: "Raise",
    2: "Fold",
    3: "Check"
}

env = make("aec", "classic/texas_holdem-v4", render_mode="human")
env.reset(seed=42)




model = DQN()



for episode in range(10):

    env.reset()

    for agent in env.agent_iter():

        obs, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            env.step(None)
            continue

        state = obs["observation"]
        mask = obs["action_mask"]

        if agent == "player_0":
            action = choose_action(model, state, mask, epsilon=0.1)
        else:
            action = env.action_space(agent).sample(mask)

        env.step(action)

env.close()

# state = torch.zeros(72)
# output = model.forward(state)

# print(output)


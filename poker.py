from pettingzoo import make
import time
import torch
import torch.nn as nn
import numpy as np
import random
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, next_mask, done):
        self.buffer.append((
            state.copy(),
            action,
            reward,
            next_state.copy(),
            next_mask.copy(),
            done
        ))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

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

replay = ReplayBuffer()

for episode in range(10):

    env.reset()
    pending = None

    for agent in env.agent_iter():

        obs, reward, terminated, truncated, info = env.last()
        done = terminated or truncated

        if agent == "player_0" and pending is not None:

            old_state, old_action = pending

            if done:
                next_state = np.zeros(72, dtype=np.float32)
                next_mask = np.zeros(4, dtype=np.int8)
            else:
                next_state = obs["observation"].copy()
                next_mask = obs["action_mask"].copy()

            replay.add(
                old_state,
                old_action,
                reward,
                next_state,
                next_mask,
                done
            )

            pending = None

        if done:
            env.step(None)
            continue

        state = obs["observation"]
        mask = obs["action_mask"]

        if agent == "player_0":

            action = choose_action(model, state, mask, epsilon=0.1)

            pending = (state.copy(), action)

        else:
            action = env.action_space(agent).sample(mask)

        env.step(action)

    print("Episode:", episode, "Replay size:", len(replay))

env.close()

# state = torch.zeros(72)
# output = model.forward(state)

# print(output)


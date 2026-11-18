from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int = 500_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, next_mask, done) -> None:
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            np.array(next_mask, dtype=np.int8),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, next_masks, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.long),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.array(next_states), dtype=torch.float32),
            torch.as_tensor(np.array(next_masks), dtype=torch.bool),
            torch.as_tensor(dones, dtype=torch.bool),
        )

    def __len__(self) -> int:
        return len(self.buffer)
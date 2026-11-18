
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..env import N_ACTIONS, OBS_DIM
from .base import Agent


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, epsilon: float = 0.0, name: str = "dqn"):
        self.net = QNetwork()
        self.epsilon = epsilon
        self.name = name

    def act(self, obs, mask, epsilon: float | None = None) -> int:
        eps = self.epsilon if epsilon is None else epsilon
        if eps > 0 and random.random() < eps:
            return int(random.choice(np.flatnonzero(mask).tolist()))

        state = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            q = self.net(state)
            q = q.masked_fill(~torch.as_tensor(mask, dtype=torch.bool), float("-inf"))
        return int(q.argmax().item())

    def snapshot(self, epsilon: float = 0.0, name: str | None = None) -> "DQNAgent":
        clone = copy.deepcopy(self)
        clone.net.eval()
        clone.epsilon = epsilon
        clone.name = name or f"{self.name}_snapshot"
        return clone

    def save(self, path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": 1,
            "obs_dim": OBS_DIM,
            "n_actions": N_ACTIONS,
            "model_state": self.net.state_dict(),
            "extra": extra or {},
        }, path)

    @classmethod
    def load(cls, path, name: str = "dqn") -> "DQNAgent":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

        agent = cls(name=name)

        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        agent.net.load_state_dict(state)
        agent.net.eval()

        return agent
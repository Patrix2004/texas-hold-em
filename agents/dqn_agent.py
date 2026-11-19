
"""DQN z pamiecia kart wlasnych w pojedynczym rozdaniu.

Siec otrzymuje 128 cech:
- 52 karty wlasne,
- 52 karty wspolne,
- 20 cech historii podbic,
- 4 bity maski legalnych akcji.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..cards import HandTracker, N_CARDS
from ..env import N_ACTIONS, OBS_DIM
from .base import Agent

MODEL_OBS_DIM = 2 * N_CARDS + (OBS_DIM - N_CARDS) + N_ACTIONS


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(MODEL_OBS_DIM, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, N_ACTIONS)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, epsilon: float = 0.0, name: str = "dqn"):
        self.net = QNetwork()
        self.epsilon = epsilon
        self.name = name
        self.tracker = HandTracker()

    def reset(self) -> None:
        self.tracker.reset()

    def encode(self, obs, mask) -> np.ndarray:
        raw = np.asarray(obs, dtype=np.float32).reshape(-1)
        legal = np.asarray(mask, dtype=np.float32).reshape(-1)

        if raw.size != OBS_DIM or legal.size != N_ACTIONS:
            raise ValueError(
                f"Oczekiwano {OBS_DIM} cech i {N_ACTIONS} akcji"
            )

        if not legal.any():
            raise ValueError("Brak legalnych akcji w niekoncowym stanie")

        self.tracker.update(raw)

        hole = np.zeros(N_CARDS, dtype=np.float32)
        board = np.zeros(N_CARDS, dtype=np.float32)

        if self.tracker.hole is not None:
            hole[list(self.tracker.hole)] = 1.0
            board[self.tracker.board] = 1.0

        # Zachowujemy tylko historie podbic.
        # Oryginalne 52 cechy kart sa juz reprezentowane
        # osobno w hole i board.
        betting_history = raw[N_CARDS:]

        return np.concatenate((
            hole,
            board,
            betting_history,
            legal
        )).astype(np.float32)

    def act_encoded(
        self,
        encoded_state,
        mask,
        epsilon: float | None = None
    ) -> int:
        eps = self.epsilon if epsilon is None else epsilon
        legal_actions = np.flatnonzero(mask).tolist()

        if not legal_actions:
            raise ValueError("Nie mozna wybrac akcji z pustej maski")

        if eps > 0 and random.random() < eps:
            return int(random.choice(legal_actions))

        state = torch.as_tensor(
            encoded_state,
            dtype=torch.float32
        )

        with torch.no_grad():
            q = self.net(state).masked_fill(
                ~torch.as_tensor(mask, dtype=torch.bool),
                float("-inf")
            )

        return int(q.argmax().item())

    def act(
        self,
        obs,
        mask,
        epsilon: float | None = None
    ) -> int:
        return self.act_encoded(
            self.encode(obs, mask),
            mask,
            epsilon
        )

    def snapshot(
        self,
        epsilon: float = 0.0,
        name: str | None = None
    ) -> "DQNAgent":
        clone = copy.deepcopy(self)
        clone.reset()
        clone.net.eval()
        clone.epsilon = epsilon
        clone.name = name or f"{self.name}_snapshot"
        return clone

    def save(self, path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "format": 2,
            "obs_dim": MODEL_OBS_DIM,
            "n_actions": N_ACTIONS,
            "model_state": self.net.state_dict(),
            "extra": extra or {},
        }, path)

    @classmethod
    def load(
        cls,
        path,
        name: str = "dqn"
    ) -> "DQNAgent":
        ckpt = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

        state = ckpt.get("model_state", ckpt)

        if state["net.0.weight"].shape[1] != MODEL_OBS_DIM:
            raise ValueError(
                f"Checkpoint ma niezgodna liczbe cech: "
                f"{state['net.0.weight'].shape[1]}D, "
                f"a model oczekuje {MODEL_OBS_DIM}D. "
                "Rozpocznij nowy trening albo uzyj "
                "zgodnego checkpointu."
            )

        agent = cls(name=name)
        agent.net.load_state_dict(state)
        agent.net.eval()

        return agent

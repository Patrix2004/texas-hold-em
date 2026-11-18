from __future__ import annotations

import random

import numpy as np

from .base import Agent


class RandomAgent(Agent):
    """Losuje równomiernie spośród legalnych akcji."""

    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, obs, mask) -> int:
        return int(self.rng.choice(np.flatnonzero(mask).tolist()))
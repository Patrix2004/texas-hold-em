from __future__ import annotations

import numpy as np

from .agents.base import Agent
from .agents.dqn_agent import DQNAgent, QNetwork
from .agents.heuristic_agent import HeuristicAgent
from .agents.monte_carlo_agent import MonteCarloAgent
from .agents.random_agent import RandomAgent
from .agents.aggresive_agent import AggressiveAgent

def moving_average(x, window: int) -> np.ndarray:
    """Średnia krocząca; wynik ma długość n - window + 1."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    window = max(1, min(window, len(x)))
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[window:] - c[:-window]) / window

BASELINES = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "monte_carlo": MonteCarloAgent,
    "aggressive": AggressiveAgent,
}


def make_baseline(name: str, **kwargs) -> Agent:
    if name not in BASELINES:
        raise KeyError(f"Nieznany baseline '{name}'. Dostępne: {list(BASELINES)}")
    return BASELINES[name](**kwargs)


__all__ = [
    "Agent", "DQNAgent", "QNetwork", "RandomAgent", "HeuristicAgent",
    "MonteCarloAgent", "BASELINES", "make_baseline",
]
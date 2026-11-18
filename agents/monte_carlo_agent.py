"""Agent Monte Carlo: szacuje equity ręki symulacjami rozdań przeciwko losowej ręce."""
from __future__ import annotations

import random

from ..cards import HandTracker, chips_info, hand_score
from .base import Agent, aggress, give_up, passive


def estimate_equity(hole, board, n_simulations: int, rng: random.Random) -> float:
    """Prawdopodobieństwo wygranej (remis = 0.5) vs losowa ręka przeciwnika."""
    known = set(hole) | set(board)
    deck = [c for c in range(52) if c not in known]
    missing_board = 5 - len(board)
    score = 0.0
    for _ in range(n_simulations):
        draw = rng.sample(deck, 2 + missing_board)
        opp = draw[:2]
        full_board = list(board) + draw[2:]
        mine = hand_score(list(hole) + full_board)
        theirs = hand_score(opp + full_board)
        if mine > theirs:
            score += 1.0
        elif mine == theirs:
            score += 0.5
    return score / n_simulations


class MonteCarloAgent(Agent):
    name = "monte_carlo"

    def __init__(self, n_simulations=300, raise_threshold=0.70, call_margin=0.05, seed=None):
        self.n_simulations = n_simulations
        self.raise_threshold = raise_threshold
        self.call_margin = call_margin
        self.rng = random.Random(seed)
        self.tracker = HandTracker()

    def reset(self) -> None:
        self.tracker.reset()

    def act(self, obs, mask) -> int:
        if not self.tracker.update(obs):
            return passive(mask)

        _, _, to_call, pot = chips_info(obs)
        equity = estimate_equity(self.tracker.hole, self.tracker.board,
                                 self.n_simulations, self.rng)
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0

        if equity >= self.raise_threshold:
            return aggress(mask)
        if to_call == 0:
            return passive(mask)
        if equity >= pot_odds + self.call_margin:
            return passive(mask)
        return give_up(mask)
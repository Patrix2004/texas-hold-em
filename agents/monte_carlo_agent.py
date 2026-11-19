
"""Monte Carlo z poprawnym uwzglednieniem pot odds."""
from __future__ import annotations

import random

from ..cards import HandTracker, hand_score
from ..env import CALL, CHECK, RAISE, BettingInfo
from .base import Agent, give_up, passive

def estimate_equity(
    hole, board, n_simulations: int, rng: random.Random
) -> float:
    if n_simulations <= 0:
        raise ValueError("n_simulations musi byc > 0")

    known = set(hole) | set(board)
    deck = [c for c in range(52) if c not in known]
    missing_board = 5 - len(board)

    if len(hole) != 2 or not (0 <= missing_board <= 5):
        raise ValueError("Nieprawidlowe karty")

    score = 0.0

    for _ in range(n_simulations):
        draw = rng.sample(deck, 2 + missing_board)
        full_board = list(board) + draw[2:]

        mine = hand_score(list(hole) + full_board)
        theirs = hand_score(draw[:2] + full_board)

        score += (
            1.0 if mine > theirs
            else 0.5 if mine == theirs
            else 0.0
        )

    return score / n_simulations

class MonteCarloAgent(Agent):
    name = "monte_carlo"

    def __init__(
        self,
        n_simulations=300,
        raise_threshold=0.70,
        call_margin=0.05,
        seed=None
    ):
        self.n_simulations = n_simulations
        self.raise_threshold = raise_threshold
        self.call_margin = call_margin
        self.rng = random.Random(seed)
        self.tracker = HandTracker()
        self._betting_info: BettingInfo | None = None

    def reset(self) -> None:
        self.tracker.reset()
        self._betting_info = None

    def set_betting_info(self, info: BettingInfo) -> None:
        self._betting_info = info

    def act(self, obs, mask) -> int:
        info = self._betting_info
        self._betting_info = None

        if info is None:
            raise RuntimeError(
                "Brak danych licytacji: "
                "wywolaj set_betting_info przed act"
            )

        if bool(mask[CHECK]) != (info.to_call == 0):
            raise RuntimeError(
                "Maska akcji i koszt call sa niespojne"
            )

        if not self.tracker.update(obs):
            return passive(mask)

        equity = estimate_equity(
            self.tracker.hole,
            self.tracker.board,
            self.n_simulations,
            self.rng
        )

        # Silna reka - podbijamy
        if equity >= self.raise_threshold and mask[RAISE]:
            return RAISE

        # Darmowe sprawdzenie
        if mask[CHECK]:
            return CHECK

        # Sprawdzenie, gdy equity uzasadnia koszt
        if (
            mask[CALL]
            and equity >= info.pot_odds + self.call_margin
        ):
            return CALL

        return give_up(mask)
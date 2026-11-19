
"""Agent heurystyczny uwzgledniajacy koszt call."""
from __future__ import annotations

import random

from ..cards import (
    HandTracker,
    card_suit,
    card_value,
    hand_score,
)
from ..env import CALL, CHECK, RAISE, BettingInfo
from .base import Agent, give_up, passive

CATEGORY_STRENGTH = {
    1: 0.50,
    2: 0.70,
    3: 0.80,
    4: 0.85,
    5: 0.90,
    6: 0.95,
    7: 1.00,
    8: 1.00,
}

def preflop_strength(hole) -> float:
    a, b = (card_value(c) for c in hole)

    high, low = max(a, b), min(a, b)
    strength = (high + low) / 28.0

    if high == low:
        strength += 0.25

    if card_suit(hole[0]) == card_suit(hole[1]):
        strength += 0.05

    if high - low <= 2:
        strength += 0.04

    return min(strength, 1.0)

def postflop_strength(hole, board) -> float:
    full = hand_score([*hole, *board])
    board_only = hand_score(board)

    if full[0] > board_only[0]:
        strength = CATEGORY_STRENGTH[full[0]]

        if (
            full[0] == 1
            and full[1] >= max(card_value(c) for c in board)
        ):
            strength = 0.60

        return strength

    if len(board) == 5 and full > board_only:
        return 0.45

    return (
        0.30 if max(card_value(c) for c in hole) >= 12
        else 0.20
    )

class HeuristicAgent(Agent):
    name = "heuristic"

    def __init__(
        self,
        raise_threshold=0.75,
        call_threshold=0.40,
        bluff_prob=0.03,
        pot_sensitivity=0.80,
        seed=None
    ):
        self.raise_threshold = raise_threshold
        self.call_threshold = call_threshold
        self.bluff_prob = bluff_prob
        self.pot_sensitivity = pot_sensitivity
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

        hole = self.tracker.hole
        board = self.tracker.board

        if not board:
            strength = preflop_strength(hole)
        else:
            strength = postflop_strength(hole, board)

        # Silna reka - podbicie
        if strength >= self.raise_threshold and mask[RAISE]:
            return RAISE

        # Darmowy check, czasami blef
        if mask[CHECK]:
            if (
                mask[RAISE]
                and self.rng.random() < self.bluff_prob
            ):
                return RAISE

            return CHECK

        # Prog call zalezy od pot odds.
        # Strength nie jest prawdopodobienstwem wygranej!
        adaptive_threshold = (
            self.call_threshold
            + self.pot_sensitivity * (info.pot_odds - 0.25)
        )

        adaptive_threshold = max(
            0.0,
            min(1.0, adaptive_threshold)
        )

        if mask[CALL] and strength >= adaptive_threshold:
            return CALL

        return give_up(mask)
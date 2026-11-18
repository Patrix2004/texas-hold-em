"""Agent heurystyczny oparty wyłącznie na obserwacji (karty + żetony).

Preflop: prosta ocena ręki (wysokie karty, para, suited, connectory).
Postflop: czy ręka (hole + stół) poprawiła układ względem samego stołu.
Żetony: obs[52] (moje) i obs[53] (max) -> kwota do dopłacenia i szacowane pot odds.
"""
from __future__ import annotations

import random

from ..cards import HandTracker, card_suit, card_value, chips_info, hand_score
from .base import Agent, aggress, give_up, passive

# siła układu poprawionego kartą z ręki (kategoria -> siła 0..1)
CATEGORY_STRENGTH = {1: 0.50, 2: 0.70, 3: 0.80, 4: 0.85, 5: 0.90, 6: 0.95, 7: 1.0, 8: 1.0}


def preflop_strength(hole) -> float:
    a, b = (card_value(c) for c in hole)
    high, low = max(a, b), min(a, b)
    s = (high + low) / 28.0
    if high == low:
        s += 0.25
    if card_suit(hole[0]) == card_suit(hole[1]):
        s += 0.05
    if high - low <= 2:
        s += 0.04
    return min(s, 1.0)


def postflop_strength(hole, board) -> float:
    full = hand_score([*hole, *board])
    board_only = hand_score(board)
    if full[0] > board_only[0]:
        s = CATEGORY_STRENGTH[full[0]]
        # para: top pair / overpair jest wyraźnie lepsza od niższej pary
        if full[0] == 1 and full[1] >= max(card_value(c) for c in board):
            s = 0.60
        return s
    # nic nie trafione -> słaba ręka, minimalnie lepsza z wysoką kartą
    return 0.30 if max(card_value(c) for c in hole) >= 12 else 0.20


class HeuristicAgent(Agent):
    name = "heuristic"

    def __init__(self, raise_threshold=0.75, call_threshold=0.40,
                 cheap_call_pot_odds=0.20, bluff_prob=0.03, seed=None):
        self.raise_threshold = raise_threshold
        self.call_threshold = call_threshold
        self.cheap_call_pot_odds = cheap_call_pot_odds
        self.bluff_prob = bluff_prob
        self.rng = random.Random(seed)
        self.tracker = HandTracker()

    def reset(self) -> None:
        self.tracker.reset()

    def act(self, obs, mask) -> int:
        ready = self.tracker.update(obs)
        if not ready:  # nie znamy ręki (agent wszedł w środku rozdania)
            return passive(mask)

        _, _, to_call, pot = chips_info(obs)
        hole, board = self.tracker.hole, self.tracker.board
        strength = preflop_strength(hole) if not board else postflop_strength(hole, board)
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0

        if strength >= self.raise_threshold:
            return aggress(mask)
        if to_call == 0:
            if self.rng.random() < self.bluff_prob:
                return aggress(mask)
            return passive(mask)
        if strength >= self.call_threshold or pot_odds <= self.cheap_call_pot_odds:
            return passive(mask)  # call (check nie jest legalne przy to_call > 0)
        return give_up(mask)
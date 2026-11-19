"""Karty RLCard / PettingZoo Limit Texas Hold'em i ocena ukladow.

Surowa obserwacja 72D:
- 0..51: suma kart gracza i kart wspolnych (nie rozroznia ich),
- 52..56, 57..61, 62..66, 67..71: one-hot liczby podbic w rundach.
Nie ma w niej wysokosci puli ani kwoty do sprawdzenia.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

N_CARDS = 52
RANKS = "A23456789TJQK"
SUITS = "shdc"


def card_value(c: int) -> int:
    r = c % 13
    return 14 if r == 0 else r + 1


def card_suit(c: int) -> int:
    return c // 13


def card_str(c: int) -> str:
    return RANKS[c % 13] + SUITS[c // 13]


def observed_cards(obs) -> list[int]:
    return [int(c) for c in np.flatnonzero(np.asarray(obs)[:N_CARDS])]


class HandTracker:
    """Odtwarza karty wlasne ze stanu preflop i oddziela karty wspolne.

    Wywolaj reset() na poczatku KAZDEGO rozdania. Agent musi otrzymac
    obserwacje preflop, aby pozniej moc rozpoznac swoje karty.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.hole: tuple[int, ...] | None = None
        self.board: list[int] = []

    def update(self, obs) -> bool:
        cards = observed_cards(obs)
        if len(cards) <= 2:
            self.hole = tuple(cards) if len(cards) == 2 else None
            self.board = []
        elif self.hole is not None:
            self.board = [c for c in cards if c not in self.hole]
        return self.hole is not None and (len(cards) <= 2 or len(self.board) >= 3)


def _straight_high(values) -> int | None:
    s = set(values)
    if 14 in s:
        s.add(1)
    for high in range(14, 4, -1):
        if all(v in s for v in range(high - 4, high + 1)):
            return high
    return None


def hand_score(cards) -> tuple:
    """Porownywalna krotka. 0 high card, 1 para, ..., 8 poker.

    Dla porownania koncowych ukladow stosuj 7 kart.
    """
    values = [card_value(c) for c in cards]
    suits = [card_suit(c) for c in cards]
    counts = Counter(values)

    flush_suit = next((s for s, n in Counter(suits).items() if n >= 5), None)
    flush_vals = [v for v, s in zip(values, suits) if s == flush_suit] if flush_suit is not None else []

    if flush_suit is not None:
        sf = _straight_high(flush_vals)
        if sf:
            return (8, sf)

    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    quads = [v for v, n in groups if n == 4]
    trips = [v for v, n in groups if n == 3]
    pairs = [v for v, n in groups if n == 2]

    if quads:
        kicker = max((v for v in counts if v != quads[0]), default=0)
        return (7, quads[0], kicker)
    if trips and (len(trips) > 1 or pairs):
        return (6, trips[0], max(trips[1:] + pairs))
    if flush_suit is not None:
        return (5, *sorted(flush_vals, reverse=True)[:5])
    st = _straight_high(values)
    if st:
        return (4, st)
    if trips:
        kickers = sorted((v for v in counts if v != trips[0]), reverse=True)[:2]
        return (3, trips[0], *kickers)
    if len(pairs) >= 2:
        top = pairs[:2]
        kicker = max((v for v in counts if v not in top), default=0)
        return (2, *top, kicker)
    if len(pairs) == 1:
        kickers = sorted((v for v in counts if v != pairs[0]), reverse=True)[:3]
        return (1, pairs[0], *kickers)
    return (0, *sorted(values, reverse=True)[:5])
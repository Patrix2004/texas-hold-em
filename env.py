
"""Stale srodowiska i publiczne informacje o licytacji."""
from __future__ import annotations

from dataclasses import dataclass
from pettingzoo import make

CALL, RAISE, FOLD, CHECK = 0, 1, 2, 3

ACTION_IDS = (CALL, RAISE, FOLD, CHECK)

ACTION_NAMES = {
    CALL: "call",
    RAISE: "raise",
    FOLD: "fold",
    CHECK: "check",
}

OBS_DIM = 72
N_ACTIONS = 4

SEATS = ("player_0", "player_1")

def make_env():
    return make("aec", "classic/texas_holdem-v4")

@dataclass(frozen=True)
class BettingInfo:
    pot: float
    to_call: float
    pot_odds: float

def get_betting_info(env, seat: str) -> BettingInfo:
    if seat not in SEATS:
        raise ValueError(f"Nieznane miejsce: {seat}")

    base = env.unwrapped
    rlcard_env = base.env
    seat_index = SEATS.index(seat)

    raw = rlcard_env.get_state(seat_index)["raw_obs"]

    all_chips = [
        float(x) for x in raw["all_chips"]
    ]
    my_chips = float(raw["my_chips"])

    if len(all_chips) != len(SEATS):
        raise ValueError(
            f"Oczekiwano dwoch graczy: {all_chips}"
        )

    if any(x < 0 for x in all_chips):
        raise ValueError("Ujemne wklady graczy")

    if abs(all_chips[seat_index] - my_chips) > 1e-6:
        raise ValueError(
            "my_chips nie jest zgodne z all_chips"
        )

    pot = sum(all_chips)
    to_call = max(
        0.0,
        max(all_chips) - my_chips
    )

    pot_odds = (
        to_call / (pot + to_call)
        if to_call > 0
        else 0.0
    )

    return BettingInfo(
        pot=pot,
        to_call=to_call,
        pot_odds=pot_odds
    )
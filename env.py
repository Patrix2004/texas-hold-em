"""Stałe środowiska i fabryka env w jednym miejscu."""
from __future__ import annotations

from pettingzoo import make

CALL, RAISE, FOLD, CHECK = 0, 1, 2, 3
ACTION_IDS = (CALL, RAISE, FOLD, CHECK)
ACTION_NAMES = {CALL: "call", RAISE: "raise", FOLD: "fold", CHECK: "check"}

OBS_DIM = 72
N_ACTIONS = 4
SEATS = ("player_0", "player_1")


def make_env():
    return make("aec", "classic/texas_holdem-v4")
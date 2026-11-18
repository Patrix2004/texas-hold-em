from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..env import CALL, CHECK, FOLD, RAISE


class Agent(ABC):
    """Wspólny interfejs: act(obs_vector, action_mask) -> id akcji."""

    name = "agent"

    def reset(self) -> None:
        """Wywoływane na początku każdego rozdania."""

    @abstractmethod
    def act(self, obs, mask) -> int:
        ...


def first_legal(mask, prefs) -> int:
    for a in prefs:
        if mask[a]:
            return int(a)
    return int(np.flatnonzero(mask)[0])


def passive(mask) -> int:
    return first_legal(mask, (CHECK, CALL, FOLD))


def give_up(mask) -> int:
    """Check jeśli za darmo, w przeciwnym razie fold."""
    return first_legal(mask, (CHECK, FOLD, CALL))


def aggress(mask) -> int:
    return first_legal(mask, (RAISE, CALL, CHECK))

"""Ewaluacja agentow Texas Hold'em.

Obsluguje:
- rozgrywki heads-up,
- zamiane miejsc graczy,
- powtarzalne seedy rozdan,
- srednia nagrode,
- win rate,
- rozklad akcji,
- przekazywanie poprawnych pot odds.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .agents.base import Agent
from .env import (
    ACTION_IDS,
    ACTION_NAMES,
    SEATS,
    get_betting_info,
    make_env,
)


EVAL_SEED = 100_000


@dataclass
class MatchResult:
    name_a: str
    name_b: str
    rewards: np.ndarray
    action_counts: Counter

    @property
    def hands(self) -> int:
        return len(self.rewards)

    @property
    def mean(self) -> float:
        return float(self.rewards.mean())

    @property
    def sem(self) -> float:
        if self.hands < 2:
            return 0.0

        return float(
            self.rewards.std(ddof=1)
            / np.sqrt(self.hands)
        )

    @property
    def ci95(self) -> float:
        return 1.96 * self.sem

    @property
    def win_rate(self) -> float:
        return float(
            (self.rewards > 0).mean()
        )

    def action_rates(self) -> dict[str, float]:
        total = sum(self.action_counts.values()) or 1

        return {
            ACTION_NAMES[a]: (
                self.action_counts.get(a, 0) / total
            )
            for a in ACTION_IDS
        }

    def summary(self) -> dict:
        return {
            "opponent": self.name_b,
            "hands": self.hands,
            "mean": self.mean,
            "sem": self.sem,
            "win_rate": self.win_rate,
            "action_rates": self.action_rates(),
        }


def play_hand(
    env,
    seats: dict[str, Agent],
    seed: int | None = None
):
    """Rozgrywa jedno rozdanie i zwraca nagrody oraz akcje."""

    env.reset(seed=seed)

    # Reset pamieci obu agentow na poczatku rozdania.
    for agent in set(seats.values()):
        agent.reset()

    rewards = {
        seat: 0.0
        for seat in SEATS
    }

    counts = {
        seat: Counter()
        for seat in SEATS
    }

    for seat in env.agent_iter():
        obs, reward, terminated, truncated, _ = env.last()

        rewards[seat] += float(reward)

        if terminated or truncated:
            env.step(None)
            continue

        acting_agent = seats[seat]

        # NOWE: Monte Carlo i Heuristic otrzymuja
        # rzeczywista pule i koszt call z RLCard.
        if hasattr(acting_agent, "set_betting_info"):
            betting_info = get_betting_info(
                env, seat
            )

            acting_agent.set_betting_info(
                betting_info
            )

        action = acting_agent.act(
            obs["observation"],
            obs["action_mask"]
        )

        counts[seat][action] += 1

        env.step(action)

    return rewards, counts


def play_match(
    agent_a: Agent,
    agent_b: Agent,
    hands_per_seat: int = 200,
    seed: int = EVAL_SEED,
    env=None,
    name_a="A",
    name_b="B"
) -> MatchResult:
    """Rozgrywa mecz, zamieniajac miejsca agentow."""

    own_env = env is None

    if own_env:
        env = make_env()

    rewards = []
    counts = Counter()

    try:
        for i in range(hands_per_seat):
            # Ten sam seed dla obu ustawien miejsc.
            for a_seat, b_seat in (
                (SEATS[0], SEATS[1]),
                (SEATS[1], SEATS[0])
            ):
                r, c = play_hand(
                    env,
                    {
                        a_seat: agent_a,
                        b_seat: agent_b,
                    },
                    seed=seed + i
                )

                rewards.append(r[a_seat])
                counts.update(c[a_seat])

    finally:
        if own_env:
            env.close()

    return MatchResult(
        name_a=name_a,
        name_b=name_b,
        rewards=np.asarray(rewards, dtype=float),
        action_counts=counts
    )


def evaluate_against(
    agent: Agent,
    opponents: dict[str, Agent],
    hands_per_seat: int = 200,
    overrides: dict[str, int] | None = None,
    seed: int = EVAL_SEED,
    agent_name: str = "agent"
) -> dict[str, MatchResult]:
    """Testuje agenta przeciwko wybranym baseline'om."""

    overrides = overrides or {}

    env = make_env()
    results = {}

    try:
        for name, opponent in opponents.items():
            n = overrides.get(
                name,
                hands_per_seat
            )

            results[name] = play_match(
                agent_a=agent,
                agent_b=opponent,
                hands_per_seat=n,
                seed=seed,
                env=env,
                name_a=agent_name,
                name_b=name
            )

    finally:
        env.close()

    return results


def round_robin(
    agents: dict[str, Agent],
    hands_per_seat: int = 200,
    overrides: dict[str, int] | None = None,
    seed: int = EVAL_SEED
):
    """Kazdy z kazdym; macierz srednich nagrod."""

    overrides = overrides or {}

    names = list(agents)
    n = len(names)

    matrix = np.full(
        (n, n),
        np.nan
    )

    results = {}
    env = make_env()

    try:
        for i in range(n):
            for j in range(i + 1, n):
                a = names[i]
                b = names[j]

                hands = min(
                    overrides.get(a, hands_per_seat),
                    overrides.get(b, hands_per_seat)
                )

                res = play_match(
                    agents[a],
                    agents[b],
                    hands_per_seat=hands,
                    seed=seed,
                    env=env,
                    name_a=a,
                    name_b=b
                )

                results[(a, b)] = res

                matrix[i, j] = res.mean
                matrix[j, i] = -res.mean

    finally:
        env.close()

    return names, matrix, results

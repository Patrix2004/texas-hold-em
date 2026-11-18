"""Pojedynki heads-up między dowolnymi agentami (z zamianą miejsc i tymi samymi rozdaniami)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .agents.base import Agent
from .env import ACTION_IDS, ACTION_NAMES, SEATS, make_env

EVAL_SEED = 100_000


@dataclass
class MatchResult:
    name_a: str
    name_b: str
    rewards: np.ndarray        # nagroda agenta A w każdym rozdaniu
    action_counts: Counter     # akcje agenta A

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
        return float(self.rewards.std(ddof=1) / np.sqrt(self.hands))

    @property
    def ci95(self) -> float:
        return 1.96 * self.sem

    @property
    def win_rate(self) -> float:
        return float((self.rewards > 0).mean())

    def action_rates(self) -> dict[str, float]:
        total = sum(self.action_counts.values()) or 1
        return {ACTION_NAMES[a]: self.action_counts.get(a, 0) / total for a in ACTION_IDS}

    def summary(self) -> dict:
        return {
            "opponent": self.name_b,
            "hands": self.hands,
            "mean": self.mean,
            "sem": self.sem,
            "win_rate": self.win_rate,
            "action_rates": self.action_rates(),
        }


def play_hand(env, seats: dict[str, Agent], seed: int | None = None):
    """Jedno rozdanie. Zwraca (nagrody per miejsce, liczniki akcji per miejsce)."""
    env.reset(seed=seed)
    for agent in set(seats.values()):
        agent.reset()

    rewards = {s: 0.0 for s in SEATS}
    counts = {s: Counter() for s in SEATS}

    for seat in env.agent_iter():
        obs, reward, terminated, truncated, _ = env.last()
        rewards[seat] += reward
        if terminated or truncated:
            env.step(None)
            continue
        action = seats[seat].act(obs["observation"], obs["action_mask"])
        counts[seat][action] += 1
        env.step(action)
    return rewards, counts


def play_match(agent_a: Agent, agent_b: Agent, hands_per_seat: int = 200,
               seed: int = EVAL_SEED, env=None, name_a="A", name_b="B") -> MatchResult:
    own_env = env is None
    env = env or make_env()

    rewards, counts = [], Counter()
    for i in range(hands_per_seat):
        # to samo rozdanie (seed) grane z obu stron stołu -> mniejsza wariancja
        for a_seat, b_seat in ((SEATS[0], SEATS[1]), (SEATS[1], SEATS[0])):
            r, c = play_hand(env, {a_seat: agent_a, b_seat: agent_b}, seed + i)
            rewards.append(r[a_seat])
            counts.update(c[a_seat])

    if own_env:
        env.close()
    return MatchResult(name_a, name_b, np.asarray(rewards, dtype=float), counts)


def evaluate_against(agent: Agent, opponents: dict[str, Agent], hands_per_seat: int = 200,
                     overrides: dict[str, int] | None = None, seed: int = EVAL_SEED,
                     agent_name: str = "agent") -> dict[str, MatchResult]:
    overrides = overrides or {}
    env = make_env()
    results = {}
    for name, opp in opponents.items():
        n = overrides.get(name, hands_per_seat)
        results[name] = play_match(agent, opp, n, seed, env, agent_name, name)
    env.close()
    return results


# def round_robin(agents: dict[str, Agent], hands_per_seat: int = 200,
#                 overrides: dict[str, int] | None = None, seed: int = EVAL_SEED):
#     """Każdy z każdym. Zwraca (nazwy, macierz średnich nagród [wiersz vs kolumna], wyniki).

#     Gra jest sumą zerową, więc wynik (j, i) to po prostu -(i, j).
#     """
#     overrides = overrides or {}
#     names = list(agents)
#     n = len(names)
#     matrix = np.full((n, n), np.nan)
#     results: dict[tuple[str, str], MatchResult] = {}
#     env = make_env()
#     for i in range(n):
#         for j in range(i + 1, n):
#             a, b = names[i], names[j]
#             hands = min(overrides.get(a, hands_per_seat), overrides.get(b, hands_per_seat))
#             res = play_match(agents[a], agents[b], hands, seed, env, a, b)
#             results[(a, b)] = res
#             matrix[i, j] = res.mean
#             matrix[j, i] = -res.mean
#     env.close()
#     return names, matrix, results
"""Wykresy treningu i ewaluacji. Nie nadpisuje otwartych obrazow."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .env import ACTION_NAMES  # noqa: E402
from .utils import moving_average  # noqa: E402


def _save(fig, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=130)
    finally:
        plt.close(fig)


def plot_training_curves(history: dict, path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    ax = axes.ravel()
    evals = history.get("eval", [])

    # W v2 zapisujemy usrednione punkty zamiast milionow osobnych nagrod.
    reward_points = history.get("reward_points", [])
    if reward_points:
        x, y = zip(*reward_points)
        ax[0].plot(x, y)
    elif history.get("episode_rewards"):
        # Zgodnosc z history.json pochodzacym ze starej wersji.
        rewards = history["episode_rewards"]
        window = max(1, min(5000, len(rewards) // 10))
        ma = moving_average(rewards, window)
        stride = max(1, len(ma) // 3000)
        ax[0].plot(np.arange(window, len(rewards) + 1)[::stride], ma[::stride])
    ax[0].set(title="Nagroda podczas treningu (srednia kroczaca)",
              xlabel="epizod", ylabel="nagroda")
    ax[0].axhline(0, color="gray", lw=0.8)

    loss = history.get("loss", [])
    if loss:
        steps, vals = zip(*loss)
        ax[1].plot(steps, vals, alpha=0.3, label="loss")
        window = max(1, min(50, len(vals) // 10))
        ax[1].plot(steps[window - 1:], moving_average(vals, window),
                   label=f"srednia ({window})")
        ax[1].set_yscale("log")
        ax[1].legend()
    ax[1].set(title="Loss (Huber)", xlabel="krok uczenia")

    eps = history.get("epsilon", [])
    if eps:
        e, v = zip(*eps)
        ax[2].plot(e, v)
    ax[2].set(title="Epsilon", xlabel="epizod")

    if evals:
        names = list(evals[0]["baselines"])
        xs = [ev["episode"] for ev in evals]
        for name in names:
            means = np.array([ev["baselines"][name]["mean"] for ev in evals])
            sems = np.array([ev["baselines"][name]["sem"] for ev in evals])
            ax[3].plot(xs, means, marker="o", label=name)
            ax[3].fill_between(xs, means - 1.96 * sems, means + 1.96 * sems,
                               alpha=0.15)
            ax[4].plot(
                xs, [ev["baselines"][name]["win_rate"] for ev in evals],
                marker="o", label=name
            )
        ax[3].legend()
        ax[4].legend()
        for action in ACTION_NAMES.values():
            rates = [
                np.mean([ev["baselines"][name]["action_rates"][action]
                         for name in names])
                for ev in evals
            ]
            ax[5].plot(xs, rates, marker="o", label=action)
        ax[5].legend()
    ax[3].axhline(0, color="gray", lw=0.8)
    ax[3].set(title="Srednia nagroda vs baseline (95% CI)",
              xlabel="epizod", ylabel="nagroda / rozdanie")
    ax[4].axhline(0.5, color="gray", lw=0.8)
    ax[4].set(title="Win rate vs baseline", xlabel="epizod", ylim=(0, 1))
    ax[5].set(title="Czestosc akcji agenta", xlabel="epizod", ylim=(0, 1))
    _save(fig, path)


def plot_vs_opponents(results: dict, path, title="Agent vs przeciwnicy") -> None:
    names = list(results)
    means = [results[name].mean for name in names]
    cis = [results[name].ci95 for name in names]
    wins = [results[name].win_rate for name in names]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["tab:green" if mean > 0 else "tab:red" for mean in means]
    a1.bar(names, means, yerr=cis, color=colors, capsize=4)
    a1.axhline(0, color="gray", lw=0.8)
    a1.set(title=f"{title}: srednia nagroda (95% CI)", ylabel="nagroda / rozdanie")
    a2.bar(names, wins, color="tab:blue")
    a2.axhline(0.5, color="gray", lw=0.8)
    a2.set(title="Win rate", ylim=(0, 1))
    for a in (a1, a2):
        a.tick_params(axis="x", rotation=20)
    _save(fig, path)


def plot_action_distribution(results: dict, path) -> None:
    names = list(results)
    actions = list(ACTION_NAMES.values())
    fig, ax = plt.subplots(figsize=(9, 1 + 0.7 * len(names)))
    left = np.zeros(len(names))
    for action in actions:
        vals = np.array([results[name].action_rates()[action] for name in names])
        ax.barh(names, vals, left=left, label=action)
        left += vals
    ax.set(title="Rozklad akcji agenta wobec przeciwnika", xlim=(0, 1))
    ax.legend(ncol=len(actions), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, path)


def plot_cumulative_reward(results: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, result in results.items():
        ax.plot(np.cumsum(result.rewards), label=name)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(title="Skumulowana nagroda agenta", xlabel="rozdanie", ylabel="suma nagrod")
    ax.legend()
    _save(fig, path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rysuje krzywe z history.json")
    parser.add_argument("history")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = args.out or str(Path(args.history).with_name("training_curves.png"))
    plot_training_curves(json.loads(Path(args.history).read_text()), out)
    print("Zapisano:", out)
"""Wizualizacje: krzywe treningowe i wyniki ewaluacji."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .env import ACTION_NAMES  # noqa: E402
from .utils import moving_average  # noqa: E402


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ trening
def plot_training_curves(history: dict, path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    ax = axes.ravel()
    evals = history.get("eval", [])

    # 1) nagroda w self-play
    rewards = history.get("episode_rewards", [])
    if rewards:
        window = max(1, min(5000, len(rewards) // 10))
        ma = moving_average(rewards, window)
        ax[0].plot(np.arange(window, len(rewards) + 1), ma)
    ax[0].set(title="Nagroda w self-play (średnia krocząca)", xlabel="epizod", ylabel="nagroda")
    ax[0].axhline(0, color="gray", lw=0.8)

    # 2) loss
    loss = history.get("loss", [])
    if loss:
        steps, vals = zip(*loss)
        ax[1].plot(steps, vals, alpha=0.3, label="loss")
        w = max(1, min(50, len(vals) // 10))
        ax[1].plot(steps[w - 1:], moving_average(vals, w), label=f"średnia ({w})")
        ax[1].set_yscale("log")
        ax[1].legend()
    ax[1].set(title="Loss (Huber)", xlabel="krok uczenia")

    # 3) epsilon
    eps = history.get("epsilon", [])
    if eps:
        e, v = zip(*eps)
        ax[2].plot(e, v)
    ax[2].set(title="Epsilon", xlabel="epizod")

    # 4-5) wyniki vs baseline'y
    if evals:
        names = list(evals[0]["baselines"])
        eps_x = [ev["episode"] for ev in evals]
        for name in names:
            m = np.array([ev["baselines"][name]["mean"] for ev in evals])
            s = np.array([ev["baselines"][name]["sem"] for ev in evals])
            ax[3].plot(eps_x, m, marker="o", label=name)
            ax[3].fill_between(eps_x, m - 1.96 * s, m + 1.96 * s, alpha=0.15)
            ax[4].plot(eps_x, [ev["baselines"][name]["win_rate"] for ev in evals],
                       marker="o", label=name)
        ax[3].legend()
        ax[4].legend()

        # 6) rozkład akcji agenta (średnia po baseline'ach)
        for action in ACTION_NAMES.values():
            rate = [np.mean([ev["baselines"][n]["action_rates"][action] for n in names])
                    for ev in evals]
            ax[5].plot(eps_x, rate, marker="o", label=action)
        ax[5].legend()
    ax[3].axhline(0, color="gray", lw=0.8)
    ax[3].set(title="Średnia nagroda vs baseline (95% CI)", xlabel="epizod", ylabel="nagroda / rozdanie")
    ax[4].axhline(0.5, color="gray", lw=0.8)
    ax[4].set(title="Win rate vs baseline", xlabel="epizod", ylim=(0, 1))
    ax[5].set(title="Częstość akcji agenta", xlabel="epizod", ylim=(0, 1))

    _save(fig, path)


# --------------------------------------------------------------- ewaluacja
def plot_vs_opponents(results: dict, path, title="Agent vs przeciwnicy") -> None:
    names = list(results)
    means = [results[n].mean for n in names]
    cis = [results[n].ci95 for n in names]
    wins = [results[n].win_rate for n in names]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["tab:green" if m > 0 else "tab:red" for m in means]
    a1.bar(names, means, yerr=cis, color=colors, capsize=4)
    a1.axhline(0, color="gray", lw=0.8)
    a1.set(title=f"{title}: średnia nagroda (95% CI)", ylabel="nagroda / rozdanie")
    a2.bar(names, wins, color="tab:blue")
    a2.axhline(0.5, color="gray", lw=0.8)
    a2.set(title="Win rate", ylim=(0, 1))
    for a in (a1, a2):
        a.tick_params(axis="x", rotation=20)
    _save(fig, path)


# def plot_round_robin(names, matrix, path) -> None:
#     fig, ax = plt.subplots(figsize=(1.4 * len(names) + 3, 1.2 * len(names) + 2))
#     lim = np.nanmax(np.abs(matrix)) or 1.0
#     im = ax.imshow(matrix, cmap="RdYlGn", vmin=-lim, vmax=lim)
#     ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
#     ax.set_yticks(range(len(names)), names)
#     for i in range(len(names)):
#         for j in range(len(names)):
#             if i != j:
#                 ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
#     ax.set(title="Round-robin: średnia nagroda wiersza vs kolumny")
#     fig.colorbar(im, ax=ax)
#     _save(fig, path)


def plot_action_distribution(results: dict, path) -> None:
    names = list(results)
    actions = list(ACTION_NAMES.values())
    fig, ax = plt.subplots(figsize=(9, 1 + 0.7 * len(names)))
    left = np.zeros(len(names))
    for action in actions:
        vals = np.array([results[n].action_rates()[action] for n in names])
        ax.barh(names, vals, left=left, label=action)
        left += vals
    ax.set(title="Rozkład akcji agenta wobec przeciwnika", xlim=(0, 1))
    ax.legend(ncol=len(actions), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, path)


def plot_cumulative_reward(results: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, res in results.items():
        ax.plot(np.cumsum(res.rewards), label=name)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(title="Skumulowana nagroda agenta", xlabel="rozdanie", ylabel="suma nagród")
    ax.legend()
    _save(fig, path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Rysuje krzywe treningowe z history.json")
    p.add_argument("history")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or str(Path(args.history).with_name("training_curves.png"))
    plot_training_curves(json.loads(Path(args.history).read_text()), out)
    print("Zapisano:", out)
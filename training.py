
"""Trening Double DQN z self-play i przeciwnikami bazowymi.

Uruchomienie z katalogu nadrzednego:
    python -m texas-hold-em-main.training

Wznowienie:
    python -m texas-hold-em-main.training --resume

Wznowienie przywraca model, optimizer, liczniki i pule przeciwnikow,
ale nie odtwarza replay buffera.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .agents.dqn_agent import DQNAgent, MODEL_OBS_DIM
from .config import TrainConfig
from .env import N_ACTIONS, SEATS, get_betting_info, make_env
from .evaluation import EVAL_SEED, evaluate_against
from .plotting import plot_training_curves
from .replay import ReplayBuffer
from .utils import make_baseline


def seeded_baseline(name: str, seed: int):
    # AggressiveAgent nie przyjmuje seed w konstruktorze.
    return (
        make_baseline(name)
        if name == "aggressive"
        else make_baseline(name, seed=seed)
    )


def dqn_update(
    net,
    target_net,
    optimizer,
    batch,
    gamma: float,
    grad_clip: float
) -> float:
    """Double DQN z maskowaniem nielegalnych akcji."""
    states, actions, rewards, next_states, next_masks, dones = batch

    q_now = net(states).gather(
        1, actions.unsqueeze(1)
    ).squeeze(1)

    with torch.no_grad():
        next_q = torch.zeros_like(rewards)
        active = ~dones

        if active.any():
            valid_masks = next_masks[active]

            if not valid_masks.any(dim=1).all():
                raise ValueError(
                    "Nieterminalne przejscie z pusta maska akcji"
                )

            online_q = net(next_states[active]).masked_fill(
                ~valid_masks, float("-inf")
            )

            best_actions = online_q.argmax(
                dim=1, keepdim=True
            )

            next_q[active] = target_net(
                next_states[active]
            ).gather(1, best_actions).squeeze(1)

        targets = rewards + gamma * next_q

    loss = F.smooth_l1_loss(q_now, targets)

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        net.parameters(), grad_clip
    )

    optimizer.step()

    return float(loss.item())


class Trainer:
    def __init__(
        self,
        agent: DQNAgent,
        cfg: TrainConfig,
        out_dir,
        baselines: dict | None = None
    ):
        self.agent = agent
        self.cfg = cfg

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.target_net = copy.deepcopy(agent.net).eval()

        self.optimizer = torch.optim.Adam(
            agent.net.parameters(),
            lr=cfg.lr
        )

        self.replay = ReplayBuffer(cfg.replay_capacity)

        # Przeciwnicy do ewaluacji
        self.baselines = (
            baselines
            if baselines is not None
            else {
                name: seeded_baseline(
                    name, cfg.seed + 10_000 + i
                )
                for i, name in enumerate(cfg.eval_baselines)
            }
        )

        # Pula starszych wersji DQN
        self.pool = [
            agent.snapshot(cfg.opponent_epsilon)
        ]

        probabilities = {
            "random": cfg.random_opponent_prob,
            "aggressive": cfg.aggressive_opponent_prob,
            "monte_carlo": cfg.monte_carlo_opponent_prob,
            "heuristic": cfg.heuristic_opponent_prob,
        }

        total = (
            sum(probabilities.values())
            + cfg.selfplay_opponent_prob
        )

        if (
            any(x < 0 for x in probabilities.values())
            or cfg.selfplay_opponent_prob < 0
        ):
            raise ValueError(
                "Wagi przeciwnikow nie moga byc ujemne"
            )

        if abs(total - 1.0) > 1e-8:
            raise ValueError(
                f"Wagi przeciwnikow musza sumowac sie do 1: {total}"
            )

        self.training_opponents = [
            (
                seeded_baseline(
                    name, cfg.seed + i + 100
                ),
                prob
            )
            for i, (name, prob) in enumerate(
                probabilities.items()
            )
            if prob > 0
        ]

        self.env = make_env()

        self.completed = 0
        self.transitions = 0
        self.training_steps = 0
        self.best_score = -float("inf")

        self._loss_sum = 0.0
        self._loss_n = 0

        self._eval_reward_sum = 0.0
        self._eval_reward_n = 0

        self.recent_rewards = deque(maxlen=5000)

        self.history = {
            "config": cfg.to_dict(),
            "reward_points": [],
            "epsilon": [],
            "loss": [],
            "eval": [],
        }

    # --------------------------------------------------
    # Uczenie sieci DQN
    # --------------------------------------------------

    def _maybe_train(self) -> None:
        cfg = self.cfg
        self.transitions += 1

        if len(self.replay) < max(
            cfg.warmup_steps, cfg.batch_size
        ):
            return

        if self.transitions % cfg.train_every:
            return

        loss = dqn_update(
            self.agent.net,
            self.target_net,
            self.optimizer,
            self.replay.sample(cfg.batch_size),
            cfg.gamma,
            cfg.grad_clip
        )

        self.training_steps += 1
        self._loss_sum += loss
        self._loss_n += 1

        if self.training_steps % cfg.loss_log_every == 0:
            self.history["loss"].append(
                (self.training_steps, loss)
            )

        if self.training_steps % cfg.target_update_every == 0:
            self.target_net.load_state_dict(
                self.agent.net.state_dict()
            )

    # --------------------------------------------------
    # Jedno rozdanie treningowe
    # --------------------------------------------------

    def _play_episode(self, episode: int) -> float:
        cfg, env = self.cfg, self.env

        env.reset(
            seed=1_000_000 + cfg.seed + episode
        )

        self.agent.reset()

        learner_id = SEATS[episode % 2]
        eps = cfg.epsilon(episode)

        candidates = self.training_opponents + [
            (
                snapshot,
                cfg.selfplay_opponent_prob / len(self.pool)
            )
            for snapshot in self.pool
        ]

        opponent = random.choices(
            [a for a, _ in candidates],
            weights=[w for _, w in candidates],
            k=1
        )[0]

        # Czyscimy pamiec kart z poprzedniego rozdania
        opponent.reset()

        pending = None
        episode_reward = 0.0

        for seat in env.agent_iter():
            obs, reward, terminated, truncated, _ = env.last()
            done = terminated or truncated

            if seat == learner_id:
                episode_reward += float(reward)

                if pending is not None:
                    old_state, old_action = pending

                    if done:
                        next_state = np.zeros(
                            MODEL_OBS_DIM,
                            dtype=np.float32
                        )
                        next_mask = np.zeros(
                            N_ACTIONS,
                            dtype=np.int8
                        )
                    else:
                        next_mask = np.array(
                            obs["action_mask"],
                            dtype=np.int8,
                            copy=True
                        )

                        next_state = self.agent.encode(
                            obs["observation"],
                            next_mask
                        )

                    self.replay.add(
                        old_state,
                        old_action,
                        reward * cfg.reward_scale,
                        next_state,
                        next_mask,
                        done
                    )

                    pending = None
                    self._maybe_train()

            if done:
                env.step(None)
                continue

            raw = obs["observation"]
            mask = obs["action_mask"]

            if seat == learner_id:
                encoded = self.agent.encode(raw, mask)

                action = self.agent.act_encoded(
                    encoded,
                    mask,
                    epsilon=eps
                )

                pending = (
                    encoded.copy(),
                    action
                )

            else:
                # NOWE: podajemy prawdziwe dane licytacji
                # agentom Monte Carlo i Heuristic.
                if hasattr(opponent, "set_betting_info"):
                    betting_info = get_betting_info(
                        env, seat
                    )

                    opponent.set_betting_info(
                        betting_info
                    )

                action = opponent.act(raw, mask)

            env.step(action)

        if pending is not None:
            raise RuntimeError(
                "Rozdanie zakonczone bez finalnej nagrody"
            )

        return episode_reward

    # --------------------------------------------------
    # Ewaluacja
    # --------------------------------------------------

    def _evaluate(self, n: int) -> None:
        cfg = self.cfg

        for i, baseline in enumerate(
            self.baselines.values()
        ):
            if hasattr(baseline, "rng"):
                baseline.rng.seed(
                    cfg.seed + EVAL_SEED + i
                )

        results = evaluate_against(
            self.agent,
            self.baselines,
            cfg.eval_hands_per_seat,
            cfg.eval_hands_overrides,
            seed=EVAL_SEED
        )

        summaries = {
            name: result.summary()
            for name, result in results.items()
        }

        score = float(np.mean([
            s["mean"]
            for s in summaries.values()
        ]))

        record = {
            "episode": n,
            "training_reward": (
                self._eval_reward_sum / self._eval_reward_n
                if self._eval_reward_n
                else None
            ),
            "mean_loss": (
                self._loss_sum / self._loss_n
                if self._loss_n
                else None
            ),
            "epsilon": cfg.epsilon(n),
            "score": score,
            "baselines": summaries,
        }

        self.history["eval"].append(record)

        self._loss_sum = 0.0
        self._loss_n = 0

        self._eval_reward_sum = 0.0
        self._eval_reward_n = 0

        self._save_model(
            "latest.pth",
            {"episode": n}
        )

        if score > self.best_score:
            self.best_score = score

            self._save_model(
                "best.pth",
                {
                    "episode": n,
                    "score": score
                }
            )

        self._save_training_state()
        self._dump_history()

        per_opp = " | ".join(
            f"{name}: {result['mean']:+.3f} "
            f"(WR {result['win_rate']:.2f})"
            for name, result in summaries.items()
        )

        tqdm.write(
            f"[ep {n}] "
            f"replay={len(self.replay)} "
            f"pool={len(self.pool)} "
            f"eps={record['epsilon']:.3f} "
            f"reward={record['training_reward']:.3f} "
            f"loss={record['mean_loss']} "
            f"score={score:+.3f} || {per_opp}"
        )

    # --------------------------------------------------
    # Zapisywanie modeli
    # --------------------------------------------------

    def _save_model(
        self,
        filename: str,
        extra: dict
    ) -> None:
        destination = self.out_dir / filename
        temporary = self.out_dir / (filename + ".tmp")

        self.agent.save(
            temporary,
            extra=extra
        )

        temporary.replace(destination)

    def _save_training_state(self) -> None:
        """Zapisuje stan treningu bez replay buffera."""

        data = {
            "format": 2,
            "config": self.cfg.to_dict(),
            "episode": self.completed,
            "model_state": self.agent.net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "pool_states": [
                snapshot.net.state_dict()
                for snapshot in self.pool
            ],
            "transitions": self.transitions,
            "training_steps": self.training_steps,
            "best_score": self.best_score,
            "history": self.history,
            "loss_sum": self._loss_sum,
            "loss_n": self._loss_n,
            "eval_reward_sum": self._eval_reward_sum,
            "eval_reward_n": self._eval_reward_n,
            "recent_rewards": list(self.recent_rewards),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        }

        temporary = self.out_dir / "resume.pth.tmp"

        torch.save(data, temporary)

        temporary.replace(
            self.out_dir / "resume.pth"
        )

    def load_training_state(self) -> None:
        path = self.out_dir / "resume.pth"

        if not path.exists():
            raise FileNotFoundError(
                f"Brak checkpointu: {path}"
            )

        # Wczytuj w ten sposob wylacznie wlasne pliki.
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        if data.get("format") != 2:
            raise ValueError(
                "Niekompatybilny format checkpointu"
            )

        old_config = data["config"]
        allowed_changes = {"episodes"}

        for key, old_value in old_config.items():
            if (
                key not in allowed_changes
                and self.cfg.to_dict().get(key) != old_value
            ):
                raise ValueError(
                    f"Zmieniono parametr '{key}'"
                )

        self.agent.net.load_state_dict(
            data["model_state"]
        )

        self.target_net.load_state_dict(
            data["target_state"]
        )

        self.optimizer.load_state_dict(
            data["optimizer_state"]
        )

        self.pool = []

        for weights in data["pool_states"]:
            snap = self.agent.snapshot(
                self.cfg.opponent_epsilon
            )

            snap.net.load_state_dict(weights)
            self.pool.append(snap)

        self.completed = int(data["episode"])
        self.transitions = int(data["transitions"])
        self.training_steps = int(data["training_steps"])
        self.best_score = float(data["best_score"])

        self.history = data["history"]

        self._loss_sum = float(data["loss_sum"])
        self._loss_n = int(data["loss_n"])

        self._eval_reward_sum = float(
            data["eval_reward_sum"]
        )
        self._eval_reward_n = int(
            data["eval_reward_n"]
        )

        self.recent_rewards = deque(
            data["recent_rewards"],
            maxlen=5000
        )

        random.setstate(data["python_rng"])
        np.random.set_state(data["numpy_rng"])
        torch.set_rng_state(data["torch_rng"])

        tqdm.write(
            f"Wznowiono od epizodu {self.completed}. "
            "Replay buffer bedzie odtwarzany od zera."
        )

    # --------------------------------------------------
    # Historia i wykresy
    # --------------------------------------------------

    def _dump_history(self) -> None:
        temporary = self.out_dir / "history.json.tmp"

        with temporary.open(
            "w",
            encoding="utf-8"
        ) as handle:
            json.dump(
                self.history,
                handle,
                ensure_ascii=False
            )

        temporary.replace(
            self.out_dir / "history.json"
        )

        try:
            plot_training_curves(
                self.history,
                self.out_dir
                / f"training_curves_{self.completed}.png"
            )
        except OSError as exc:
            tqdm.write(
                f"Nie udalo sie zapisac wykresu: {exc}"
            )

    # --------------------------------------------------
    # Glowna petla treningowa
    # --------------------------------------------------

    def run(self) -> None:
        try:
            with tqdm(
                initial=self.completed,
                total=self.cfg.episodes,
                desc="Training",
                unit="episode"
            ) as bar:

                for episode in range(
                    self.completed,
                    self.cfg.episodes
                ):
                    n = episode + 1

                    reward = self._play_episode(
                        episode
                    )

                    self.completed = n

                    self.recent_rewards.append(
                        reward
                    )

                    self._eval_reward_sum += reward
                    self._eval_reward_n += 1

                    if n % self.cfg.reward_log_every == 0:
                        self.history["reward_points"].append(
                            (
                                n,
                                float(np.mean(
                                    self.recent_rewards
                                ))
                            )
                        )

                        self.history["epsilon"].append(
                            (
                                n,
                                self.cfg.epsilon(episode)
                            )
                        )

                    if n % self.cfg.snapshot_every == 0:
                        self.pool.append(
                            self.agent.snapshot(
                                self.cfg.opponent_epsilon
                            )
                        )

                        if len(self.pool) > self.cfg.pool_size:
                            self.pool.pop(0)

                    if n % self.cfg.eval_every == 0:
                        self._evaluate(n)

                    bar.update(1)

                    if n % 100 == 0:
                        bar.set_postfix({
                            "eps": (
                                f"{self.cfg.epsilon(episode):.3f}"
                            ),
                            "replay": len(self.replay),
                            "pool": len(self.pool),
                        })

        except KeyboardInterrupt:
            tqdm.write(
                "Przerwano Ctrl+C; zapisuje stan."
            )

        finally:
            try:
                self._save_model(
                    "final.pth",
                    {"episode": self.completed}
                )
                self._save_training_state()
                self._dump_history()
            finally:
                self.env.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out-dir",
        default="runs/dqn_v2"
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    args = parser.parse_args()

    cfg = TrainConfig()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    trainer = Trainer(
        DQNAgent(),
        cfg,
        args.out_dir
    )

    if args.resume:
        trainer.load_training_state()

    trainer.run()


if __name__ == "__main__":
    main()

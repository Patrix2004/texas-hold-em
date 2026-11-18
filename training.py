"""Pętla treningowa DQN z self-play (pula zamrożonych kopii + losowy przeciwnik)."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from .agents.dqn_agent import DQNAgent
from .agents.heuristic_agent import HeuristicAgent
from .agents.monte_carlo_agent import MonteCarloAgent
from .agents.aggresive_agent import AggressiveAgent
from .agents.random_agent import RandomAgent
from .utils import make_baseline
from .config import TrainConfig
from .env import N_ACTIONS, OBS_DIM, SEATS, make_env
from .evaluation import EVAL_SEED, evaluate_against
from .plotting import plot_training_curves
from .replay import ReplayBuffer


def dqn_update(net, target_net, optimizer, batch, gamma: float, grad_clip: float) -> float:
    """Double DQN z maskowaniem nielegalnych akcji w następnym stanie."""
    states, actions, rewards, next_states, next_masks, dones = batch

    current_q = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = torch.zeros_like(rewards)
        not_done = ~dones
        if not_done.any():
            online_q = net(next_states[not_done]).masked_fill(~next_masks[not_done], float("-inf"))
            best = online_q.argmax(dim=1, keepdim=True)
            next_q[not_done] = target_net(next_states[not_done]).gather(1, best).squeeze(1)
        target = rewards + gamma * next_q

    loss = F.smooth_l1_loss(current_q, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


class Trainer:
    def __init__(self, agent: DQNAgent, cfg: TrainConfig, out_dir, baselines: dict | None = None):
        self.agent = agent
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.target_net = copy.deepcopy(agent.net).eval()
        self.optimizer = torch.optim.Adam(agent.net.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_capacity)

        self.baselines = baselines or {n: make_baseline(n) for n in cfg.eval_baselines}
        self.pool = [agent.snapshot(cfg.opponent_epsilon)]
        self.training_opponents = [
            (RandomAgent(), 0.20),
            (make_baseline("aggressive"), 0.25),
            (make_baseline("monte_carlo"), 0.25),
        ]
        
        self.env = make_env()
        self.transitions = 0
        self.training_steps = 0
        self.best_score = -float("inf")
        self._loss_sum = 0.0
        self._loss_n = 0
        self.history = {
            "config": cfg.to_dict(),
            "episode_rewards": [],
            "epsilon": [],
            "loss": [],
            "eval": [],
        }

    # ------------------------------------------------------------ uczenie
    def _maybe_train(self) -> None:
        cfg = self.cfg
        self.transitions += 1
        if len(self.replay) < cfg.warmup_steps or self.transitions % cfg.train_every:
            return

        batch = self.replay.sample(cfg.batch_size)
        loss = dqn_update(self.agent.net, self.target_net, self.optimizer,
                          batch, cfg.gamma, cfg.grad_clip)
        self.training_steps += 1
        self._loss_sum += loss
        self._loss_n += 1

        if self.training_steps % cfg.loss_log_every == 0:
            self.history["loss"].append((self.training_steps, loss))
        if self.training_steps % cfg.target_update_every == 0:
            self.target_net.load_state_dict(self.agent.net.state_dict())

    def _play_episode(self, episode: int) -> float:
        cfg, env = self.cfg, self.env
        env.reset()

        learner_id = SEATS[episode % 2]
        eps = cfg.epsilon(episode)
        opponents = self.training_opponents + [
            (snapshot, 0.30 / len(self.pool))
            for snapshot in self.pool
        ]

        opponent = random.choices(
            [agent for agent, _ in opponents],
            weights=[weight for _, weight in opponents],
            k=1
        )[0]

        pending = None
        episode_reward = 0.0

        for seat in env.agent_iter():
            obs, reward, terminated, truncated, _ = env.last()
            done = terminated or truncated

            if seat == learner_id:
                episode_reward += reward
                if pending is not None:
                    state, action = pending
                    if done:
                        next_state = np.zeros(OBS_DIM, dtype=np.float32)
                        next_mask = np.zeros(N_ACTIONS, dtype=np.int8)
                    else:
                        next_state, next_mask = obs["observation"], obs["action_mask"]
                    self.replay.add(state, action, reward * cfg.reward_scale,
                                    next_state, next_mask, done)
                    pending = None
                    self._maybe_train()

            if done:
                env.step(None)
                continue

            state, mask = obs["observation"], obs["action_mask"]
            if seat == learner_id:
                action = self.agent.act(state, mask, epsilon=eps)
                pending = (np.array(state, copy=True), action)
            else:
                action = opponent.act(state, mask)
            env.step(action)

        return episode_reward

    # --------------------------------------------------------- ewaluacja
    def _evaluate(self, n: int) -> None:
        cfg = self.cfg
        results = evaluate_against(self.agent, self.baselines, cfg.eval_hands_per_seat,
                                   cfg.eval_hands_overrides, seed=EVAL_SEED)
        summaries = {name: r.summary() for name, r in results.items()}
        score = float(np.mean([s["mean"] for s in summaries.values()]))
        recent = self.history["episode_rewards"][-cfg.eval_every:]

        record = {
            "episode": n,
            "selfplay_reward": float(np.mean(recent)) if recent else None,
            "mean_loss": self._loss_sum / self._loss_n if self._loss_n else None,
            "epsilon": cfg.epsilon(n),
            "score": score,
            "baselines": summaries,
        }
        self._loss_sum, self._loss_n = 0.0, 0
        self.history["eval"].append(record)

        self.agent.save(self.out_dir / "latest.pth", extra={"episode": n})
        if score > self.best_score:
            self.best_score = score
            self.agent.save(self.out_dir / "best.pth", extra={"episode": n, "score": score})
        self._dump_history()

        per_opp = " | ".join(f"{k}: {v['mean']:+.3f} (WR {v['win_rate']:.2f})"
                             for k, v in summaries.items())
        tqdm.write(f"[ep {n}] replay={len(self.replay)} pool={len(self.pool)} "
                f"eps={record['epsilon']:.3f} self-play={record['selfplay_reward']:.3f} "
                f"loss={record['mean_loss']} score={score:+.3f} || {per_opp}")

    def _dump_history(self) -> None:
        with open(self.out_dir / "history.json", "w") as f:
            json.dump(self.history, f)
        plot_training_curves(self.history, self.out_dir / "training_curves.png")

    # -------------------------------------------------------------- run
    def run(self) -> None:
        cfg = self.cfg
        try:
            with tqdm(total=cfg.episodes, desc="Training", unit="episode") as pbar:
                for episode in range(cfg.episodes):
                    n = episode + 1
                    self.history["episode_rewards"].append(round(self._play_episode(episode), 4))

                    if n % 1000 == 0:
                        self.history["epsilon"].append((n, cfg.epsilon(episode)))
                    if n % cfg.snapshot_every == 0:
                        self.pool.append(self.agent.snapshot(cfg.opponent_epsilon))
                        if len(self.pool) > cfg.pool_size:
                            self.pool.pop(0)
                    if n % cfg.eval_every == 0:
                        self._evaluate(n)

                    pbar.update(1)
                    pbar.set_postfix({
                        "eps": f"{cfg.epsilon(episode):.3f}",
                        "replay": len(self.replay),
                        "pool": len(self.pool),
                    })

        except KeyboardInterrupt:
            print("Przerwano — zapisuję bieżący stan.")
        finally:
            self.agent.save(self.out_dir / "final.pth",
                            extra={"episode": len(self.history["episode_rewards"])})
            self._dump_history()
            self.env.close()

if __name__ == "__main__":
    cfg = TrainConfig()
    agent = DQNAgent()

    trainer = Trainer(
        agent=agent,
        cfg=cfg,
        out_dir="runs/dqn"
    )

    trainer.run()
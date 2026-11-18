from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class TrainConfig:
    episodes: int = 10_000
    seed: int = 0

    # --- DQN ---
    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 500_000
    warmup_steps: int = 1_000
    train_every: int = 1 # co ile dodanych przejść robimy krok uczenia
    target_update_every: int = 1_000
    reward_scale: float = 1.0
    grad_clip: float = 10.0

    # --- eksploracja ---
    eps_start: float = 0.2
    eps_end: float = 0.05
    eps_decay_episodes: int = 50_000

    # --- self-play ---
    random_opponent_prob: float = 0.2
    opponent_epsilon: float = 0.05
    snapshot_every: int = 5_000
    pool_size: int = 10

    # --- ewaluacja w trakcie treningu ---
    eval_every: int = 5_000
    eval_hands_per_seat: int = 200
    eval_hands_overrides: dict = field(default_factory=lambda: {"monte_carlo": 50})
    eval_baselines: tuple = ("random", "heuristic", "monte_carlo", "aggressive")
    loss_log_every: int = 200

    def epsilon(self, episode: int) -> float:
        frac = min(1.0, episode / max(1, self.eps_decay_episodes))
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def to_dict(self) -> dict:
        d = asdict(self)
        d["eval_baselines"] = list(d["eval_baselines"])
        return d
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

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class TrainConfig:
    episodes: int = 10_000_000
    seed: int = 0

    # DQN
    lr: float = 5e-5
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 200_000  # 180D: duzy bufor zajmuje duzo RAM
    warmup_steps: int = 2_000
    train_every: int = 4
    target_update_every: int = 2_000
    reward_scale: float = 1.0
    grad_clip: float = 10.0

    # Eksploracja
    eps_start: float = 0.2
    eps_end: float = 0.05
    eps_decay_episodes: int = 300_000

    # Przeciwnicy treningowi: wagi musza sumowac sie do 1
    # random_opponent_prob: float = 0.20
    # aggressive_opponent_prob: float = 0.25
    # monte_carlo_opponent_prob: float = 0.25
    # heuristic_opponent_prob: float = 0.00  # niewidziany baseline
    # selfplay_opponent_prob: float = 0.30
    random_opponent_prob: float = 0.15
    aggressive_opponent_prob: float = 0.15
    monte_carlo_opponent_prob: float = 0.25
    heuristic_opponent_prob: float = 0.20
    selfplay_opponent_prob: float = 0.25
    opponent_epsilon: float = 0.05
    snapshot_every: int = 10_000
    pool_size: int = 10

    # Ewaluacja i zapis
    eval_every: int = 20_000
    eval_hands_per_seat: int = 200
    eval_hands_overrides: dict = field(default_factory=lambda: {"monte_carlo": 50})
    eval_baselines: tuple = ("random", "heuristic", "monte_carlo", "aggressive")
    loss_log_every: int = 200
    reward_log_every: int = 1_000

    def epsilon(self, episode: int) -> float:
        frac = min(1.0, max(0, episode) / max(1, self.eps_decay_episodes))
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def to_dict(self) -> dict:
        d = asdict(self)
        d["eval_baselines"] = list(d["eval_baselines"])
        return d
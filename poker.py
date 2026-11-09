import copy
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pettingzoo import make


class ReplayBuffer:
    def __init__(self, capacity=500000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, next_mask, done):
        self.buffer.append((
            state.copy(),
            action,
            reward,
            next_state.copy(),
            next_mask.copy(),
            done
        ))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(72, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)


def choose_action(model, state, mask, epsilon=0.0):
    legal_actions = np.flatnonzero(mask)

    if random.random() < epsilon:
        return int(random.choice(legal_actions))

    state = torch.as_tensor(state, dtype=torch.float32)

    with torch.no_grad():
        q_values = model(state)
        q_values = q_values.masked_fill(
            ~torch.as_tensor(mask, dtype=torch.bool), float('-inf')
        )

    return int(q_values.argmax().item())


def train_step(model, target_model, optimizer, replay, batch_size=128, gamma=0.99):
    batch = replay.sample(batch_size)
    states, actions, rewards, next_states, next_masks, dones = zip(*batch)

    states = torch.as_tensor(np.array(states), dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.long)
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32)
    next_masks = torch.as_tensor(np.array(next_masks), dtype=torch.bool)
    dones = torch.as_tensor(dones, dtype=torch.bool)

    current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = torch.zeros(batch_size)
        not_done = ~dones

        if not_done.any():
            online_q = model(next_states[not_done])
            online_q = online_q.masked_fill(~next_masks[not_done], float('-inf'))
            best_actions = online_q.argmax(dim=1, keepdim=True)
            next_q[not_done] = target_model(next_states[not_done]).gather(
                1, best_actions
            ).squeeze(1)

        expected_q = rewards + gamma * next_q

    loss = F.smooth_l1_loss(current_q, expected_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    return loss.item()


def evaluate(model, episodes_per_seat=200):
    eval_env = make('aec', 'classic/texas_holdem-v4')
    total_reward = 0.0
    wins = 0
    games = 0

    model.eval()

    for learner in ('player_0', 'player_1'):
        for episode in range(episodes_per_seat):
            eval_env.reset(seed=100000 + episode)
            episode_reward = 0.0

            for agent in eval_env.agent_iter():
                obs, reward, terminated, truncated, info = eval_env.last()

                if agent == learner:
                    episode_reward += reward

                if terminated or truncated:
                    eval_env.step(None)
                    continue

                mask = obs['action_mask']

                if agent == learner:
                    action = choose_action(model, obs['observation'], mask)
                else:
                    action = int(eval_env.action_space(agent).sample(mask))

                eval_env.step(action)

            total_reward += episode_reward
            wins += episode_reward > 0
            games += 1

    eval_env.close()
    model.train()
    return total_reward / games, wins / games


model = DQN()
model.load_state_dict(torch.load('dqn_poker.pth', map_location='cpu', weights_only=True))
model.train()

target_model = copy.deepcopy(model)
target_model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
replay = ReplayBuffer()

opponents = [copy.deepcopy(model).eval()]
env = make('aec', 'classic/texas_holdem-v4')

training_steps = 0
loss = None
rewards_history = []
episodes = 300000

avg_reward, win_rate = evaluate(model)
print('Przed self-play:', 'Avg reward:', round(avg_reward, 3), 'Win rate:', round(win_rate, 3))

for episode in range(episodes):
    env.reset()
    pending = None
    episode_reward = 0.0
    learner = 'player_0' if episode % 2 == 0 else 'player_1'
    epsilon = max(0.05, 0.2 - 0.15 * episode / 15000)
    random_opponent = random.random() < 0.2
    opponent = None if random_opponent else random.choice(opponents)

    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        done = terminated or truncated

        if agent == learner:
            episode_reward += reward

            if pending is not None:
                old_state, old_action = pending

                if done:
                    next_state = np.zeros(72, dtype=np.float32)
                    next_mask = np.zeros(4, dtype=np.int8)
                else:
                    next_state = obs['observation'].copy()
                    next_mask = obs['action_mask'].copy()

                replay.add(
                    old_state, old_action, reward,
                    next_state, next_mask, done
                )
                pending = None

                if len(replay) >= 1000:
                    loss = train_step(model, target_model, optimizer, replay)
                    training_steps += 1

                    if training_steps % 1000 == 0:
                        target_model.load_state_dict(model.state_dict())

        if done:
            env.step(None)
            continue

        state = obs['observation']
        mask = obs['action_mask']

        if agent == learner:
            action = choose_action(model, state, mask, epsilon)
            pending = (state.copy(), action)
        elif random_opponent:
            action = int(env.action_space(agent).sample(mask))
        else:
            action = choose_action(opponent, state, mask, epsilon=0.05)

        env.step(action)

    rewards_history.append(episode_reward)

    if (episode + 1) % 5000 == 0:
        opponents.append(copy.deepcopy(model).eval())

        if len(opponents) > 10:
            opponents.pop(0)

        torch.save(model.state_dict(), 'dqn_poker_selfplay.pth')
        avg_reward, win_rate = evaluate(model)

        print(
            'Episode:', episode + 1,
            'Replay size:', len(replay),
            'Loss:', round(loss, 4) if loss is not None else None,
            'Epsilon:', round(epsilon, 3),
            'Pool:', len(opponents),
            'Self-play reward:', round(np.mean(rewards_history[-5000:]), 3),
            'Vs random reward:', round(avg_reward, 3),
            'Win rate:', round(win_rate, 3)
        )

torch.save(model.state_dict(), 'dqn_poker_selfplay.pth')
env.close()

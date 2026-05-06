"""
Entraînement DQN pour le Sailing Challenge.
A exécuter LOCALEMENT (PyTorch requis) depuis la racine du repo.
Produit 'weights.npz' à inclure dans le zip de soumission.

Usage:
    python train_dqn.py
    python train_dqn.py --episodes 5000 --hidden 256
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from env_sailing import SailingEnv
from wind_scenarios import get_wind_scenario

# ==================== HYPERPARAMÈTRES ====================
DEFAULTS = dict(
    n_episodes=3000,
    hidden_dim=128,
    lr=1e-3,
    gamma=0.995,           # Même discount que l'environnement
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.9975,
    batch_size=256,
    buffer_size=100_000,
    target_update=100,
    save_path='weights.npz',
    reward_shaping=True,   # Récompense intermédiaire de progression
)


# ==================== RÉSEAU ====================
class QNetwork(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, n_actions=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# ==================== REPLAY BUFFER ====================
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.array(s2, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ==================== FEATURES ====================
def extract_features(obs):
    """
    Extrait 8 features pertinentes depuis l'observation brute (32778 éléments).
    Normalise pour aider le réseau.
    """
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    wx, wy = obs[4], obs[5]

    goal_x, goal_y = 64.0, 127.0
    dx = (goal_x - x) / 128.0
    dy = (goal_y - y) / 128.0

    return np.array([
        x / 128.0,
        y / 128.0,
        vx / 8.0,
        vy / 8.0,
        wx / 10.0,
        wy / 10.0,
        dx,
        dy
    ], dtype=np.float32)


def shaped_reward(obs, next_obs, reward, done):
    """
    Reward shaping : bonus de progression vers le but.
    Ne change pas la récompense terminale.
    """
    if reward > 0:  # But atteint
        return reward
    y_before = obs[1]
    y_after = next_obs[1]
    # Petit bonus si on avance vers le nord
    progress = (y_after - y_before) / 128.0
    return reward + 0.5 * progress


# ==================== ENTRAÎNEMENT ====================
def train_dqn(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    print(f"Config : {cfg}")

    scenarios = ['training_1', 'training_2', 'training_3']

    q_net = QNetwork(input_dim=8, hidden_dim=cfg['hidden_dim'], n_actions=9).to(device)
    q_target = QNetwork(input_dim=8, hidden_dim=cfg['hidden_dim'], n_actions=9).to(device)
    q_target.load_state_dict(q_net.state_dict())
    q_target.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=cfg['lr'])
    buffer = ReplayBuffer(cfg['buffer_size'])

    epsilon = cfg['epsilon_start']
    scores_window = deque(maxlen=100)
    success_window = deque(maxlen=100)
    best_success_rate = 0.0
    best_score = -np.inf

    for episode in range(cfg['n_episodes']):
        # Alterner les scénarios pour la généralisation
        scenario_name = scenarios[episode % len(scenarios)]
        scenario = get_wind_scenario(scenario_name)
        env = SailingEnv(**scenario)

        obs, _ = env.reset(seed=episode)
        state = extract_features(obs)
        total_reward = 0
        steps = 0
        episode_success = False
        raw_reward_sum = 0.0

        for step in range(500):
            if random.random() < epsilon:
                action = random.randint(0, 8)
            else:
                with torch.no_grad():
                    s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = q_net(s_t).argmax().item()

            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            true_success = env_reward > 0
            if true_success:
                episode_success = True

            raw_reward_sum += env_reward

            reward = env_reward
            if cfg['reward_shaping']:
                reward = shaped_reward(obs, next_obs, env_reward, done)

            next_state = extract_features(next_obs)
            buffer.push(state, action, reward, next_state, float(done))
            obs = next_obs
            state = next_state
            total_reward += reward
            steps += 1

            # Apprentissage
            if len(buffer) >= cfg['batch_size']:
                s_b, a_b, r_b, s2_b, d_b = buffer.sample(cfg['batch_size'])
                s_b = torch.FloatTensor(s_b).to(device)
                a_b = torch.LongTensor(a_b).to(device)
                r_b = torch.FloatTensor(r_b).to(device)
                s2_b = torch.FloatTensor(s2_b).to(device)
                d_b = torch.FloatTensor(d_b).to(device)

                q_vals = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    # Double DQN
                    next_acts = q_net(s2_b).argmax(1)
                    next_q = q_target(s2_b).gather(1, next_acts.unsqueeze(1)).squeeze(1)
                    targets = r_b + cfg['gamma'] * next_q * (1 - d_b)

                loss = nn.MSELoss()(q_vals, targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        if episode % cfg['target_update'] == 0:
            q_target.load_state_dict(q_net.state_dict())

        epsilon = max(cfg['epsilon_end'], epsilon * cfg['epsilon_decay'])

        # Succès = récompense originale > 0 (but atteint)
        # Note: avec reward shaping total_reward peut être > 0 sans succès
        # On re-vérifie via les étapes
        success = 1 if episode_success else 0
        scores_window.append(raw_reward_sum)
        success_window.append(success)

        if episode % 100 == 0 or episode == cfg['n_episodes'] - 1:
            avg = np.mean(scores_window)
            sr = np.mean(success_window)
            print(f"Ep {episode:4d} | Score: {avg:7.3f} | Succès: {sr:.2%} | ε: {epsilon:.3f}")

            # Sauvegarde du meilleur modèle
            current_score = np.mean(scores_window)
            current_success = np.mean(success_window)

            if (current_success > best_success_rate) or (
                current_success == best_success_rate and current_score > best_score
            ):
                best_success_rate = current_success
                best_score = current_score
                export_weights(q_net, cfg['save_path'])
    
    print(f"\nEntraînement terminé. Meilleur taux de succès: {best_success_rate:.2%}")
    print(f"Poids sauvés dans '{cfg['save_path']}'")
    return q_net


def export_weights(model, path):
    """Exporte les poids en numpy pour l'inférence sans torch."""
    weights = {}
    for i, (name, param) in enumerate(model.state_dict().items()):
        key = f'p{i}_{name.replace(".", "_")}'
        weights[key] = param.cpu().numpy()
    np.savez(path, **weights)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=DEFAULTS['n_episodes'])
    p.add_argument('--hidden', type=int, default=DEFAULTS['hidden_dim'])
    p.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    p.add_argument('--no-shaping', action='store_true')
    p.add_argument('--save', type=str, default=DEFAULTS['save_path'])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = DEFAULTS.copy()
    cfg['n_episodes'] = args.episodes
    cfg['hidden_dim'] = args.hidden
    cfg['lr'] = args.lr
    cfg['reward_shaping'] = not args.no_shaping
    cfg['save_path'] = args.save
    train_dqn(cfg)

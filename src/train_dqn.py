"""
Entraînement DQN pour le Sailing Challenge.
A exécuter localement (PyTorch requis).
Produit un fichier 'weights.npz' à utiliser dans my_agent.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import sys
import os

sys.path.append('src')
from env_sailing import SailingEnv
from wind_scenarios import get_wind_scenario

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
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(s2), np.array(d, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)

# ==================== FEATURE EXTRACTION ====================
def extract_features(obs):
    """
    Extrait les 8 features pertinentes de l'observation brute.
    obs format: [x, y, vx, vy, wx, wy, ...wind_field..., ...world_map...]
    """
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    wx, wy = obs[4], obs[5]
    
    # Normalisation de la position
    x_norm = x / 128.0
    y_norm = y / 128.0
    
    # Distance au but (normalisée)
    goal_x, goal_y = 64.0, 127.0
    dx = (goal_x - x) / 128.0
    dy = (goal_y - y) / 128.0
    
    return np.array([x_norm, y_norm, vx/8.0, vy/8.0, wx/10.0, wy/10.0, dx, dy], 
                    dtype=np.float32)

# ==================== ENTRAÎNEMENT DQN ====================
def train_dqn(
    n_episodes=3000,
    max_steps=500,
    lr=1e-3,
    gamma=0.995,  # Même que le discount de l'env
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    batch_size=256,
    buffer_size=100_000,
    target_update=100,  # Fréquence de mise à jour du réseau cible
    hidden_dim=128,
    save_path='weights.npz'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entraînement sur : {device}")
    
    # Rotation entre les 3 scénarios d'entraînement
    scenarios = ['training_1', 'training_2', 'training_3']
    
    # Réseaux Q et Q_target
    q_net = QNetwork(input_dim=8, hidden_dim=hidden_dim, n_actions=9).to(device)
    q_target = QNetwork(input_dim=8, hidden_dim=hidden_dim, n_actions=9).to(device)
    q_target.load_state_dict(q_net.state_dict())
    q_target.eval()
    
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size)
    
    epsilon = epsilon_start
    scores = []
    success_rate_window = deque(maxlen=100)
    
    for episode in range(n_episodes):
        # Rotate entre les scénarios pour généraliser
        scenario_name = scenarios[episode % len(scenarios)]
        scenario = get_wind_scenario(scenario_name)
        env = SailingEnv(**scenario)
        
        obs, _ = env.reset(seed=episode)
        state = extract_features(obs)
        total_reward = 0
        
        for step in range(max_steps):
            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.randint(0, 8)
            else:
                with torch.no_grad():
                    s_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_vals = q_net(s_tensor)
                    action = q_vals.argmax().item()
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = extract_features(next_obs)
            
            buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward
            
            # Apprentissage
            if len(buffer) >= batch_size:
                s_b, a_b, r_b, s2_b, d_b = buffer.sample(batch_size)
                
                s_b = torch.FloatTensor(s_b).to(device)
                a_b = torch.LongTensor(a_b).to(device)
                r_b = torch.FloatTensor(r_b).to(device)
                s2_b = torch.FloatTensor(s2_b).to(device)
                d_b = torch.FloatTensor(d_b).to(device)
                
                # Q(s,a) courant
                q_values = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                
                # Double DQN : action choisie par q_net, valeur estimée par q_target
                with torch.no_grad():
                    next_actions = q_net(s2_b).argmax(1)
                    next_q = q_target(s2_b).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target = r_b + gamma * next_q * (1 - d_b)
                
                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                optimizer.step()
            
            if done:
                break
        
        # Mise à jour du réseau cible
        if episode % target_update == 0:
            q_target.load_state_dict(q_net.state_dict())
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        scores.append(total_reward)
        success_rate_window.append(1.0 if total_reward > 0 else 0.0)
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            success_rate = np.mean(success_rate_window)
            print(f"Episode {episode:4d} | Score moyen: {avg_score:6.2f} | "
                  f"Succès: {success_rate:.2%} | Epsilon: {epsilon:.3f}")
    
    # Sauvegarde des poids en numpy
    export_weights_to_numpy(q_net, save_path)
    print(f"\nPoids exportés dans '{save_path}'")
    return q_net


def export_weights_to_numpy(model, path):
    """Exporte les poids PyTorch en matrices numpy."""
    weights = {}
    state_dict = model.state_dict()
    for i, (name, param) in enumerate(state_dict.items()):
        weights[f'param_{i}_{name.replace(".", "_")}'] = param.cpu().numpy()
    np.savez(path, **weights)
    print(f"Poids exportés : {list(weights.keys())}")


if __name__ == '__main__':
    train_dqn(n_episodes=3000, save_path='weights.npz')
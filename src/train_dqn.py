"""
train_dqn.py
============
Script d'entraînement DQN (Double DQN) pour le Sailing Challenge.

Usage :
    cd sailing_project
    python src/train_dqn.py

Le modèle entraîné est sauvegardé dans models/best_agent.npz.
Un checkpoint est sauvegardé tous les 100 épisodes pour reprendre l'entraînement.

Algorithme :
  - Double DQN (van Hasselt et al., 2016) : sépare la sélection et
    l'évaluation de l'action pour réduire la surestimation des Q-valeurs.
  - Experience Replay avec buffer de taille 50 000.
  - Target Network mis à jour toutes les 200 steps.
  - ε-greedy avec décroissance exponentielle.
  - Reward shaping basé sur le potentiel (Ng et al., 1999).
  - Entraînement sur les 3 scénarios de vent en rotation pour généraliser.
"""

import sys
import os
import time
import numpy as np

# Ajouter le répertoire racine au path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.env_sailing     import SailingEnv
from src.wind_scenarios  import get_wind_scenario
from src.mlp_numpy       import MLP
from src.replay_buffer   import ReplayBuffer
from src.features        import extract_features, N_FEATURES
from src.reward_shaping  import shaped_reward

N_ACTIONS = 9
SCENARIOS = ['training_1', 'training_2', 'training_3']

def random_wind_scenario(rng):
    dirs = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
        (-0.55, 1), (0.55, 1), (-0.55, -1), (0.55, -1),
    ]

    pattern = tuple(
        tuple(dirs[int(rng.integers(0, len(dirs)))] for _ in range(3))
        for _ in range(3)
    )

    return {
        "wind_init_params": {
            "base_speed": float(rng.uniform(8.0, 12.0)),
            "base_max_rotation_angle_degree": float(rng.uniform(5.0, 25.0)),
            "pattern": pattern,
        },
        "wind_evol_params": {
            "mean_rotation_angle_degree": float(rng.uniform(1.0, 5.0)),
            "std_rotation_angle_degree": float(rng.uniform(0.2, 1.5)),
        },
    }


def train(
    # ── Durée ──────────────────────────────────────────
    n_episodes       : int   = 4000,
    # ── Architecture réseau ────────────────────────────
    n_hidden1        : int   = 256,
    n_hidden2        : int   = 128,
    # ── Optimisation ───────────────────────────────────
    lr               : float = 5e-4,
    gamma            : float = 0.995,
    grad_clip        : float = 1.0,
    # ── Exploration ε-greedy ───────────────────────────
    epsilon_start    : float = 1.0,
    epsilon_end      : float = 0.05,
    epsilon_decay    : float = 0.9975,
    # ── Replay buffer ──────────────────────────────────
    buffer_size      : int   = 50_000,
    batch_size       : int   = 256,
    min_buffer       : int   = 1_000,    # steps avant de commencer à apprendre
    # ── Target network ─────────────────────────────────
    target_update_freq: int  = 200,      # steps entre chaque synchro
    # ── Sauvegarde ─────────────────────────────────────
    save_dir         : str   = None,
    # ── Reproductibilité ───────────────────────────────
    seed             : int   = 42,
):
    """
    Lance l'entraînement DQN et sauvegarde le meilleur modèle.

    Returns:
        net            : MLP entraîné
        rewards_history: liste des récompenses brutes par épisode
    """
    if save_dir is None:
        save_dir = os.path.join(ROOT, 'models')
    os.makedirs(save_dir, exist_ok=True)

    best_path  = os.path.join(save_dir, 'best_agent.npz')
    ckpt_path  = os.path.join(save_dir, 'checkpoint.npz')

    rng = np.random.default_rng(seed)

    # ── Réseaux ────────────────────────────────────────────────────────────
    net        = MLP(N_FEATURES, n_hidden1, n_hidden2, N_ACTIONS, seed=seed)
    target_net = MLP(N_FEATURES, n_hidden1, n_hidden2, N_ACTIONS, seed=seed)
    target_net.copy_from(net)

    # ── Buffer ─────────────────────────────────────────────────────────────
    buffer = ReplayBuffer(buffer_size, N_FEATURES)

    # ── État d'entraînement ────────────────────────────────────────────────
    epsilon       = epsilon_start
    total_steps   = 0
    start_episode = 0
    best_score    = -np.inf
    rewards_history = []

    # ── Reprise depuis checkpoint ──────────────────────────────────────────
    if os.path.exists(ckpt_path):
        try:
            net.load(ckpt_path)
            target_net.copy_from(net)
            meta = np.load(ckpt_path)
            start_episode = int(meta.get('episode', np.array(0)))
            epsilon       = float(meta.get('epsilon', np.array(epsilon_start)))
            total_steps   = int(meta.get('total_steps', np.array(0)))
            print(f"✓ Reprise depuis l'épisode {start_episode} | ε={epsilon:.4f}")
        except Exception as e:
            print(f"⚠ Impossible de charger le checkpoint : {e}")

    print(f"\n{'='*65}")
    print(f"  DQN Training — Sailing Challenge")
    print(f"{'='*65}")
    print(f"  Features    : {N_FEATURES}")
    print(f"  Architecture: {N_FEATURES} → {n_hidden1} → {n_hidden2} → {N_ACTIONS}")
    print(f"  Episodes    : {n_episodes}  |  LR={lr}  |  γ={gamma}")
    print(f"  Buffer      : {buffer_size}  |  Batch={batch_size}")
    print(f"  Scénarios   : {SCENARIOS}")
    print(f"{'='*65}\n")

    t0 = time.time()

    for episode in range(start_episode, n_episodes):

        # ── Choix du scénario (rotation) ───────────────────────────────────
        # 50 % scénarios officiels, 50 % scénarios randomisés
        if rng.random() < 0.5:
            scenario_name = SCENARIOS[episode % len(SCENARIOS)]
            scenario_params = get_wind_scenario(scenario_name)
        else:
            scenario_name = "random"
            scenario_params = random_wind_scenario(rng)

        env = SailingEnv(**scenario_params)

        obs_raw, _ = env.reset(seed=int(rng.integers(0, 100_000)))
        state      = extract_features(obs_raw)

        ep_env_reward = 0.0
        done = False
        step = 0

        # ── Boucle épisode ─────────────────────────────────────────────────
        while not done and step < 500:
            # ε-greedy
            if rng.random() < epsilon:
                action = int(rng.integers(0, N_ACTIONS))
            else:
                action = int(np.argmax(net.forward(state)))

            obs_raw2, env_rew, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            next_state = extract_features(obs_raw2)

            # Reward shaping
            rew = shaped_reward(obs_raw, obs_raw2, env_rew, done)

            buffer.push(state, action, rew, next_state, done)

            ep_env_reward += env_rew
            state   = next_state
            obs_raw = obs_raw2
            total_steps += 1
            step += 1

            # ── Apprentissage ──────────────────────────────────────────────
            if len(buffer) >= max(min_buffer, batch_size):
                s_b, a_b, r_b, ns_b, d_b = buffer.sample(batch_size, rng)

                # Double DQN : sélection avec online, évaluation avec target
                q_online_next  = net.forward(ns_b)
                best_actions   = np.argmax(q_online_next, axis=1)
                q_target_next  = target_net.forward(ns_b)
                q_next_vals    = q_target_next[np.arange(batch_size), best_actions]

                targets = r_b + gamma * (1.0 - d_b) * q_next_vals

                net.backward(s_b, a_b, targets, lr=lr, grad_clip=grad_clip)

            # ── Sync target network ────────────────────────────────────────
            if total_steps % target_update_freq == 0:
                target_net.copy_from(net)

        # ── Fin d'épisode ──────────────────────────────────────────────────
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_history.append(ep_env_reward)

        # ── Logs + sauvegarde tous les 100 épisodes ────────────────────────
        if (episode + 1) % 100 == 0:
            recent   = rewards_history[-100:]
            avg      = float(np.mean(recent))
            success  = sum(r > 0 for r in recent) / len(recent)
            elapsed  = time.time() - t0
            eta_min  = (elapsed / (episode - start_episode + 1)) * (n_episodes - episode - 1) / 60

            print(f"Ep {episode+1:5d}/{n_episodes} | {scenario_name:12s} | "
                  f"AvgRew={avg:6.1f} | Success={success:5.1%} | "
                  f"ε={epsilon:.4f} | Steps={total_steps:,} | "
                  f"ETA={eta_min:.0f}min")

            # Sauvegarde du meilleur modèle
            if avg > best_score:
                best_score = avg
                net.save(best_path)
                print(f"  ✓ Nouveau meilleur modèle sauvegardé (score={best_score:.2f})")

            # Checkpoint (pour reprendre l'entraînement)
            np.savez(ckpt_path,
                     W1=net.W1, b1=net.b1,
                     W2=net.W2, b2=net.b2,
                     W3=net.W3, b3=net.b3,
                     episode=np.array(episode + 1),
                     epsilon=np.array(epsilon),
                     total_steps=np.array(total_steps))

    # ── Sauvegarde finale ──────────────────────────────────────────────────
    net.save(best_path)
    np.save(os.path.join(save_dir, 'rewards_history.npy'), np.array(rewards_history))

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Entraînement terminé en {elapsed/60:.1f} min")
    print(f"  Meilleur score : {best_score:.2f}")
    print(f"  Modèle sauvegardé : {best_path}")
    print(f"{'='*65}")

    return net, rewards_history


if __name__ == '__main__':
    train()

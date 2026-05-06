"""
replay_buffer.py
================
Experience Replay Buffer pour DQN.
Stocke les transitions (s, a, r, s', done) et permet d'échantillonner
des mini-batches aléatoires pour l'entraînement.
"""

import numpy as np


class ReplayBuffer:
    """
    Buffer circulaire de taille fixe.

    Args:
        capacity  : nombre max de transitions stockées
        n_features: dimension du vecteur d'état
    """

    def __init__(self, capacity: int, n_features: int):
        self.capacity = capacity
        self.n_features = n_features
        self.size = 0
        self.ptr  = 0   # pointeur d'écriture (circulaire)

        self.states      = np.zeros((capacity, n_features), dtype=np.float32)
        self.actions     = np.zeros(capacity,               dtype=np.int32)
        self.rewards     = np.zeros(capacity,               dtype=np.float32)
        self.next_states = np.zeros((capacity, n_features), dtype=np.float32)
        self.dones       = np.zeros(capacity,               dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """Ajoute une transition au buffer."""
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = float(done)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        """Retourne un batch aléatoire de transitions."""
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size

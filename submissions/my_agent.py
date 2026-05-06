"""
DQN Agent for the Sailing Challenge - Numpy-only inference
Trained weights embedded as numpy arrays.
"""
import numpy as np
import os

try:
    from evaluator.base_agent import BaseAgent
except ImportError:
    from agents.base_agent import BaseAgent


def relu(x):
    return np.maximum(0, x)


class MyAgent(BaseAgent):
    """
    Agent DQN entraîné. Inférence en numpy pur (pas de torch requis).
    """

    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()
        self.grid_size = (128, 128)
        self.goal = np.array([64.0, 127.0])
        
        # Poids du réseau (chargés depuis le fichier ou embarqués)
        self.weights = None
        self._load_weights()

    def _load_weights(self):
        """Charge les poids depuis weights.npz s'il existe."""
        # Cherche le fichier dans le même dossier que ce script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(script_dir, 'weights.npz')
        
        if os.path.exists(weights_path):
            data = np.load(weights_path)
            keys = sorted(data.files)
            # Architecture: Linear(8,128) -> ReLU -> Linear(128,128) -> ReLU -> Linear(128,9)
            self.W1 = data[keys[0]]  # (128, 8)
            self.b1 = data[keys[1]]  # (128,)
            self.W2 = data[keys[2]]  # (128, 128)
            self.b2 = data[keys[3]]  # (128,)
            self.W3 = data[keys[4]]  # (9, 128)
            self.b3 = data[keys[5]]  # (9,)
        else:
            # Fallback: politique heuristique si pas de poids
            print("ATTENTION: weights.npz non trouvé, utilisation de la politique heuristique")
            self.W1 = None

    def _forward(self, state):
        """Passe forward du réseau (numpy)."""
        x = relu(self.W1 @ state + self.b1)
        x = relu(self.W2 @ x + self.b2)
        return self.W3 @ x + self.b3

    def _extract_features(self, obs):
        """Extrait les features pertinentes."""
        x, y = obs[0], obs[1]
        vx, vy = obs[2], obs[3]
        wx, wy = obs[4], obs[5]
        dx = (self.goal[0] - x) / 128.0
        dy = (self.goal[1] - y) / 128.0
        return np.array([x/128.0, y/128.0, vx/8.0, vy/8.0,
                         wx/10.0, wy/10.0, dx, dy], dtype=np.float32)

    def _heuristic_action(self, obs):
        """Politique heuristique de secours (tacking)."""
        x, y = obs[0], obs[1]
        wx, wy = obs[4], obs[5]
        
        # Direction vers le but
        dx = 64.0 - x
        dy = 127.0 - y
        
        # Si vent favorable vers le nord, aller nord
        if wy > 0:  # vent poussant vers le haut
            if abs(dx) < 5:
                return 0  # Nord
            elif dx > 0:
                return 1  # NE
            else:
                return 7  # NW
        else:
            # Tacking : zigzag à 45 degrés
            if x < 32:
                return 1  # NE
            elif x > 96:
                return 7  # NW
            elif dx > 0:
                return 1  # NE
            else:
                return 7  # NW

    def act(self, observation: np.ndarray) -> int:
        """Sélectionne une action."""
        if self.W1 is None:
            return self._heuristic_action(observation)
        
        features = self._extract_features(observation)
        q_values = self._forward(features)
        return int(np.argmax(q_values))

    def reset(self) -> None:
        """Reset entre épisodes."""
        pass

    def seed(self, seed=None) -> None:
        """Set random seed."""
        self.np_random = np.random.default_rng(seed)
"""
mlp_numpy.py
============
Réseau de neurones MLP implémenté entièrement en NumPy.
Compatible avec Codabench (pas de PyTorch ni TensorFlow requis).

Architecture : Input → Dense(ReLU) → Dense(ReLU) → Output
"""

import numpy as np
import os


class MLP:
    """
    Multi-Layer Perceptron à 2 couches cachées, entièrement en NumPy.

    Forward pass, backward pass (gradient descent), save/load inclus.
    """

    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, seed=42):
        rng = np.random.default_rng(seed)
        # Initialisation He (recommandée pour ReLU)
        self.W1 = rng.standard_normal((n_input, n_hidden1)) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros(n_hidden1, dtype=np.float32)
        self.W2 = rng.standard_normal((n_hidden1, n_hidden2)) * np.sqrt(2.0 / n_hidden1)
        self.b2 = np.zeros(n_hidden2, dtype=np.float32)
        self.W3 = rng.standard_normal((n_hidden2, n_output)) * np.sqrt(2.0 / n_hidden2)
        self.b3 = np.zeros(n_output, dtype=np.float32)

        self.W1 = self.W1.astype(np.float32)
        self.W2 = self.W2.astype(np.float32)
        self.W3 = self.W3.astype(np.float32)

    def forward(self, x):
        """
        Calcul forward. Accepte un vecteur (n_input,) ou un batch (batch, n_input).
        Retourne (n_output,) ou (batch, n_output).
        """
        single = (x.ndim == 1)
        if single:
            x = x[np.newaxis, :]
        h1 = np.maximum(0.0, x @ self.W1 + self.b1)    # ReLU
        h2 = np.maximum(0.0, h1 @ self.W2 + self.b2)   # ReLU
        out = h2 @ self.W3 + self.b3
        return out[0] if single else out

    def backward(self, x, actions, targets, lr=5e-4, grad_clip=1.0):
        """
        Mise à jour des poids par descente de gradient (MSE sur Q(s,a)).

        Args:
            x       : états (batch, n_input)
            actions : actions choisies (batch,) int
            targets : valeurs cibles Q(s,a) (batch,) float
            lr      : taux d'apprentissage
            grad_clip : valeur max du gradient (clipping)
        """
        batch = x.shape[0]

        # --- Forward avec cache ---
        h1 = np.maximum(0.0, x @ self.W1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.W2 + self.b2)
        q_all = h2 @ self.W3 + self.b3   # (batch, n_actions)

        # Erreur TD uniquement sur l'action choisie
        td_error = q_all[np.arange(batch), actions] - targets   # (batch,)

        # --- Backward ---
        grad_out = np.zeros_like(q_all)
        grad_out[np.arange(batch), actions] = (2.0 / batch) * td_error

        dW3 = h2.T @ grad_out
        db3 = grad_out.sum(axis=0)

        grad_h2 = grad_out @ self.W3.T
        grad_h2 *= (h2 > 0)   # masque ReLU

        dW2 = h1.T @ grad_h2
        db2 = grad_h2.sum(axis=0)

        grad_h1 = grad_h2 @ self.W2.T
        grad_h1 *= (h1 > 0)   # masque ReLU

        dW1 = x.T @ grad_h1
        db1 = grad_h1.sum(axis=0)

        # --- Gradient clipping ---
        for g in [dW1, dW2, dW3]:
            np.clip(g, -grad_clip, grad_clip, out=g)

        # --- Mise à jour SGD ---
        self.W1 -= lr * dW1;  self.b1 -= lr * db1
        self.W2 -= lr * dW2;  self.b2 -= lr * db2
        self.W3 -= lr * dW3;  self.b3 -= lr * db3

        return float(np.mean(td_error ** 2))   # MSE loss

    def copy_from(self, other):
        """Copie les poids d'un autre réseau (pour le target network)."""
        self.W1 = other.W1.copy(); self.b1 = other.b1.copy()
        self.W2 = other.W2.copy(); self.b2 = other.b2.copy()
        self.W3 = other.W3.copy(); self.b3 = other.b3.copy()

    def save(self, path):
        """Sauvegarde les poids dans un fichier .npz"""
        np.savez(path, W1=self.W1, b1=self.b1,
                       W2=self.W2, b2=self.b2,
                       W3=self.W3, b3=self.b3)

    def load(self, path):
        """Charge les poids depuis un fichier .npz"""
        if not path.endswith('.npz'):
            path = path + '.npz'
        d = np.load(path)
        self.W1 = d['W1']; self.b1 = d['b1']
        self.W2 = d['W2']; self.b2 = d['b2']
        self.W3 = d['W3']; self.b3 = d['b3']

    def weights_as_dict(self):
        """Retourne les poids sous forme de dict (pour embarquer dans l'agent final)."""
        return {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
        }

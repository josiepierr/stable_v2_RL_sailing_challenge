"""
save_agent.py
=============
Génère le fichier my_agent.py prêt pour la soumission Codabench.

Usage :
    cd sailing_project
    python src/save_agent.py --weights models/best_agent.npz

Le fichier généré est un .py autonome qui :
  - hérite de BaseAgent (import Codabench)
  - embarque les poids du réseau directement dans le code (pas de fichier externe)
  - effectue la même extraction de features que pendant l'entraînement
  - est zippable directement pour Codabench
"""

import os
import sys
import argparse
import textwrap
import zipfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.mlp_numpy  import MLP
from src.features   import N_FEATURES

N_ACTIONS = 9


def generate_agent_code(weights_path: str, n_hidden1=256, n_hidden2=128) -> str:
    """
    Charge les poids et génère le code Python de l'agent autonome.
    Les poids sont embarqués dans le code sous forme de listes Python.
    """
    net = MLP(N_FEATURES, n_hidden1, n_hidden2, N_ACTIONS)
    net.load(weights_path)

    def arr_to_str(arr):
        """Convertit un ndarray en liste Python (lisible dans le code)."""
        return repr(arr.tolist())

    W1_str = arr_to_str(net.W1)
    b1_str = arr_to_str(net.b1)
    W2_str = arr_to_str(net.W2)
    b2_str = arr_to_str(net.b2)
    W3_str = arr_to_str(net.W3)
    b3_str = arr_to_str(net.b3)

    code = f'''\
"""
my_agent.py — DQN Agent for the Sailing Challenge
===================================================
Agent pré-entraîné par Double DQN (numpy pur).
Compatible Codabench (pas de PyTorch requis).

Architecture MLP : {N_FEATURES} → {n_hidden1} → {n_hidden2} → {N_ACTIONS}
"""

import numpy as np
from evaluator.base_agent import BaseAgent   # import Codabench


# ══════════════════════════════════════════════════════════════════════════
# CONSTANTES DE L'ENVIRONNEMENT
# ══════════════════════════════════════════════════════════════════════════
_GRID      = 128
_W_START   = 6
_W_SIZE    = _GRID * _GRID * 2      # 32 768
_WLD_START = _W_START + _W_SIZE
_ZONES     = 8
_ZONE_SZ   = _GRID // _ZONES        # 16
_PATCH     = 5
_GOAL      = np.array([64.0, 127.0])
_ISL_X1, _ISL_Y1 = 38, 43
_ISL_X2, _ISL_Y2 = 90, 85
_ISL_CX = (_ISL_X1 + _ISL_X2) / 2
_ISL_CY = (_ISL_Y1 + _ISL_Y2) / 2


def _extract_features(obs):
    """Transforme l\'observation brute (49 158,) en vecteur de features."""
    x,  y  = float(obs[0]), float(obs[1])
    vx, vy = float(obs[2]), float(obs[3])
    wx, wy = float(obs[4]), float(obs[5])

    wf = obs[_W_START:_WLD_START].reshape(_GRID, _GRID, 2)

    # Position
    fp = np.array([x / _GRID, y / _GRID], dtype=np.float32)
    # Vitesse
    fv = np.array([vx / 8.0, vy / 8.0], dtype=np.float32)
    # Vent local (direction)
    wn = np.sqrt(wx*wx + wy*wy) + 1e-8
    fw = np.array([wx / wn, wy / wn], dtype=np.float32)
    # Direction au but
    fg = np.array([(_GOAL[0]-x)/_GRID, (_GOAL[1]-y)/_GRID], dtype=np.float32)
    # Direction à l\'île
    fi = np.array([(_ISL_CX-x)/_GRID, (_ISL_CY-y)/_GRID], dtype=np.float32)
    # Distances aux bords de l\'île
    fb = np.array([(x-_ISL_X1)/_GRID, (_ISL_X2-x)/_GRID,
                   (y-_ISL_Y1)/_GRID, (_ISL_Y2-y)/_GRID], dtype=np.float32)
    # Vent moyen par zone 8x8
    fz = wf.reshape(_ZONES, _ZONE_SZ, _ZONES, _ZONE_SZ, 2).mean(axis=(1,3)).reshape(-1) / 10.0
    fz = fz.astype(np.float32)
    # Patch local 5x5
    xi = int(np.clip(x, 0, _GRID-1))
    yi = int(np.clip(y, 0, _GRID-1))
    h  = _PATCH // 2
    ys = np.clip(np.arange(yi-h, yi+h+1), 0, _GRID-1)
    xs = np.clip(np.arange(xi-h, xi+h+1), 0, _GRID-1)
    fp2 = (wf[np.ix_(ys, xs)] / 10.0).reshape(-1).astype(np.float32)

    return np.concatenate([fp, fv, fw, fg, fi, fb, fz, fp2])


# ══════════════════════════════════════════════════════════════════════════
# POIDS DU RÉSEAU (embarqués)
# ══════════════════════════════════════════════════════════════════════════
_W1 = np.array({W1_str}, dtype=np.float32)
_b1 = np.array({b1_str}, dtype=np.float32)
_W2 = np.array({W2_str}, dtype=np.float32)
_b2 = np.array({b2_str}, dtype=np.float32)
_W3 = np.array({W3_str}, dtype=np.float32)
_b3 = np.array({b3_str}, dtype=np.float32)


def _forward(x):
    """Forward pass MLP (numpy pur)."""
    h1  = np.maximum(0.0, x @ _W1 + _b1)
    h2  = np.maximum(0.0, h1 @ _W2 + _b2)
    return h2 @ _W3 + _b3


# ══════════════════════════════════════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════════════════════════════════════
class MyAgent(BaseAgent):
    """
    Agent DQN pré-entraîné pour le Sailing Challenge.
    Politique fixe — aucun apprentissage pendant l\'évaluation.
    """

    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()

    def act(self, observation: np.ndarray) -> int:
        """Retourne l\'action avec la plus haute Q-valeur."""
        features = _extract_features(observation)
        q_values = _forward(features)
        return int(np.argmax(q_values))

    def reset(self) -> None:
        """Réinitialise l\'état interne (rien à faire pour cet agent)."""
        pass

    def seed(self, seed=None) -> None:
        """Fixe la graine aléatoire."""
        self.np_random = np.random.default_rng(seed)
'''
    return code


def save_submission(weights_path: str,
                    output_dir: str = None,
                    n_hidden1: int = 256,
                    n_hidden2: int = 128):
    """
    Génère my_agent.py et my_submission.zip dans output_dir.
    """
    if output_dir is None:
        output_dir = os.path.join(ROOT, 'submissions')
    os.makedirs(output_dir, exist_ok=True)

    agent_path = os.path.join(output_dir, 'my_agent.py')
    zip_path   = os.path.join(output_dir, 'my_submission.zip')

    print(f"Chargement des poids : {weights_path}")
    code = generate_agent_code(weights_path, n_hidden1, n_hidden2)

    with open(agent_path, 'w') as f:
        f.write(code)
    print(f"✓ Agent généré : {agent_path}")

    # Créer le zip avec le .py à la racine
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(agent_path, arcname='my_agent.py')
    print(f"✓ ZIP de soumission créé : {zip_path}")

    # Vérification
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
    print(f"  Contenu du ZIP : {names}")
    assert 'my_agent.py' in names, "ERREUR : my_agent.py n'est pas à la racine du ZIP!"
    print("  Structure ZIP correcte ✓")

    return agent_path, zip_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Génère le fichier de soumission Codabench')
    parser.add_argument('--weights',   default='models/best_agent.npz',
                        help='Chemin vers les poids .npz')
    parser.add_argument('--output',    default='submissions',
                        help='Répertoire de sortie')
    parser.add_argument('--n_hidden1', type=int, default=256)
    parser.add_argument('--n_hidden2', type=int, default=128)
    args = parser.parse_args()

    weights_path = os.path.join(ROOT, args.weights)
    save_submission(weights_path, os.path.join(ROOT, args.output),
                    args.n_hidden1, args.n_hidden2)

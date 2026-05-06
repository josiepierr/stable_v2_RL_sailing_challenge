"""
reward_shaping.py
=================
Reward shaping basé sur le potentiel (Ng et al., 1999).

La formule F(s, s') = γ · Φ(s') - Φ(s) garantit que la politique
optimale ne change pas (théorème de préservation de politique).

Notre potentiel Φ(s) = - distance_normalisée_au_but

On ajoute aussi :
  - une pénalité si le bateau est dans la zone dangereuse de l'île
  - une pénalité si le bateau est bloqué (même position deux steps de suite)
"""

import numpy as np

# ── Constantes (cohérentes avec features.py) ───────────────────────────────
GRID_SIZE  = 128
ISLAND_X1, ISLAND_Y1 = 38, 43
ISLAND_X2, ISLAND_Y2 = 90, 85
GOAL = np.array([64.0, 127.0])

GAMMA = 0.995           # facteur de discount de l'environnement
SHAPING_SCALE = 5.0     # poids du shaping par rapport à la récompense brute
ISLAND_MARGIN = 6       # marge en cellules autour de l'île
ISLAND_PENALTY = -0.5   # pénalité par step dans la zone dangereuse
STUCK_PENALTY  = -0.1   # pénalité si le bateau ne bouge pas


def potential(x: float, y: float) -> float:
    """Potentiel = - distance normalisée au but."""
    dist = np.sqrt((x - GOAL[0])**2 + (y - GOAL[1])**2)
    return -dist / (GRID_SIZE * np.sqrt(2.0))


def shaped_reward(
    obs:      np.ndarray,
    next_obs: np.ndarray,
    env_reward: float,
    done:     bool,
) -> float:
    """
    Calcule la récompense shapée.

    Args:
        obs       : observation au step t   (tableau brut de l'env)
        next_obs  : observation au step t+1
        env_reward: récompense brute de l'environnement (100 si but, sinon 0)
        done      : True si l'épisode est terminé

    Returns:
        récompense totale shapée
    """
    x,  y  = float(obs[0]),      float(obs[1])
    x2, y2 = float(next_obs[0]), float(next_obs[1])

    # Shaping basé sur le potentiel
    phi_now  = potential(x,  y)
    phi_next = potential(x2, y2)
    shaping  = GAMMA * phi_next - phi_now

    # Pénalité zone dangereuse (marge autour de l'île)
    in_danger = (
        (ISLAND_X1 - ISLAND_MARGIN) <= x2 <= (ISLAND_X2 + ISLAND_MARGIN) and
        (ISLAND_Y1 - ISLAND_MARGIN) <= y2 <= (ISLAND_Y2 + ISLAND_MARGIN)
    )
    island_pen = ISLAND_PENALTY if in_danger else 0.0

    # Pénalité blocage (bateau immobile, hors but)
    stuck = (x == x2 and y == y2 and not done)
    stuck_pen = STUCK_PENALTY if stuck else 0.0

    return env_reward + SHAPING_SCALE * shaping + island_pen + stuck_pen

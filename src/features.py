"""
features.py
===========
Feature engineering pour le Sailing Challenge.

L'observation brute fait 49 158 valeurs (x, y, vx, vy, wx, wy,
+ champ de vent 128x128x2 + world_map 128x128).
On la compresse en ~190 features pertinentes.

Structure de l'observation brute :
  obs[0:2]                           -> (x, y)
  obs[2:4]                           -> (vx, vy)
  obs[4:6]                           -> (wx, wy) vent local
  obs[6 : 6+128*128*2]               -> champ de vent aplati
  obs[6+128*128*2 : 6+128*128*2+128*128] -> world_map aplati
"""

import numpy as np

# ── Constantes de l'environnement ──────────────────────────────────────────
GRID_SIZE   = 128
WIND_FIELD_SIZE = GRID_SIZE * GRID_SIZE * 2          # 32 768
WORLD_SIZE  = GRID_SIZE * GRID_SIZE                   # 16 384
WIND_START  = 6
WORLD_START = WIND_START + WIND_FIELD_SIZE            # 32 774

# Île (coordonnées du code source env_sailing.py)
ISLAND_X1, ISLAND_Y1 = 38, 43    # coin bas-gauche du rectangle
ISLAND_X2, ISLAND_Y2 = 90, 85    # coin haut-droit du rectangle
ISLAND_CX  = (ISLAND_X1 + ISLAND_X2) / 2   # 64
ISLAND_CY  = (ISLAND_Y1 + ISLAND_Y2) / 2   # 64

GOAL  = np.array([64.0, 127.0])
START = np.array([64.0,   0.0])

# ── Paramètres du feature engineering ──────────────────────────────────────
WIND_ZONES   = 8      # on divise en 8x8 zones → 8*8*2 = 128 features
ZONE_SIZE    = GRID_SIZE // WIND_ZONES   # 16 cellules/zone
LOCAL_PATCH  = 5      # patch 5x5 autour du bateau → 5*5*2 = 50 features

# Nombre total de features :
#   2  (position normalisée)
#   2  (vitesse normalisée)
#   2  (vent local normalisé : direction)
#   2  (distance+direction au but)
#   2  (distance+direction au centre de l'île)
#   4  (distances aux 4 bords de l'île)
#   128 (vent moyen par zone 8x8)
#   50  (patch vent local 5x5)
# ─────────────────────────────────
# TOTAL : 192
N_FEATURES = 2 + 2 + 2 + 2 + 2 + 4 + WIND_ZONES*WIND_ZONES*2 + LOCAL_PATCH*LOCAL_PATCH*2


def extract_features(obs: np.ndarray) -> np.ndarray:
    """
    Transforme l'observation brute (49 158,) en vecteur de features (192,).

    Complètement vectorisé — environ 0.3 ms par appel.
    """
    x,  y  = float(obs[0]), float(obs[1])
    vx, vy = float(obs[2]), float(obs[3])
    wx, wy = float(obs[4]), float(obs[5])

    wind_flat  = obs[WIND_START:WORLD_START]
    wind_field = wind_flat.reshape(GRID_SIZE, GRID_SIZE, 2)   # (H, W, 2)

    # 1. Position normalisée [0, 1]
    feat_pos = np.array([x / GRID_SIZE, y / GRID_SIZE], dtype=np.float32)

    # 2. Vitesse normalisée (max_speed = 8)
    feat_vel = np.array([vx / 8.0, vy / 8.0], dtype=np.float32)

    # 3. Direction du vent local (vecteur unitaire)
    w_norm = np.sqrt(wx*wx + wy*wy) + 1e-8
    feat_wind_local = np.array([wx / w_norm, wy / w_norm], dtype=np.float32)

    # 4. Direction+distance au but (normalisées)
    dx_goal = (GOAL[0] - x) / GRID_SIZE
    dy_goal = (GOAL[1] - y) / GRID_SIZE
    feat_goal = np.array([dx_goal, dy_goal], dtype=np.float32)

    # 5. Direction+distance au centre de l'île (pour l'évitement)
    dx_isl = (ISLAND_CX - x) / GRID_SIZE
    dy_isl = (ISLAND_CY - y) / GRID_SIZE
    feat_island_dir = np.array([dx_isl, dy_isl], dtype=np.float32)

    # 6. Distances signées aux 4 bords de l'île (positif = en dehors de ce bord)
    d_left   = (x - ISLAND_X1) / GRID_SIZE
    d_right  = (ISLAND_X2 - x) / GRID_SIZE
    d_bottom = (y - ISLAND_Y1) / GRID_SIZE
    d_top    = (ISLAND_Y2 - y) / GRID_SIZE
    feat_island_sides = np.array([d_left, d_right, d_bottom, d_top], dtype=np.float32)

    # 7. Vent moyen par zone 8x8 — vectorisé
    # wind_field (128,128,2) → (8,16,8,16,2) → mean sur axes (1,3)
    wf_b = wind_field.reshape(WIND_ZONES, ZONE_SIZE, WIND_ZONES, ZONE_SIZE, 2)
    zone_means = wf_b.mean(axis=(1, 3)) / 10.0   # (8, 8, 2) normalisé
    feat_wind_zones = zone_means.reshape(-1).astype(np.float32)  # (128,)

    # 8. Patch de vent local 5x5 centré sur le bateau — vectorisé
    xi = int(np.clip(x, 0, GRID_SIZE - 1))
    yi = int(np.clip(y, 0, GRID_SIZE - 1))
    half = LOCAL_PATCH // 2
    ys = np.clip(np.arange(yi - half, yi + half + 1), 0, GRID_SIZE - 1)
    xs = np.clip(np.arange(xi - half, xi + half + 1), 0, GRID_SIZE - 1)
    patch = wind_field[np.ix_(ys, xs)] / 10.0   # (5, 5, 2)
    feat_patch = patch.reshape(-1).astype(np.float32)   # (50,)

    # Concaténation finale
    features = np.concatenate([
        feat_pos,          # 2
        feat_vel,          # 2
        feat_wind_local,   # 2
        feat_goal,         # 2
        feat_island_dir,   # 2
        feat_island_sides, # 4
        feat_wind_zones,   # 128
        feat_patch,        # 50
    ])
    return features   # (192,)


def n_features() -> int:
    """Retourne le nombre de features (utile pour initialiser le réseau)."""
    return N_FEATURES

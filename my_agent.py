"""
Wind-Aware A* Agent — version rapide pour Codabench.

Optimisations pour tenir dans le budget temps (1800s / 50 seeds = 36s/seed) :
  - A* limité à 8000 nœuds max (< 1s sur CPU lent)
  - Grille COARSE 32x32 au lieu de 128x128 pour le planning
  - Replanning limité toutes les 30 actions
  - Fallback heuristique immédiat si A* dépasse son budget

Soumission :
    zip my_submission.zip my_agent.py
"""

import numpy as np
import heapq
import time

try:
    from evaluator.base_agent import BaseAgent
except ImportError:
    from agents.base_agent import BaseAgent


# ──────────────────────────────────────────────────────────
#  PHYSIQUE (reproduction exacte de sailing_physics.py)
# ──────────────────────────────────────────────────────────

_DIRS = np.array([
    [0, 1],[1, 1],[1, 0],[1,-1],
    [0,-1],[-1,-1],[-1,0],[-1,1],[0, 0]
], dtype=np.float64)

_PERF    = 0.4
_MAXSPD  = 8.0
_INERTIA = 0.3


def _eff(boat_d, wind):
    wn = np.linalg.norm(wind)
    bn = np.linalg.norm(boat_d)
    if wn < 1e-8 or bn < 1e-8:
        return 0.0
    a = float(np.arccos(np.clip(np.dot(-wind/wn, boat_d/bn), -1., 1.)))
    if a < np.pi/4:        return 0.05
    if a < np.pi/2:        return 0.5 + 0.5*(a - np.pi/4)/(np.pi/4)
    if a < 3*np.pi/4:      return 1.0
    return max(0.5, 1.0 - 0.5*(a - 3*np.pi/4)/(np.pi/4))


def _physics(px, py, vx, vy, action, wind):
    """Un step exact de l'env. Retourne (nx,ny,nvx,nvy) entiers."""
    d  = _DIRS[action]
    wn = np.linalg.norm(wind)
    if wn > 1e-8 and action < 8:
        eff = _eff(d, wind)
        tv  = d * eff * wn * _PERF
        spd = np.linalg.norm(tv)
        if spd > _MAXSPD:
            tv = tv / spd * _MAXSPD
        nv = tv + _INERTIA * (np.array([vx, vy]) - tv)
        if np.linalg.norm(nv) > _MAXSPD:
            nv = nv / np.linalg.norm(nv) * _MAXSPD
    else:
        nv = _INERTIA * np.array([vx, vy])

    # Discrétisation identique à l'env
    nvd = np.where(nv < 0, np.ceil(nv), np.floor(nv)).astype(np.int32)
    nx = int(np.clip(px + nvd[0], 0, 127))
    ny = int(np.clip(py + nvd[1], 0, 127))
    return nx, ny, int(nvd[0]), int(nvd[1])


# ──────────────────────────────────────────────────────────
#  A* SUR GRILLE COARSE (SCALE = 4 → grille 32x32)
#  Réduit l'espace d'états d'un facteur 16
# ──────────────────────────────────────────────────────────

SCALE = 4   # 128 / SCALE = 32

def _coarse(v):
    """Convertit coordonnée fine en coordonnée coarse."""
    return int(v) // SCALE

def _fine_center(c):
    """Coordonnée fine du centre d'une cellule coarse."""
    return c * SCALE + SCALE // 2


def _avg_wind(wind_field, cy, cx):
    """Vent moyen sur la cellule coarse (cx,cy) de la grille 32x32."""
    y0 = cy * SCALE; y1 = min(y0 + SCALE, 128)
    x0 = cx * SCALE; x1 = min(x0 + SCALE, 128)
    patch = wind_field[y0:y1, x0:x1, :]   # (SCALE,SCALE,2)
    return patch.mean(axis=(0,1))


def _island_coarse(island_map):
    """Masque île sur grille 32x32 : 1 si au moins une cellule fine est île."""
    H, W = 32, 32
    mask = np.zeros((H, W), dtype=np.int8)
    for cy in range(H):
        for cx in range(W):
            y0=cy*SCALE; y1=min(y0+SCALE,128)
            x0=cx*SCALE; x1=min(x0+SCALE,128)
            if island_map[y0:y1, x0:x1].max() > 0:
                mask[cy, cx] = 1
    return mask


def _heur(cx, cy, gcx, gcy):
    return max(abs(cx-gcx), abs(cy-gcy)) * 1.0


def astar_coarse(pos, vel, wind_field, island_map,
                 goal=(64, 127), max_nodes=8000, time_budget=5.0):
    """
    A* sur grille 32x32. Beaucoup plus rapide que sur 128x128.
    Retourne la liste d'actions 0-7, ou [] si échec/timeout.
    """
    t0 = time.time()
    isl = _island_coarse(island_map)

    # Construire le vent coarse une seule fois
    wind_c = np.zeros((32, 32, 2), dtype=np.float32)
    for cy in range(32):
        for cx in range(32):
            wind_c[cy, cx] = _avg_wind(wind_field, cy, cx)

    sx, sy  = _coarse(pos[0]), _coarse(pos[1])
    svx,svy = int(vel[0]), int(vel[1])
    gcx, gcy = _coarse(goal[0]), _coarse(goal[1])

    start = (sx, sy, svx, svy)
    heap  = [(_heur(sx,sy,gcx,gcy), 0.0, start, [])]
    best  = {}
    nodes = 0

    while heap and nodes < max_nodes:
        if time.time() - t0 > time_budget:
            break

        f, g, state, path = heapq.heappop(heap)
        cx, cy, vx, vy = state

        if cx == gcx and cy == gcy:
            return path

        if best.get(state, 1e18) <= g:
            continue
        best[state] = g
        nodes += 1

        # Position fine du centre de la cellule
        fx = _fine_center(cx); fy = _fine_center(cy)
        wind = wind_c[cy, cx].astype(np.float64)

        for a in range(8):
            nx, ny, nvx, nvy = _physics(fx, fy, vx, vy, a, wind)
            ncx, ncy = _coarse(nx), _coarse(ny)

            if isl[ncy, ncx]:
                continue

            ns  = (ncx, ncy, nvx, nvy)
            ng  = g + 1.0
            if best.get(ns, 1e18) <= ng:
                continue

            h = _heur(ncx, ncy, gcx, gcy)
            heapq.heappush(heap, (ng+h, ng, ns, path+[a]))

    return []   # Echec ou timeout → fallback


# ──────────────────────────────────────────────────────────
#  AGENT
# ──────────────────────────────────────────────────────────

class MyAgent(BaseAgent):
    """
    Agent A* Wind-Aware rapide — exploite le champ de vent complet.

    Planning sur grille réduite 32x32 pour tenir dans le budget temps
    de Codabench (1800s / 50 seeds). Replanning périodique léger.
    """

    REPLAN_EVERY = 30    # Steps entre deux replanifications
    MAX_NODES    = 8000  # Nœuds A* max par appel (~0.3s)
    TIME_BUDGET  = 4.0   # Secondes max par appel A*

    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()
        self._plan     = []
        self._pidx     = 0
        self._since    = 0

    def reset(self):
        self._plan  = []
        self._pidx  = 0
        self._since = 0

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    # ── Parsing ────────────────────────────────────────────
    @staticmethod
    def _parse(obs):
        pos  = np.array([float(obs[0]), float(obs[1])])
        vel  = np.array([float(obs[2]), float(obs[3])])
        wf   = obs[6 : 6 + 128*128*2].reshape(128, 128, 2).astype(np.float64)
        imap = obs[6 + 128*128*2 :].reshape(128, 128)
        return pos, vel, wf, imap

    # ── Replanning ─────────────────────────────────────────
    def _replan(self, obs):
        pos, vel, wf, imap = self._parse(obs)
        self._plan  = astar_coarse(pos, vel, wf, imap,
                                   goal=(64, 127),
                                   max_nodes=self.MAX_NODES,
                                   time_budget=self.TIME_BUDGET)
        self._pidx  = 0
        self._since = 0

    # ── Heuristique de louvoyage (fallback rapide) ─────────
    def _heuristic(self, obs):
        x, y   = float(obs[0]), float(obs[1])
        wx, wy = float(obs[4]), float(obs[5])
        # Au-dessus de l'île
        if y > 87:
            dx = 64. - x
            if abs(dx) < 4: return 0
            return 1 if dx > 0 else 7
        # Dans/sur l'île
        if 36 <= x <= 92 and y >= 15:
            return 6 if x >= 64 else 2
        # Vent favorable
        if wy > 0.3:
            dx = 64. - x
            if abs(dx) < 5: return 0
            return 1 if dx > 0 else 7
        # Louvoyage
        if x < 15:   return 1
        if x > 112:  return 7
        return 1 if x < 64 else 7

    # ── act() ──────────────────────────────────────────────
    def act(self, observation: np.ndarray) -> int:
        need = (
            not self._plan
            or self._pidx >= len(self._plan)
            or self._since >= self.REPLAN_EVERY
        )
        if need:
            self._replan(observation)

        if self._pidx < len(self._plan):
            a = self._plan[self._pidx]
            self._pidx  += 1
            self._since += 1
            return int(a)

        return self._heuristic(observation)

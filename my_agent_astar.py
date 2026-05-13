"""
Wind-Aware A* Planning Agent for the RL Sailing Challenge.

Stratégie : lit le champ de vent COMPLET depuis l'observation (128x128x2),
planifie une trajectoire via A* qui respecte la physique exacte du voilier,
et replannifie périodiquement pour s'adapter à l'évolution du vent.

Avantages :
  - Zéro entraînement requis
  - Généralise parfaitement à tout scénario de vent inconnu
  - Utilise l'île exacte de l'observation (pas hardcodée)
  - Physique identique à l'environnement

Soumission :  zip my_submission.zip my_agent_astar.py
"""

import numpy as np
import heapq

try:
    from evaluator.base_agent import BaseAgent
except ImportError:
    from agents.base_agent import BaseAgent


# ============================================================
#  PHYSIQUE (reproduction exacte de sailing_physics.py + env)
# ============================================================

# Vecteurs unitaires des 8 directions + Stay
_DIRS = np.array([
    [0,  1],   # 0 Nord
    [1,  1],   # 1 NE
    [1,  0],   # 2 Est
    [1, -1],   # 3 SE
    [0, -1],   # 4 Sud
    [-1,-1],   # 5 SO
    [-1, 0],   # 6 Ouest
    [-1, 1],   # 7 NO
    [0,  0],   # 8 Stay
], dtype=float)

_BOAT_PERF = 0.4
_MAX_SPEED = 8.0
_INERTIA   = 0.3


def _efficiency(boat_dir, wind):
    """Reproduction exacte de calculate_sailing_efficiency."""
    wn = np.linalg.norm(wind)
    bn = np.linalg.norm(boat_dir)
    if wn < 1e-8 or bn < 1e-8:
        return 0.0
    wind_from = -wind / wn
    bd = boat_dir / bn
    angle = float(np.arccos(np.clip(np.dot(wind_from, bd), -1.0, 1.0)))
    if   angle < np.pi / 4:      return 0.05
    elif angle < np.pi / 2:      return 0.5 + 0.5 * (angle - np.pi/4) / (np.pi/4)
    elif angle < 3*np.pi / 4:    return 1.0
    else:
        e = 1.0 - 0.5 * (angle - 3*np.pi/4) / (np.pi/4)
        return max(0.5, e)


def _step_physics(pos, vel, action, wind):
    """
    Simule exactement un step (position + vitesse résultantes).
    Retourne (new_pos, new_vel_disc) comme entiers numpy.
    """
    d = _DIRS[action]
    wn = np.linalg.norm(wind)

    if wn > 1e-8 and (action < 8):
        eff  = _efficiency(d, wind)
        tv   = d * eff * wn * _BOAT_PERF
        spd  = np.linalg.norm(tv)
        if spd > _MAX_SPEED:
            tv = tv / spd * _MAX_SPEED
        nv = tv + _INERTIA * (vel - tv)
        if np.linalg.norm(nv) > _MAX_SPEED:
            nv = nv / np.linalg.norm(nv) * _MAX_SPEED
    else:
        nv = _INERTIA * vel

    # Discrétisation identique à l'env
    nv_disc = np.where(nv < 0, np.ceil(nv), np.floor(nv)).astype(np.int32)
    np_ = np.clip(pos + nv_disc, 0, 127).astype(np.int32)
    return np_, nv_disc


# ============================================================
#  A* WIND-AWARE
# ============================================================

def _heuristic(x, y, gx, gy):
    """Distance Chebyshev / vitesse max atteignable ≈ 5 cases/step."""
    return max(abs(x - gx), abs(y - gy)) / 5.0


def astar(start_pos, start_vel, wind_field, island_map,
          goal=(64, 127), max_nodes=100_000):
    """
    A* dans l'espace (x, y, vx, vy).

    wind_field : (128, 128, 2) — wind_field[y, x] = (wx, wy)
    island_map : (128, 128)    — 1 = île, 0 = eau
    Retourne la liste d'actions (entiers 0-7) ou [] si échec.
    """
    sx, sy = int(start_pos[0]), int(start_pos[1])
    svx, svy = int(start_vel[0]), int(start_vel[1])
    gx, gy = int(goal[0]), int(goal[1])

    start_state = (sx, sy, svx, svy)
    h0 = _heuristic(sx, sy, gx, gy)

    # heap : (f, g, state, path)
    heap = [(h0, 0.0, start_state, [])]
    best_g = {}          # state -> meilleur coût connu
    nodes = 0

    while heap and nodes < max_nodes:
        f, g, state, path = heapq.heappop(heap)
        x, y, vx, vy = state

        if (x, y) == (gx, gy):
            return path

        if state in best_g and best_g[state] <= g:
            continue
        best_g[state] = g
        nodes += 1

        pos = np.array([x, y], dtype=np.int32)
        vel = np.array([vx, vy], dtype=float)
        wind = wind_field[y, x]

        for a in range(8):   # on n'envisage pas Stay en planification
            np_, nv = _step_physics(pos, vel, a, wind)
            nx, ny = int(np_[0]), int(np_[1])

            if island_map[ny, nx] == 1:
                continue

            ns = (nx, ny, int(nv[0]), int(nv[1]))
            ng = g + 1.0

            if ns in best_g and best_g[ns] <= ng:
                continue

            nh = _heuristic(nx, ny, gx, gy)
            heapq.heappush(heap, (ng + nh, ng, ns, path + [a]))

    return []   # échec


# ============================================================
#  AGENT
# ============================================================

class MyAgent(BaseAgent):
    """
    Agent Wind-Aware A* pour le Sailing Challenge.

    Lit le champ de vent complet de l'observation et planifie
    la meilleure trajectoire à chaque épisode (et périodiquement).
    """

    # Replannifie toutes les REPLAN steps (le vent évolue lentement)
    REPLAN_EVERY  = 20
    MAX_NODES     = 100_000

    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()
        self.goal = (64, 127)
        self._plan = []
        self._plan_idx = 0
        self._steps_since_replan = 0

    def reset(self):
        self._plan = []
        self._plan_idx = 0
        self._steps_since_replan = 0

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    # ------------------------------------------------------------------

    @staticmethod
    def _parse(obs):
        pos  = np.array([int(obs[0]), int(obs[1])], dtype=np.int32)
        vel  = np.array([float(obs[2]), float(obs[3])])
        wf   = obs[6 : 6 + 128*128*2].reshape(128, 128, 2)
        imap = obs[6 + 128*128*2 :].reshape(128, 128)
        return pos, vel, wf, imap

    def _replan(self, obs):
        pos, vel, wf, imap = self._parse(obs)
        self._plan = astar(pos, vel, wf, imap,
                           goal=self.goal, max_nodes=self.MAX_NODES)
        self._plan_idx = 0
        self._steps_since_replan = 0

    # ------------------------------------------------------------------
    # Fallback heuristique (si A* échoue ou plan épuisé)
    # ------------------------------------------------------------------

    def _fallback(self, obs):
        x, y   = float(obs[0]), float(obs[1])
        wx, wy = float(obs[4]), float(obs[5])

        # Au-dessus de l'île → ligne droite vers le but
        if y > 87:
            dx = 64 - x
            if abs(dx) < 4:  return 0
            return 1 if dx > 0 else 7

        # Dans/proche de l'île → sortir latéralement
        if 36 <= x <= 92 and y >= 15:
            return 6 if x >= 64 else 2

        # Sous l'île : louvoyage adapté au vent
        if wy > 0.3:          # vent favorable vers le nord
            dx = 64 - x
            if abs(dx) < 5: return 0
            return 1 if dx > 0 else 7
        else:                  # vent défavorable → louvoyage
            if x < 15:   return 1   # NE
            if x > 112:  return 7   # NO
            return 1 if x < 64 else 7

    # ------------------------------------------------------------------

    def act(self, observation: np.ndarray) -> int:
        need = (
            not self._plan or
            self._plan_idx >= len(self._plan) or
            self._steps_since_replan >= self.REPLAN_EVERY
        )
        if need:
            self._replan(observation)

        if self._plan_idx < len(self._plan):
            a = self._plan[self._plan_idx]
            self._plan_idx += 1
            self._steps_since_replan += 1
            return int(a)

        return self._fallback(observation)

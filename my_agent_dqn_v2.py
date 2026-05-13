"""
DQN v2 Agent for the RL Sailing Challenge — inférence numpy pure.

Charge weights_v2.npz produit par train_dqn_v2.py.
Sinon fallback vers l'agent A* (si my_agent_astar.py est présent dans le zip)
ou vers l'heuristique de louvoyage.

Soumission :
    zip my_submission_dqn.zip my_agent_dqn_v2.py weights_v2.npz
"""

import numpy as np
import os

try:
    from evaluator.base_agent import BaseAgent
except ImportError:
    from agents.base_agent import BaseAgent

# ─────────────────────────────────────────────
#  Physique (pour reward shaping des features)
# ─────────────────────────────────────────────
GOAL = np.array([64., 127.])

def _eff(boat_dir, wind):
    wn = np.linalg.norm(wind); bn = np.linalg.norm(boat_dir)
    if wn < 1e-8 or bn < 1e-8: return 0.0
    a = float(np.arccos(np.clip(np.dot(-wind/wn, boat_dir/bn), -1., 1.)))
    if a < np.pi/4:   return 0.05
    if a < np.pi/2:   return 0.5 + 0.5*(a-np.pi/4)/(np.pi/4)
    if a < 3*np.pi/4: return 1.0
    return max(0.5, 1.0-0.5*(a-3*np.pi/4)/(np.pi/4))

def relu(x): return np.maximum(0., x)

# ─────────────────────────────────────────────
#  Features (identiques à train_dqn_v2.py)
# ─────────────────────────────────────────────
def feats(obs):
    x, y   = float(obs[0]), float(obs[1])
    vx, vy = float(obs[2]), float(obs[3])
    wx, wy = float(obs[4]), float(obs[5])
    dx, dy = GOAL[0]-x, GOAL[1]-y
    dist   = np.sqrt(dx*dx+dy*dy)

    gn = dist; wn = np.sqrt(wx*wx+wy*wy)
    ang = 0.5
    if gn>1e-8 and wn>1e-8:
        ang = float(np.arccos(np.clip((dx*wx+dy*wy)/(gn*wn),-1.,1.)))/np.pi

    wf = obs[6:6+128*128*2].reshape(128,128,2)
    sx = dx/gn if gn>1e-8 else 0.; sy = dy/gn if gn>1e-8 else 1.
    wxa,wya,n = 0.,0.,0
    for k in range(5,30,5):
        px=int(np.clip(x+sx*k,0,127)); py=int(np.clip(y+sy*k,0,127))
        wxa+=wf[py,px,0]; wya+=wf[py,px,1]; n+=1
    wxa/=n; wya/=n; wi=np.sqrt(wxa*wxa+wya*wya)

    near = 1. if (28<=x<=100 and 10<=y<=95) else 0.

    return np.array([
        x/128, y/128, vx/8, vy/8, wx/10, wy/10,
        dx/128, dy/128, dist/181, ang,
        wxa/10, wya/10, wi/10, near
    ], dtype=np.float32)

# ─────────────────────────────────────────────
#  Forward numpy (architecture 14→256→256→128→9)
# ─────────────────────────────────────────────
def forward(x, layers):
    """layers = liste de (W, b) tuples."""
    h = x
    for i, (W, b) in enumerate(layers):
        h = W @ h + b
        if i < len(layers)-1:
            h = relu(h)
    return h


class MyAgent(BaseAgent):

    WEIGHTS_FILE = 'weights_v2.npz'

    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()
        self.layers = None
        self._load()

    def _load(self):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, self.WEIGHTS_FILE)
        if not os.path.exists(path):
            print(f"[MyAgent-DQN] {path} non trouvé — fallback heuristique")
            return
        data = np.load(path)
        keys = sorted(data.files)
        # Poids du réseau : W0,b0, W1,b1, W2,b2, W3,b3 (4 couches)
        if len(keys) < 8:
            print(f"[MyAgent-DQN] fichier incomplet ({len(keys)} clés) — fallback")
            return
        self.layers = []
        for i in range(0, len(keys), 2):
            if i+1 >= len(keys): break
            W = data[keys[i]].astype(np.float32)
            b = data[keys[i+1]].astype(np.float32)
            self.layers.append((W, b))
        print(f"[MyAgent-DQN] {len(self.layers)} couches chargées depuis {path}")

    def reset(self): pass
    def seed(self, seed=None): self.np_random = np.random.default_rng(seed)

    # ── Fallback heuristique de louvoyage ──────────────────────────
    def _heuristic(self, obs):
        x, y   = float(obs[0]), float(obs[1])
        wx, wy = float(obs[4]), float(obs[5])
        if y > 87:
            dx = 64-x
            if abs(dx)<4: return 0
            return 1 if dx>0 else 7
        if 36<=x<=92 and y>=15:
            return 6 if x>=64 else 2
        if wy > 0.3:
            dx = 64-x
            if abs(dx)<5: return 0
            return 1 if dx>0 else 7
        if x<15:  return 1
        if x>112: return 7
        return 1 if x<64 else 7

    # ── Politique principale ────────────────────────────────────────
    def act(self, observation: np.ndarray) -> int:
        if self.layers is None:
            return self._heuristic(observation)
        f = feats(observation)
        qv = forward(f, self.layers)
        return int(np.argmax(qv))

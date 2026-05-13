"""
DQN amélioré pour le Sailing Challenge — compatible SSPCloud CPU.

Améliorations clés vs v1 :
  1. Features enrichies (14 dim) incluant le vent moyen DEVANT le bateau
  2. Reward shaping potentiel-based rigoureux (Ng et al. 1999)
  3. Curriculum learning (training_3 facile d'abord)
  4. Huber loss (plus stable que MSE)
  5. Architecture plus profonde (256-256-128)
  6. Export numpy complet pour soumission sans torch

Usage :
    python train_dqn_v2.py                          # 4000 épisodes
    python train_dqn_v2.py --episodes 6000 --hidden 256
"""

import numpy as np, torch, torch.nn as nn, torch.optim as optim
from collections import deque
import random, sys, os, argparse, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from env_sailing import SailingEnv
from wind_scenarios import get_wind_scenario

# ─────────────────────────────────────────────
#  Physique (pour reward shaping)
# ─────────────────────────────────────────────
def _eff(boat_dir, wind):
    wn = np.linalg.norm(wind); bn = np.linalg.norm(boat_dir)
    if wn < 1e-8 or bn < 1e-8: return 0.0
    a = float(np.arccos(np.clip(np.dot(-wind/wn, boat_dir/bn), -1., 1.)))
    if a < np.pi/4:   return 0.05
    if a < np.pi/2:   return 0.5 + 0.5*(a-np.pi/4)/(np.pi/4)
    if a < 3*np.pi/4: return 1.0
    return max(0.5, 1.0-0.5*(a-3*np.pi/4)/(np.pi/4))

GOAL  = np.array([64., 127.])
GAMMA = 0.995

def phi(obs):
    """Potentiel Φ(s) = progression_vers_but + efficacité_voile."""
    x, y   = float(obs[0]), float(obs[1])
    wx, wy = float(obs[4]), float(obs[5])
    eff = _eff(GOAL - np.array([x,y]), np.array([wx,wy]))
    return 2.0*(y/127.0) + 1.0*eff

def shaped(obs, nobs, r):
    return r + GAMMA*phi(nobs) - phi(obs)

# ─────────────────────────────────────────────
#  Features (14 dim)
# ─────────────────────────────────────────────
FDIM = 14

def feats(obs):
    x, y   = float(obs[0]), float(obs[1])
    vx, vy = float(obs[2]), float(obs[3])
    wx, wy = float(obs[4]), float(obs[5])
    dx, dy = GOAL[0]-x, GOAL[1]-y
    dist   = np.sqrt(dx*dx + dy*dy)

    # Angle vent/but
    gn = np.sqrt(dx*dx+dy*dy)
    wn = np.sqrt(wx*wx+wy*wy)
    ang = 0.5
    if gn>1e-8 and wn>1e-8:
        ang = float(np.arccos(np.clip(
            (dx*wx+dy*wy)/(gn*wn), -1., 1.)))/np.pi

    # Vent moyen devant (5 points échantillonnés)
    wf = obs[6:6+128*128*2].reshape(128,128,2)
    sx = dx/gn if gn>1e-8 else 0.; sy = dy/gn if gn>1e-8 else 1.
    wxa, wya, n = 0., 0., 0
    for k in range(5, 30, 5):
        px=int(np.clip(x+sx*k,0,127)); py=int(np.clip(y+sy*k,0,127))
        wxa+=wf[py,px,0]; wya+=wf[py,px,1]; n+=1
    wxa/=n; wya/=n
    wi = np.sqrt(wxa*wxa+wya*wya)

    near = 1. if (28<=x<=100 and 10<=y<=95) else 0.

    return np.array([
        x/128, y/128, vx/8, vy/8, wx/10, wy/10,
        dx/128, dy/128, dist/181, ang,
        wxa/10, wya/10, wi/10, near
    ], dtype=np.float32)

# ─────────────────────────────────────────────
#  Réseau
# ─────────────────────────────────────────────
class QNet(nn.Module):
    def __init__(self, d=FDIM, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,h), nn.ReLU(),
            nn.Linear(h,h), nn.ReLU(),
            nn.Linear(h,h//2), nn.ReLU(),
            nn.Linear(h//2,9))
    def forward(self,x): return self.net(x)

# ─────────────────────────────────────────────
#  Buffer
# ─────────────────────────────────────────────
class Buf:
    def __init__(self,cap): self.b=deque(maxlen=cap)
    def push(self,*x): self.b.append(x)
    def sample(self,n):
        B=random.sample(self.b,n); s,a,r,s2,d=zip(*B)
        return (np.array(s,np.float32),np.array(a,np.int64),
                np.array(r,np.float32),np.array(s2,np.float32),
                np.array(d,np.float32))
    def __len__(self): return len(self.b)

# ─────────────────────────────────────────────
#  Curriculum
# ─────────────────────────────────────────────
def scenario(ep, total):
    f = ep/total
    if f<0.30:  return 'training_3'
    if f<0.60:  return ['training_3','training_1'][ep%2]
    return ['training_1','training_2','training_3'][ep%3]

# ─────────────────────────────────────────────
#  Entraînement
# ─────────────────────────────────────────────
def train(cfg):
    dev = torch.device('cpu')
    print(f"Device: {dev} | Config: {cfg}\n")

    q  = QNet(FDIM, cfg['h']).to(dev)
    qt = QNet(FDIM, cfg['h']).to(dev)
    qt.load_state_dict(q.state_dict()); qt.eval()
    opt = optim.Adam(q.parameters(), lr=cfg['lr'])
    buf = Buf(cfg['cap'])

    eps = 1.0
    best_sr = 0.
    sw = deque(maxlen=200); rw = deque(maxlen=200)
    t0 = time.time()

    for ep in range(cfg['n']):
        sc = scenario(ep, cfg['n'])
        env = SailingEnv(**get_wind_scenario(sc))
        obs,_ = env.reset(seed=ep)
        s = feats(obs); ok=False; er=0.

        for step in range(500):
            a = random.randint(0,8) if random.random()<eps else \
                int(q(torch.FloatTensor(s).unsqueeze(0)).argmax())

            nobs,r,term,trunc,_ = env.step(a)
            done = term or trunc
            if r>0: ok=True
            rs = shaped(obs, nobs, r)
            s2 = feats(nobs)
            buf.push(s,a,rs,s2,float(done))
            obs,s = nobs,s2; er+=r

            if len(buf)>=cfg['bs']:
                sb,ab,rb,s2b,db = buf.sample(cfg['bs'])
                sb  = torch.FloatTensor(sb).to(dev)
                ab  = torch.LongTensor(ab).to(dev)
                rb  = torch.FloatTensor(rb).to(dev)
                s2b = torch.FloatTensor(s2b).to(dev)
                db  = torch.FloatTensor(db).to(dev)
                qv  = q(sb).gather(1,ab.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    nq = qt(s2b).gather(1,q(s2b).argmax(1).unsqueeze(1)).squeeze(1)
                    tgt= rb + GAMMA*nq*(1-db)
                loss = nn.SmoothL1Loss()(qv,tgt)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(),1.0); opt.step()
            if done: break

        if ep % cfg['tu']==0: qt.load_state_dict(q.state_dict())
        eps = max(0.05, eps*cfg['ed'])
        sw.append(int(ok)); rw.append(er)
        sr = float(np.mean(sw))

        tag=''
        if sr>best_sr and len(sw)>=50:
            best_sr=sr; _save(q,cfg['out']); tag=f'  ← BEST {sr:.2%}'

        if ep%200==0 or ep==cfg['n']-1:
            print(f"Ep {ep:5d} | {sc} | "
                  f"sr={sr:.2%} | score={np.mean(rw):.2f} | "
                  f"ε={eps:.3f} | {(time.time()-t0)/60:.1f}min{tag}")

    _save(q, cfg['out'])
    print(f"\nFin. Meilleur taux de succès: {best_sr:.2%}")
    return q

def _save(m, path):
    d={}
    for i,(n,p) in enumerate(m.state_dict().items()):
        d[f'p{i:02d}_{n.replace(".","_")}']=p.detach().cpu().numpy()
    np.savez(path,**d)
    print(f"  Poids exportés → {path}")

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--episodes',type=int,  default=4000)
    ap.add_argument('--hidden',  type=int,  default=256)
    ap.add_argument('--lr',      type=float,default=5e-4)
    ap.add_argument('--save',    type=str,  default='weights_v2.npz')
    a=ap.parse_args()
    train(dict(n=a.episodes,h=a.hidden,lr=a.lr,
               ed=0.9975,bs=256,cap=200_000,tu=100,out=a.save))

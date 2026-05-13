"""
Évaluation locale et comparaison des agents.

Usage :
    python evaluate_agents.py                          # compare tous les agents
    python evaluate_agents.py --agent my_agent_astar.py --seeds 50
    python evaluate_agents.py --scenario training_3

Métriques identiques à Codabench :
    score_i = 100 * 0.995^steps_i  si succès, 0 sinon
    score_final = moyenne sur N seeds
"""

import sys, os, argparse, importlib.util
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from env_sailing import SailingEnv
from wind_scenarios import get_wind_scenario, WIND_SCENARIOS


def load_agent_class(path):
    """Charge la classe MyAgent depuis un fichier .py."""
    here = os.path.dirname(os.path.abspath(path))
    if here not in sys.path:
        sys.path.insert(0, here)
    spec   = importlib.util.spec_from_file_location("agent_mod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


def evaluate(AgentClass, scenario_name, n_seeds=20, verbose=False):
    sc  = get_wind_scenario(scenario_name)
    env = SailingEnv(**sc)
    agent = AgentClass()

    scores, successes, all_steps = [], [], []

    for seed in range(n_seeds):
        obs, _ = env.reset(seed=seed)
        agent.reset()
        agent.seed(seed)

        last_step = 499
        success   = False

        for step in range(500):
            action = agent.act(obs)
            obs, reward, term, trunc, _ = env.step(action)
            if reward > 0:
                success   = True
                last_step = step
            if term or trunc:
                break

        codabench_score = 100.0 * (0.995 ** last_step) if success else 0.0
        scores.append(codabench_score)
        successes.append(int(success))
        if success:
            all_steps.append(last_step)
        if verbose:
            print(f"  seed {seed:3d}: {'✓' if success else '✗'}  "
                  f"score={codabench_score:.3f}  steps={last_step}")

    mean_score = float(np.mean(scores))
    mean_sr    = float(np.mean(successes))
    mean_steps = float(np.mean(all_steps)) if all_steps else float('nan')

    print(f"  {scenario_name:12s} | score={mean_score:7.3f} | "
          f"sr={mean_sr:.2%} ({sum(successes)}/{n_seeds}) | "
          f"steps={mean_steps:.1f}")
    return mean_score, mean_sr


def compare_agents(agent_paths, scenarios, n_seeds):
    print(f"\n{'='*65}")
    print(f"{'Agent':<30} {'Scénario':<14} {'Score':>7} {'Succès':>8} {'Steps':>7}")
    print(f"{'='*65}")

    for path in agent_paths:
        if not os.path.exists(path):
            print(f"  [MANQUANT] {path}")
            continue
        name = os.path.basename(path)
        try:
            Cls = load_agent_class(path)
        except Exception as e:
            print(f"  [ERREUR chargement {name}] {e}")
            continue

        all_sc_scores = []
        for sc in scenarios:
            sc_obj = get_wind_scenario(sc)
            env    = SailingEnv(**sc_obj)
            scores, successes, steps_list = [], [], []
            for seed in range(n_seeds):
                obs,_ = env.reset(seed=seed)
                agent = Cls(); agent.reset(); agent.seed(seed)
                last_step=499; success=False
                for step in range(500):
                    a = agent.act(obs)
                    obs,r,term,trunc,_ = env.step(a)
                    if r>0: success=True; last_step=step
                    if term or trunc: break
                sc_score = 100.*(0.995**last_step) if success else 0.
                scores.append(sc_score); successes.append(int(success))
                if success: steps_list.append(last_step)
            ms = float(np.mean(scores)); sr = float(np.mean(successes))
            mst = float(np.mean(steps_list)) if steps_list else float('nan')
            all_sc_scores.append(ms)
            print(f"  {name:<28} {sc:<14} {ms:>7.3f} {sr:>7.2%} {mst:>7.1f}")
        if len(scenarios)>1:
            print(f"  {name:<28} {'ALL (avg)':<14} {np.mean(all_sc_scores):>7.3f}")
        print()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent',    type=str, default=None,
                    help='Chemin vers un agent spécifique (sinon compare tout)')
    ap.add_argument('--seeds',    type=int, default=20)
    ap.add_argument('--scenario', type=str, default='all',
                    help=f"Un parmi {list(WIND_SCENARIOS.keys())} ou 'all'")
    ap.add_argument('--verbose',  action='store_true')
    args = ap.parse_args()

    scenarios = list(WIND_SCENARIOS.keys()) if args.scenario=='all' \
                else [args.scenario]

    if args.agent:
        Cls = load_agent_class(args.agent)
        print(f"\nÉvaluation de {args.agent}")
        for sc in scenarios:
            evaluate(Cls, sc, args.seeds, args.verbose)
    else:
        # Compare tous les agents du dossier courant
        candidates = [
            'my_agent_astar.py',
            'my_agent_dqn_v2.py',
            # ajoute d'autres ici si besoin
        ]
        compare_agents(candidates, scenarios, args.seeds)

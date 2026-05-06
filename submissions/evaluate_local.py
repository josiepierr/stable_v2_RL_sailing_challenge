"""
Script d'évaluation locale avant soumission Codabench.
A exécuter depuis la racine du repo.

Usage:
    python evaluate_local.py
    python evaluate_local.py --seeds 50 --scenario training_2
"""
import sys
import os
import argparse
import numpy as np
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from env_sailing import SailingEnv
from wind_scenarios import get_wind_scenario, WIND_SCENARIOS


def load_agent(agent_path):
    spec = importlib.util.spec_from_file_location("agent_module", agent_path)
    module = importlib.util.module_from_spec(spec)
    # Ajoute le dossier de l'agent au path pour trouver weights.npz
    sys.path.insert(0, os.path.dirname(os.path.abspath(agent_path)))
    spec.loader.exec_module(module)
    return module.MyAgent


def evaluate_agent(AgentClass, scenario_name, n_seeds=20, verbose=False):
    scenario = get_wind_scenario(scenario_name)
    env = SailingEnv(**scenario)
    agent = AgentClass()

    scores, successes, steps_list = [], [], []

    for seed in range(n_seeds):
        obs, _ = env.reset(seed=seed)
        agent.reset()
        agent.seed(seed)

        total_reward = 0.0
        last_step = 499
        success = False

        for step in range(500):
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if reward > 0:
                success = True
                last_step = step
            if terminated or truncated:
                break

        # Score Codabench : reward_sum * 0.995^steps
        score = total_reward * (0.995 ** last_step) if success else 0.0
        scores.append(score)
        successes.append(1 if success else 0)
        if success:
            steps_list.append(last_step)

        if verbose:
            print(f"  Seed {seed:3d}: {'✓' if success else '✗'} "
                  f"score={score:.3f} steps={last_step}")

    print(f"\n{'='*50}")
    print(f"Scénario : {scenario_name} | {n_seeds} seeds")
    print(f"{'='*50}")
    print(f"Score moyen        : {np.mean(scores):.4f}")
    print(f"Taux de succès     : {np.mean(successes):.2%} ({sum(successes)}/{n_seeds})")
    if steps_list:
        print(f"Étapes (succès)    : {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}")
    print(f"{'='*50}")

    return np.mean(scores), np.mean(successes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='my_agent.py')
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--scenario', type=str, default=None,
                        help=f"Un de: {list(WIND_SCENARIOS.keys())} ou 'all'")
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    print(f"Chargement de l'agent depuis : {args.agent}")
    AgentClass = load_agent(args.agent)

    scenarios = list(WIND_SCENARIOS.keys()) if args.scenario in (None, 'all') \
                else [args.scenario]

    all_scores = []
    for sc in scenarios:
        score, sr = evaluate_agent(AgentClass, sc, args.seeds, args.verbose)
        all_scores.append(score)

    if len(scenarios) > 1:
        print(f"\n>>> Score moyen toutes scénarios : {np.mean(all_scores):.4f}")


if __name__ == '__main__':
    main()

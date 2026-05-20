
# RL Sailing Challenge — Physics-Aware Waypoint Planner

This repository contains our work for the **RL Sailing Challenge v2**.  
The objective is to control a sailboat on a `128 x 128` grid, starting from the bottom center and reaching the top center while dealing with wind dynamics and avoiding a central island obstacle.

Our final Codabench submission achieved:

| Metric | Hidden Codabench score |
|---|---:|
| Score | **72.93** |
| Average steps | **63.44** |
| Success rate | **1.00** |

---

## 1. Challenge Summary

The environment is a sailing navigation task with:

- grid size: `128 x 128`;
- start position: `(64, 0)`;
- goal position: `(64, 127)`;
- central island obstacle;
- maximum episode length: `500`;
- reward: `100` only when the goal is reached;
- discount factor: `0.995`;
- 9 discrete actions: `N, NE, E, SE, S, SW, W, NW, Stay`.

The final evaluation is performed on a **hidden wind scenario** over 50 seeds.  
Therefore, the main difficulty is not only to perform well on the three public wind scenarios, but also to **generalize to an unseen wind field**.

---

## 2. Repository Structure

```text
stable_v2_RL_sailing_challenge/
│
├── notebooks/
│   ├── 01_env_inspection_CORRECT.ipynb
│   ├── 02_physics_aware_planner_CORRECT.ipynb
│   ├── 03_waypoint_strategy_CORRECT.ipynb
│   ├── 04_parameter_tuning_CORRECT.ipynb
│   ├── 05_compare_baselines.ipynb
│   ├── 06_visualize_failures.ipynb
│   ├── 07_build_submission.ipynb
│   └── original challenge notebooks
│
├── src/
│   ├── agents/
│   │   ├── my_agent.py
│   │   ├── my_agent_tuned.py
│   │   ├── agent_super_naive.py
│   │   ├── agent_trained_example.py
│   │   └── tuning variants
│   │
│   ├── env_sailing.py
│   ├── sailing_physics.py
│   ├── evaluate_submission.py
│   ├── test_agent_validity.py
│   ├── wind_scenarios/
│   └── utility files
│
├── results/
│   ├── parameter_tuning_results.csv
│   ├── parameter_tuning_final_results.csv
│   ├── parameter_tuning_raw_outputs.json
│   ├── parameter_tuning_final_raw_outputs.json
│   └── submission_report.json
│
├── submissions/
│   ├── my_agent.py
│   └── my_submission.zip
│
└── requirements.txt
````

---

## 3. Final Agent

The final submitted agent is located in:

```text
src/agents/my_agent.py
```

The submitted ZIP contains:

```text
my_agent.py
```

at the root of the archive, as required by Codabench.

The agent class is:

```python
class MyAgent(BaseAgent):
    def act(self, observation):
        ...
    
    def reset(self):
        ...
    
    def seed(self, seed=None):
        ...
```

The submission import is compatible with the evaluation bundle:

```python
from evaluator.base_agent import BaseAgent
```

with a local fallback for development.

---

## 4. Methodology

Our final approach is a **physics-aware planning agent** rather than a pure tabular RL or black-box deep RL policy.

The agent uses:

1. **Full observation parsing**

   * current position;
   * current velocity;
   * local wind;
   * full wind field;
   * full world map.

2. **Short-horizon internal simulation**

   * the agent simulates candidate action sequences using an approximation of the sailing dynamics;
   * the planning horizon used in the final version is `2`.

3. **Dynamic waypoint selection**

   * instead of committing permanently to a left or right route around the island, the final agent evaluates several candidate waypoint targets dynamically;
   * this was introduced to improve robustness on the hidden wind scenario.

4. **Heuristic trajectory scoring**
   Candidate simulated trajectories are scored using:

   * northward progress;
   * distance to the current waypoint;
   * distance to the final goal;
   * northward velocity;
   * collision penalty;
   * border penalty;
   * central island-zone penalty;
   * penalty for staying still;
   * penalty for moving south.

5. **Offline parameter tuning**

   * the hyperparameters of the heuristic planner were tuned offline on the public scenarios;
   * no learning is performed during Codabench evaluation.

---

## 5. Final Agent Parameters

The final submitted agent uses the following main parameters:

```python
horizon = 2
progress_weight = 4.0
waypoint_weight = 1.6
north_speed_weight = 8.0
goal_weight = 0.35
collision_penalty = 500000.0
border_penalty = 2000.0
center_penalty = 30.0
wp_radius = 14.0
side_margin = 0.05
stay_penalty = 200.0
south_penalty = 300.0
```

These parameters were chosen to balance:

* speed on public wind scenarios;
* robustness on hidden wind scenarios;
* collision avoidance;
* generalization beyond the three training wind fields.

---

## 6. Evolution of the Approach

We tested several versions of the agent.

### Initial rigid waypoint strategy

The first strong version used a fixed left/right route chosen at the beginning of the episode.

It performed well locally:

| Metric        | Public scenarios |
| ------------- | ---------------: |
| Score         |    around `77.6` |
| Average steps |    around `51.8` |
| Success rate  |            `1.0` |

However, it generalized poorly to the hidden Codabench scenario:

| Metric        | Hidden Codabench |
| ------------- | ---------------: |
| Score         |           `49.1` |
| Average steps |         `195.06` |
| Success rate  |           `0.72` |

This indicated overfitting to the public wind scenarios.

### Dynamic waypoint strategy

We then replaced the fixed left/right commitment with dynamic candidate waypoint selection.

This reduced local speed slightly, but greatly improved robustness:

| Metric        | Hidden Codabench |
| ------------- | ---------------: |
| Score         |        **72.93** |
| Average steps |        **63.44** |
| Success rate  |         **1.00** |

The key improvement was moving from a brittle public-scenario strategy to a more robust planner.

---

## 7. Local Evaluation

The final agent was validated with:

```bash
cd src
python test_agent_validity.py agents/my_agent.py
```

The validation succeeded.

A local evaluation on the public scenarios gave approximately:

| Scenario   | Success rate |       Reward |     Steps |
| ---------- | -----------: | -----------: | --------: |
| training_1 |         100% | around 72–73 | around 64 |
| training_2 |         100% |    around 80 | around 45 |
| training_3 |         100% | around 71–72 | around 68 |

Overall local evaluation:

| Metric       | Public scenarios |
| ------------ | ---------------: |
| Reward       |   around `74–75` |
| Steps        |   around `59–60` |
| Success rate |            `1.0` |

---

## 8. Notebooks

### `01_env_inspection_CORRECT.ipynb`

Inspects the environment, observation structure, action space, wind fields, world map, island obstacle and rendering.

### `02_physics_aware_planner_CORRECT.ipynb`

Builds the first physics-aware planner and tests short-horizon action simulation.

### `03_waypoint_strategy_CORRECT.ipynb`

Introduces waypoint-based navigation around the island and compares left/right routes.

### `04_parameter_tuning_CORRECT.ipynb`

Generates tunable versions of `MyAgent`, validates them with the official script, evaluates them locally and exports the selected configuration to:

```text
src/agents/my_agent.py
```

### `05_compare_baselines.ipynb`

Compares our agent against provided baselines, including:

* `agent_super_naive.py`;
* `agent_trained_example.py`;
* `my_agent.py`.

### `06_visualize_failures.ipynb`

Provides trajectory diagnostics, failure visualization and heatmaps.
This notebook is useful for understanding slow trajectories, collisions and route instability.

### `07_build_submission.ipynb`

Builds the final Codabench ZIP submission and verifies that:

```text
my_agent.py
```

is located at the root of the ZIP archive.

---

## 9. Baselines

The repository includes baseline agents:

```text
src/agents/agent_super_naive.py
src/agents/agent_trained_example.py
```

Our final agent significantly improves over these baselines in both success rate and number of steps.

---

## 10. Submission

To create the final ZIP submission:

```bash
cd submissions
zip my_submission.zip my_agent.py
```

The ZIP structure must be:

```text
my_submission.zip
└── my_agent.py
```

It must not contain:

```text
some_folder/my_agent.py
```

The submitted file is:

```text
submissions/my_submission.zip
```

---

## 11. Reproducibility

To reproduce the development workflow:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Inspect the environment:

```text
notebooks/01_env_inspection_CORRECT.ipynb
```

3. Review the planner and waypoint logic:

```text
notebooks/02_physics_aware_planner_CORRECT.ipynb
notebooks/03_waypoint_strategy_CORRECT.ipynb
```

4. Regenerate and evaluate the final agent:

```text
notebooks/04_parameter_tuning_CORRECT.ipynb
```

5. Build the submission:

```text
notebooks/07_build_submission.ipynb
```

---

## 12. Code Provenance

The final solution is an original implementation developed for this challenge.

The final agent is not a neural-network weight file and does not rely on uninterpretable learned parameters.
It is a deterministic physics-aware planner with offline-tuned heuristic parameters.

Generative AI was used as a coding assistant to help structure notebooks, debug errors, and refine the implementation.
All code was adapted, executed, validated, and tuned within the challenge repository before submission.

---

## 13. Important Notes

* No learning is performed during Codabench evaluation.
* The final policy is fixed at submission time.
* The agent uses only `numpy`.
* The submitted file passes `test_agent_validity.py`.
* The final hidden Codabench performance is:

|     Score | Average Steps | Success Rate |
| --------: | ------------: | -----------: |
| **72.93** |     **63.44** |     **1.00** |

```


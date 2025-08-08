# HeMAC - The Heterogeneous Multi-Agent Challenge

**HeMAC** is a standardized, PettingZoo-based benchmark environment for Heterogeneous Multi-Agent Reinforcement Learning (HeMARL). It proposes multiple scenarios where agents with diverse sensors, resources, or capabilities must cooperate to solve complex tasks under partial observability.

---

## Key Features

- **Rich Heterogeneity:** Multiple distinct agent types (Quadcopters, Observers, Provisioners) with unique observation and action spaces, capabilities, and roles.
- **Multi-Stage Benchmarking:** Three challenges (“Simple Fleet”, “Fleet”, “Complex Fleet”) with increasing difficulty and heterogeneity.
- **Scenario Variety:** Each challenge contains several scenarios for fine control over agent compositions and environmental complexity.
- **Partial Observability:** Agents perceive the world through unique, limited sensors and information, increasing coordination complexity.
- **Flexible Spaces:** Both discrete and continuous action spaces are supported for all agent types.
- **Extensibility:** Easily add new agent types, capabilities, and scenarios.

---

## Why HeMAC?

Traditional MARL benchmarks focus on homogeneous teams, falling short when representing real-world, heterogeneous agent systems. HeMAC provides:

- A controlled environment where agents must specialize and cooperate based on their unique abilities.
- Standardized tasks to facilitate reproducible, comparable HeMARL research.
- Rich partial observability and coordination challenges.

Our latest paper shows that while state-of-the-art methods (like MAPPO) excel at simpler tasks, their performance degrades with increased heterogeneity—with simpler algorithms (like IPPO) sometimes outperforming them under these conditions.

---

## Environment Overview

In **HeMAC**, a team of autonomous agents works together to find and reach moving targets in a randomly generated map featuring obstacles and special structures.

### Available Agents

- **Quadcopter:** Low-altitude, agile agents that can reach targets but have limited energy and capacity.
- **Observer:** High-altitude, fast agents with broad forward-facing views; guide Quadcopters but cannot directly reach targets.
- **Provisioner:** Ground vehicles navigating a road network to recharge/support aerial agents and assist with target retrieval.

### Challenges and Scenarios

| Name          | Agents                                | Description                                                                                     | Sample Scenarios    |
|---------------|---------------------------------------|-------------------------------------------------------------------------------------------------|---------------------|
| **Simple Fleet**  | Quadcopters, Observers                | Reach as many moving targets as possible. Observers must guide Quadcopters.                      | 1q1o, 3q1o, 5q2o    |
| **Fleet**         | Quadcopters, Observers                | Multi-target, energy constraints, obstacles, limited communication range.                        | 3q1o, 10q3o, 20q5o  |
| **Complex Fleet** | Quadcopters, Observers, Provisioners  | High heterogeneity: energy/capacity limits, provisioners restricted to roads, complex cooperation.| 3q1o1p, 5q2o1p, etc.|

Agents receive different local observations according to their sensors and roles.

---

## Installation

To install HeMAC in a fresh Python environment, python3.11 is recommended. Then run:

```bash
pip install .
```

---

## Getting Started

Example of usage (with the PettingZoo's AEC API):

```python
from hemarl.hemac import HeMAC_v0

env = HeMAC_v0.env(render_mode="human")
env.reset(seed=0)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()
    env.step(action)
env.close()
```

---

## Citation

If you use HeMAC in your research, please cite our latest paper:

> (Citation pending)

---

## Contributing

Contributions are welcome! Please open an issue or pull request for new agents, scenarios, bug fixes, or suggestions. see our contributing guide.
version 1.0.0 test coverage: 72.12 %

---

**Note:** HeMAC is under active development. Feedback is highly appreciated to help shape this benchmark for the community.

---

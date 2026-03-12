import numpy as np
from pettingzoo import AECEnv
from gymnasium.spaces import Discrete, Box


class SimpleFleetEnv(AECEnv):

    metadata = {"name": "simple_fleet"}

    def __init__(self, grid_size=10, max_steps=200):

        self.grid_size = grid_size
        self.max_steps = max_steps

        self.agents = ["quad_0", "quad_1", "obs_0"]

        self.agent_types = {
            "quad_0": "quad",
            "quad_1": "quad",
            "obs_0": "observer",
        }

        self.action_spaces = {
            agent: Discrete(5) for agent in self.agents
        }

        self.observation_spaces = {
            agent: Box(low=0, high=grid_size, shape=(3,), dtype=np.float32)
            for agent in self.agents
        }

    def reset(self, seed=None):

        self.steps = 0

        self.agent_positions = {
            agent: np.random.randint(0, self.grid_size, size=2)
            for agent in self.agents
        }

        self.target = np.random.randint(0, self.grid_size, size=2)

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}

    def observe(self, agent):

        pos = self.agent_positions[agent]

        obs = np.array([
            pos[0],
            pos[1],
            np.linalg.norm(pos - self.target)
        ], dtype=np.float32)

        return obs

    def step(self, action):

        agent = self.agent_selection

        if self.terminations[agent]:
            return

        move = {
            0: np.array([0, 0]),
            1: np.array([1, 0]),
            2: np.array([-1, 0]),
            3: np.array([0, 1]),
            4: np.array([0, -1]),
        }

        self.agent_positions[agent] += move[action]
        self.agent_positions[agent] = np.clip(
            self.agent_positions[agent], 0, self.grid_size - 1
        )

        if self.agent_types[agent] == "quad":

            if np.array_equal(self.agent_positions[agent], self.target):

                for a in self.agents:
                    self.rewards[a] = 1
                    self.terminations[a] = True

        self.steps += 1

        if self.steps >= self.max_steps:

            for a in self.agents:
                self.truncations[a] = True
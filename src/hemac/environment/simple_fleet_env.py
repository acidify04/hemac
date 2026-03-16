import os
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium.spaces import Discrete, Box

try:
    import pygame
except ImportError:
    pygame = None


class SimpleFleetEnv(AECEnv):

    metadata = {
        "name": "simple_fleet",
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(self, grid_size=10, max_steps=200, render_mode=None, render_fps=8):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.render_fps = render_fps  # 프레임 속도를 인자로 받아 조절할 수 있도록 수정 (기본값 2)

        self.possible_agents = ["quad_0", "quad_1", "obs_0"]
        self.agents = self.possible_agents[:]
    
        self.agent_types = {
            "quad_0": "quad",
            "quad_1": "quad",
            "obs_0": "observer",
        }

        self.action_spaces = {
            agent: Discrete(5) for agent in self.possible_agents
        }

        self.observation_spaces = {
            agent: Box(low=0, high=grid_size, shape=(3,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = None

        # 렌더링 관련 설정
        self.window_size = 600
        self.cell_size = self.window_size // self.grid_size
        self.screen = None
        
        if self.render_mode == "human":
            self.clock = pygame.time.Clock() if pygame else None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.steps = 0
        self.agents = self.possible_agents[:]

        self.agent_positions = {
            agent: np.random.randint(0, self.grid_size, size=2)
            for agent in self.agents
        }

        self.target = np.random.randint(0, self.grid_size, size=2)

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        if self.render_mode == "human":
            self.render()

    def observe(self, agent):
        pos = self.agent_positions[agent]

        obs = np.array([
            pos[0],
            pos[1],
            np.linalg.norm(pos - self.target)
        ], dtype=np.float32)

        return obs

    def step(self, action):
        
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        move = {
            0: np.array([0, 0]),
            1: np.array([1, 0]),
            2: np.array([-1, 0]),
            3: np.array([0, 1]),
            4: np.array([0, -1]),
        }

        if action is not None:
            self.agent_positions[agent] += move[action]
            self.agent_positions[agent] = np.clip(
                self.agent_positions[agent], 0, self.grid_size - 1
            )

        if self.agent_types[agent] == "quad":
            if np.array_equal(self.agent_positions[agent], self.target):
                for a in self.agents:
                    self.rewards[a] = 1
                    self.terminations[a] = True

        if self._agent_selector.is_last():
            self.steps += 1
            if self.steps >= self.max_steps:
                for a in self.agents:
                    self.truncations[a] = True

        self._accumulate_rewards()
        self.agent_selection = self._agent_selector.next()

    def draw(self):
        """실제 화면에 요소들을 그리는 로직"""
        pygame.event.pump()
        
        # 배경 (흰색)
        self.screen.fill((255, 255, 255))

        # 그리드(격자) 그리기
        for x in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.window_size, y))

        # 타겟 그리기 (빨간색)
        tx, ty = self.target
        target_rect = pygame.Rect(
            tx * self.cell_size, ty * self.cell_size, self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, (255, 0, 0), target_rect)

        # 에이전트 색상 매핑
        colors = {
            "quad_0": (0, 0, 255),    # 파란색
            "quad_1": (0, 191, 255),  # 하늘색
            "obs_0": (0, 255, 0)      # 초록색
        }

        # 에이전트 그리기
        for agent in self.agents:
            ax, ay = self.agent_positions[agent]
            agent_rect = pygame.Rect(
                ax * self.cell_size, ay * self.cell_size, self.cell_size, self.cell_size
            )
            pygame.draw.rect(self.screen, colors[agent], agent_rect)

    def render(self):
        """렌더링 메인 함수 (HeMAC 스타일 적용)"""
        if self.render_mode is None:
            return

        if pygame is None:
            print("Pygame이 설치되어 있지 않습니다.")
            return

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.window_size, self.window_size))
                pygame.display.set_caption("Simple Fleet Environment")
            elif self.render_mode == "rgb_array":
                # rgb_array 모드일 때는 화면을 띄우지 않고 Surface 객체만 생성
                self.screen = pygame.Surface((self.window_size, self.window_size))

        self.draw()

        # 화면 픽셀을 numpy array로 변환
        state = np.array(pygame.surfarray.pixels3d(self.screen))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.render_fps)
            
        # rgb_array 모드일 경우 HeMAC과 동일하게 축을 변환하여 반환
        return np.transpose(state, axes=(1, 0, 2)) if self.render_mode == "rgb_array" else None

    def close(self):
        """환경 종료 및 Pygame 자원 해제"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
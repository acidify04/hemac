import subprocess
import numpy as np
from src.hemac.environment.simple_fleet_env import SimpleFleetEnv
import os
import time

try:
    import pygame
except ImportError:
    pygame = None

# -----------------------------
# 좌표 → grid
# -----------------------------

def loc_to_grid(loc):
    # loc 형태: "loc_X_Y"
    parts = loc.split("_")
    return int(parts[1]), int(parts[2])


# -----------------------------
# grid → discrete action (0~4)
# -----------------------------

def grid_to_discrete_action(current_x, current_y, target_loc):
    tx, ty = loc_to_grid(target_loc)
    
    dx = tx - current_x
    dy = ty - current_y

    if dx > 0: return 1
    elif dx < 0: return 2
    elif dy > 0: return 3
    elif dy < 0: return 4
    else: return 0


# -----------------------------
# PDDL problem 생성
# -----------------------------

def generate_problem(quad0_pos, obs_pos, target, grid):

    with open("./pddl/problem.pddl", "w") as f:

        f.write("(define (problem quad_problem)\n")
        f.write("(:domain fleet_domain)\n")

        # objects
        f.write("(:objects\n")
        f.write("quad_0 - quad\n")
        f.write("obs_0 - observer\n")

        for i in range(grid):
            for j in range(grid):
                f.write(f"loc_{i}_{j} ")

        f.write("- location\n")
        f.write(")\n")

        # init
        f.write("(:init\n")

        x, y = int(quad0_pos[0]), int(quad0_pos[1])
        f.write(f"(at_quad quad_0 loc_{x}_{y})\n")

        x, y = int(obs_pos[0]), int(obs_pos[1])
        f.write(f"(at_obs obs_0 loc_{x}_{y})\n")

        tx, ty = int(target[0]), int(target[1])
        f.write(f"(target loc_{tx}_{ty})\n")

        # grid adjacency
        for i in range(grid):
            for j in range(grid):

                if i > 0:
                    f.write(f"(adjacent loc_{i}_{j} loc_{i-1}_{j})\n")
                if i < grid - 1:
                    f.write(f"(adjacent loc_{i}_{j} loc_{i+1}_{j})\n")
                if j > 0:
                    f.write(f"(adjacent loc_{i}_{j} loc_{i}_{j-1})\n")
                if j < grid - 1:
                    f.write(f"(adjacent loc_{i}_{j} loc_{i}_{j+1})\n")

        f.write(")\n")

        # goal
        f.write("(:goal\n")
        f.write(f"(at_quad quad_0 loc_{tx}_{ty})\n")
        f.write(")\n")

        f.write(")")


# -----------------------------
# planner 실행
# -----------------------------

def run_planner():

    cmd = [
        "./planner/downward/fast-downward.py",
        "./pddl/domain.pddl",
        "./pddl/problem.pddl",
        "--search",
        "astar(lmcut())"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


# -----------------------------
# plan parsing
# -----------------------------

def parse_plan():
    plan = []
    
    if not os.path.exists("sas_plan"):
        print("Warning: sas_plan 파일을 찾을 수 없습니다.")
        return plan

    with open("sas_plan") as f:
        for line in f:
            if "move" in line:
                parts = line.strip().strip("()").split()
                if len(parts) >= 4:
                    target_loc = parts[-1] 
                    plan.append(target_loc)

    return plan


# -----------------------------
# plan 실행
# -----------------------------

def execute_plan(env, path):

    step_idx = 0

    if not path:
        print("경로(path)가 비어 있어 실행할 수 없습니다.")
        return

    for agent in env.agent_iter():

        # 매 턴마다 화면 렌더링 호출
        env.render()

        obs, reward, term, trunc, info = env.last()

        if term or trunc:
            action = None
        else:
            if agent == "quad_0":
                x = int(obs[0])
                y = int(obs[1])

                if step_idx < len(path):
                    target_loc = path[step_idx]
                    tx, ty = loc_to_grid(target_loc)

                    # 에이전트가 목표 위치에 도달했다면 다음 경로(step)로 인덱스 증가
                    if x == tx and y == ty:
                        step_idx += 1
                        
                    if step_idx < len(path):
                        action = grid_to_discrete_action(x, y, path[step_idx])
                    else:
                        action = 0 # 경로 끝남 (대기)
                else:
                    action = 0 # 대기
            else:
                action = env.action_space(agent).sample()

        env.step(action)


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    # render_mode="human" 설정, render_fps로 속도 조절
    env = SimpleFleetEnv(render_mode="human", render_fps=2)

    env.reset()

    quad0_pos = env.agent_positions["quad_0"]
    obs_pos = env.agent_positions["obs_0"]
    target = env.target

    print("초기 quad_0 위치:", quad0_pos)
    print("초기 obs_0 위치:", obs_pos)
    print("타겟 위치:", target)

    generate_problem(quad0_pos, obs_pos, target, env.grid_size)

    run_planner()

    plan = parse_plan()

    print("PLAN:", plan)

    execute_plan(env, plan)

    # 시뮬레이션 종료 후 창을 닫을 때까지 유지하는 루프
    print("시뮬레이션 종료. 창을 닫으려면 게임 창의 X 버튼을 누르세요.")
    
    if pygame is not None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # 화면이 응답 없음에 빠지지 않도록 프레임 업데이트 유지
            env.clock.tick(10)
            
    env.close()
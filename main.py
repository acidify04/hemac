import subprocess
import numpy as np
from src.hemac.environment.simple_fleet_env import SimpleFleetEnv
import os

GRID_SIZE = 50
MAX_SPEED = 16


# -----------------------------
# 좌표 → grid
# -----------------------------

def pos_to_loc(x, y):
    gx = int(x // GRID_SIZE)
    gy = int(y // GRID_SIZE)
    return f"loc_{gx}_{gy}"


def loc_to_grid(loc):
    _, x, y = loc.split("_")
    return int(x), int(y)


def grid_to_center(gx, gy):
    x = gx * GRID_SIZE + GRID_SIZE / 2
    y = gy * GRID_SIZE + GRID_SIZE / 2
    return x, y


# -----------------------------
# grid → velocity
# -----------------------------

def grid_to_velocity(current_x, current_y, target_loc):

    gx, gy = loc_to_grid(target_loc)
    tx, ty = grid_to_center(gx, gy)

    vx = tx - current_x
    vy = ty - current_y

    norm = np.linalg.norm([vx, vy])

    if norm > MAX_SPEED:
        vx = vx / norm * MAX_SPEED
        vy = vy / norm * MAX_SPEED

    return np.array([vx, vy, 0], dtype=np.float32)


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
                f.write(f"c{i}_{j} ")

        f.write("- location\n")
        f.write(")\n")

        # init
        f.write("(:init\n")

        x, y = quad0_pos
        f.write(f"(at_quad quad_0 c{x}_{y})\n")

        x, y = obs_pos
        f.write(f"(at_obs obs_0 c{x}_{y})\n")

        tx, ty = target
        f.write(f"(target c{tx}_{ty})\n")

        # grid adjacency
        for i in range(grid):
            for j in range(grid):

                if i > 0:
                    f.write(f"(adjacent c{i}_{j} c{i-1}_{j})\n")

                if i < grid - 1:
                    f.write(f"(adjacent c{i}_{j} c{i+1}_{j})\n")

                if j > 0:
                    f.write(f"(adjacent c{i}_{j} c{i}_{j-1})\n")

                if j < grid - 1:
                    f.write(f"(adjacent c{i}_{j} c{i}_{j+1})\n")

        f.write(")\n")

        # goal
        f.write("(:goal\n")
        f.write(f"(at_quad quad_0 c{tx}_{ty})\n")
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

    result = subprocess.run(cmd)
    print(result.stdout)
    print(result.stderr)


# -----------------------------
# plan parsing
# -----------------------------

def parse_plan():

    plan = []

    with open("sas_plan") as f:
        for line in f:
            if "move" in line:
                action = line.split("(")[1].split()[0]
                plan.append(action)

    return plan


# -----------------------------
# plan 실행
# -----------------------------

def execute_plan(env, path):

    step_idx = 0

    for agent in env.agent_iter():

        obs, reward, term, trunc, info = env.last()

        if term or trunc:
            action = None

        else:

            if "drone" in agent:

                x = obs[0]
                y = obs[1]

                if step_idx < len(path):

                    action = grid_to_velocity(x, y, path[step_idx])

                else:

                    action = np.array([0, 0, 0], dtype=np.float32)

            else:

                action = env.action_space(agent).sample()

        env.step(action)

        if agent == env.agents[-1]:

            step_idx += 1

            if step_idx >= len(path):
                break


# -----------------------------
# MAIN
# -----------------------------


env = SimpleFleetEnv()

env.reset()

quad0_pos = env.agent_positions["quad_0"]
obs_pos = env.agent_positions['obs_0']
target = env.target

generate_problem(quad0_pos, obs_pos, target, env.grid_size)

run_planner()

plan = parse_plan()

print("PLAN:", plan)
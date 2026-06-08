import subprocess
import numpy as np
from src.hemac.environment.simple_fleet_env import SimpleFleetEnv
import os
import time
import argparse

try:
    import pygame
except ImportError:
    pygame = None

# -----------------------------
# 설정 값 (각 에이전트별 시야 범위)
# -----------------------------
OBSERVER_FOV = 7.0  # 옵저버 시야
QUAD_FOV = 3.0      # 드론 시야


# -----------------------------
# 좌표 → grid
# -----------------------------
def loc_to_grid(loc):
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
# 공유 지도 업데이트 및 탐색 목표 계산
# -----------------------------
def update_shared_map(shared_map, agent_x, agent_y, fov, grid_size):
    """현재 에이전트의 위치와 FOV를 바탕으로 탐색 완료(1)를 업데이트합니다."""
    for i in range(grid_size):
        for j in range(grid_size):
            # 미개척지(0)이거나 다른 애가 예약해둔 곳(2)이어도 내가 먼저 봤으면 탐색 완료(1)로 덮어씀
            if shared_map[i, j] in [0, 2]:
                dist = np.linalg.norm([i - agent_x, j - agent_y])
                if dist <= fov:
                    shared_map[i, j] = 1 

def reserve_shared_map(shared_map, target_x, target_y, fov, grid_size):
    """에이전트가 목표로 정한 곳과 그 주변을 '예약됨(2)'으로 마킹합니다."""
    for i in range(grid_size):
        for j in range(grid_size):
            if shared_map[i, j] == 0:
                dist = np.linalg.norm([i - target_x, j - target_y])
                if dist <= fov:
                    shared_map[i, j] = 2 # 2: 탐색 예정 (예약됨)

def get_best_unexplored_waypoint(shared_map, grid_size, fov, current_x, current_y):
    """가장 많은 미개척지(0)를 효율적으로 밝힐 수 있는 좌표를 반환합니다. 예약된 곳(2)은 무시합니다."""
    unexplored = np.argwhere(shared_map == 0)
    if len(unexplored) == 0:
        return (int(np.random.randint(0, grid_size)), int(np.random.randint(0, grid_size)))
    
    if len(unexplored) > 50:
        indices = np.random.choice(len(unexplored), 50, replace=False)
        candidates = unexplored[indices]
    else:
        candidates = unexplored

    best_point = None
    max_score = -9999
    
    y_indices, x_indices = np.indices((grid_size, grid_size))
    
    for cx, cy in candidates:
        distances = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
        information_gain = np.sum((distances <= fov) & (shared_map == 0))
        travel_cost = np.linalg.norm([cx - current_x, cy - current_y])
        
        score = information_gain - (travel_cost * 0.5)
        
        if score > max_score:
            max_score = score
            best_point = (int(cx), int(cy))
            
    if best_point is None:
        idx = np.random.choice(len(unexplored))
        return (int(unexplored[idx][0]), int(unexplored[idx][1]))
        
    return best_point


# -----------------------------
# MA-PDDL Domain 생성
# -----------------------------
def generate_domain():
    os.makedirs("./pddl", exist_ok=True)
    with open("./pddl/domain.pddl", "w") as f:
        f.write("""(define (domain fleet_domain)
  (:requirements :typing)
  (:types quad observer location)
  (:predicates
    (at_quad ?q - quad ?loc - location)
    (at_obs ?o - observer ?loc - location)
    (adjacent ?loc1 - location ?loc2 - location)
    (target ?loc - location)
  )
  (:action move_quad
    :parameters (?q - quad ?from - location ?to - location)
    :precondition (and (at_quad ?q ?from) (adjacent ?from ?to))
    :effect (and (not (at_quad ?q ?from)) (at_quad ?q ?to))
  )
  (:action move_obs
    :parameters (?o - observer ?from - location ?to - location)
    :precondition (and (at_obs ?o ?from) (adjacent ?from ?to))
    :effect (and (not (at_obs ?o ?from)) (at_obs ?o ?to))
  )
)""")


# -----------------------------
# [로직 1] 정찰 페이즈: 세 에이전트 동시 탐색
# -----------------------------
def generate_multi_search_problem(q0_pos, q1_pos, obs_pos, q0_goal, q1_goal, obs_goal, grid):
    with open("./pddl/problem.pddl", "w") as f:
        f.write("(define (problem multi_search_problem)\n")
        f.write("(:domain fleet_domain)\n")
        f.write("(:objects\n")
        f.write("quad_0 quad_1 - quad\n")
        f.write("obs_0 - observer\n")
        for i in range(grid):
            for j in range(grid):
                f.write(f"loc_{i}_{j} ")
        f.write("- location\n)\n")
        
        f.write("(:init\n")
        f.write(f"(at_quad quad_0 loc_{int(q0_pos[0])}_{int(q0_pos[1])})\n")
        f.write(f"(at_quad quad_1 loc_{int(q1_pos[0])}_{int(q1_pos[1])})\n")
        f.write(f"(at_obs obs_0 loc_{int(obs_pos[0])}_{int(obs_pos[1])})\n")
        for i in range(grid):
            for j in range(grid):
                if i > 0: f.write(f"(adjacent loc_{i}_{j} loc_{i-1}_{j})\n")
                if i < grid - 1: f.write(f"(adjacent loc_{i}_{j} loc_{i+1}_{j})\n")
                if j > 0: f.write(f"(adjacent loc_{i}_{j} loc_{i}_{j-1})\n")
                if j < grid - 1: f.write(f"(adjacent loc_{i}_{j} loc_{i}_{j+1})\n")
        f.write(")\n")
        
        f.write("(:goal (and\n")
        f.write(f"  (at_quad quad_0 loc_{int(q0_goal[0])}_{int(q0_goal[1])})\n")
        f.write(f"  (at_quad quad_1 loc_{int(q1_goal[0])}_{int(q1_goal[1])})\n")
        f.write(f"  (at_obs obs_0 loc_{int(obs_goal[0])}_{int(obs_goal[1])})\n")
        f.write("))\n")
        f.write(")")


# -----------------------------
# [로직 2] 도착 페이즈: 드론 동시 출동
# -----------------------------
def generate_arrive_problem(q0_pos, q1_pos, obs_pos, target, grid):
    with open("./pddl/problem.pddl", "w") as f:
        f.write("(define (problem arrive_problem)\n")
        f.write("(:domain fleet_domain)\n")
        f.write("(:objects\n")
        f.write("quad_0 quad_1 - quad\n")
        f.write("obs_0 - observer\n")
        for i in range(grid):
            for j in range(grid):
                f.write(f"loc_{i}_{j} ")
        f.write("- location\n)\n")
        
        f.write("(:init\n")
        f.write(f"(at_quad quad_0 loc_{int(q0_pos[0])}_{int(q0_pos[1])})\n")
        f.write(f"(at_quad quad_1 loc_{int(q1_pos[0])}_{int(q1_pos[1])})\n")
        f.write(f"(at_obs obs_0 loc_{int(obs_pos[0])}_{int(obs_pos[1])})\n")
        for i in range(grid):
            for j in range(grid):
                if i > 0: f.write(f"(adjacent loc_{i}_{j} loc_{i-1}_{j})\n")
                if i < grid - 1: f.write(f"(adjacent loc_{i}_{j} loc_{i+1}_{j})\n")
                if j > 0: f.write(f"(adjacent loc_{i}_{j} loc_{i}_{j-1})\n")
                if j < grid - 1: f.write(f"(adjacent loc_{i}_{j} loc_{i}_{j+1})\n")
        f.write(")\n")
        
        f.write("(:goal (and\n")
        f.write(f"  (at_quad quad_0 loc_{int(target[0])}_{int(target[1])})\n")
        f.write(f"  (at_quad quad_1 loc_{int(target[0])}_{int(target[1])})\n")
        f.write("))\n")
        f.write(")")


# -----------------------------
# Fast Downward 실행
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
    if result.returncode != 0 and result.stderr:
        print("플래너 에러 발생:\n", result.stderr)


# -----------------------------
# PDDL plan parsing
# -----------------------------
def parse_plan():
    plans = {"quad_0": [], "quad_1": [], "obs_0": []}
    if not os.path.exists("sas_plan"):
        print("Warning: sas_plan 파일을 찾을 수 없습니다.")
        return plans

    with open("sas_plan") as f:
        for line in f:
            if "move_quad" in line or "move_obs" in line:
                parts = line.strip().strip("()").split()
                if len(parts) >= 4:
                    agent_name = parts[1]
                    target_loc = parts[3]
                    if agent_name in plans:
                        plans[agent_name].append(target_loc)
    return plans


# -----------------------------
# Pygame UI 및 공유 지도 렌더링
# -----------------------------
def draw_ui_and_map(env, shared_map, strategy, target_ratio, explored_ratio, target_found, is_arrive_phase):
    if pygame is None or env.screen is None:
        return
        
    overlay = pygame.Surface((env.window_size, env.window_size), pygame.SRCALPHA)
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            rect = pygame.Rect(i * env.cell_size, j * env.cell_size, env.cell_size, env.cell_size)
            if shared_map[i, j] == 1:
                pygame.draw.rect(overlay, (255, 255, 0, 40), rect)   # 노란색 (탐색 완료)
            elif shared_map[i, j] == 2:
                pygame.draw.rect(overlay, (0, 200, 255, 60), rect) # 하늘색 (예약됨)
    env.screen.blit(overlay, (0, 0))

    font = pygame.font.SysFont(None, 24)
    
    ratio_str = f" (Target: {target_ratio*100:.0f}%)" if strategy in ["broad", "safe"] else ""
    strategy_text = font.render(f"Strategy: {strategy.upper()}{ratio_str}", True, (0, 0, 0))
    env.screen.blit(strategy_text, (10, 10))
    
    color = (0, 150, 0) if (strategy in ["broad", "safe"] and explored_ratio >= target_ratio) else (0, 0, 0)
    explore_text = font.render(f"Explored: {explored_ratio*100:.1f}%", True, color)
    env.screen.blit(explore_text, (10, 35))
    
    status = "ARRIVING" if is_arrive_phase else ("TARGET FOUND! (Waiting)" if target_found else "SEARCHING")
    status_color = (255, 0, 0) if is_arrive_phase else ((255, 165, 0) if target_found else (0, 0, 255))
    status_text = font.render(f"Status: {status}", True, status_color)
    env.screen.blit(status_text, (10, 60))
    
    pygame.display.flip()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # 터미널 파서 설정
    parser = argparse.ArgumentParser(description="Multi-Agent Fleet Planning")
    parser.add_argument("--strategy", type=str, choices=["fast", "broad", "safe"], default="broad", 
                        help="탐색 전략 (fast: 즉시 출동, broad/safe: 지정 비율 탐색 후 출동)")
    parser.add_argument("--ratio", type=float, default=0.75, 
                        help="broad/safe 전략 사용 시 달성해야 할 탐색 비율 (0.0 ~ 1.0)")
    args = parser.parse_args()

    GLOBAL_STRATEGY = args.strategy
    TARGET_RATIO = args.ratio

    generate_domain()

    env = SimpleFleetEnv(render_mode="human", render_fps=0)
    env.reset()

    target_found = False
    arrive_phase_triggered = False
    known_target_loc = None
    
    shared_map = np.zeros((env.grid_size, env.grid_size))
    total_cells = env.grid_size * env.grid_size
    
    q0_pos = env.agent_positions["quad_0"]
    q1_pos = env.agent_positions["quad_1"]
    o_pos  = env.agent_positions["obs_0"]

    update_shared_map(shared_map, q0_pos[0], q0_pos[1], QUAD_FOV, env.grid_size)
    update_shared_map(shared_map, q1_pos[0], q1_pos[1], QUAD_FOV, env.grid_size)
    update_shared_map(shared_map, o_pos[0], o_pos[1], OBSERVER_FOV, env.grid_size)
    
    current_goals = {}
    
    goal_0 = get_best_unexplored_waypoint(shared_map, env.grid_size, QUAD_FOV, q0_pos[0], q0_pos[1])
    reserve_shared_map(shared_map, goal_0[0], goal_0[1], QUAD_FOV, env.grid_size)
    current_goals["quad_0"] = goal_0

    goal_1 = get_best_unexplored_waypoint(shared_map, env.grid_size, QUAD_FOV, q1_pos[0], q1_pos[1])
    reserve_shared_map(shared_map, goal_1[0], goal_1[1], QUAD_FOV, env.grid_size)
    current_goals["quad_1"] = goal_1

    goal_o = get_best_unexplored_waypoint(shared_map, env.grid_size, OBSERVER_FOV, o_pos[0], o_pos[1])
    reserve_shared_map(shared_map, goal_o[0], goal_o[1], OBSERVER_FOV, env.grid_size)
    current_goals["obs_0"] = goal_o
    
    multi_agent_plans = {"quad_0": [], "quad_1": [], "obs_0": []}
    step_indices = {"quad_0": 0, "quad_1": 0, "obs_0": 0}
    
    ratio_msg = f" (목표 탐색률: {TARGET_RATIO*100:.1f}%)" if GLOBAL_STRATEGY in ["broad", "safe"] else ""
    print(f"\n[{GLOBAL_STRATEGY.upper()} 전략] 공동 정찰을 시작합니다.{ratio_msg}")
    generate_multi_search_problem(
        q0_pos, q1_pos, o_pos, 
        current_goals["quad_0"], current_goals["quad_1"], current_goals["obs_0"], 
        env.grid_size
    )
    run_planner()
    multi_agent_plans.update(parse_plan())

    for agent in env.agent_iter():
        env.render()
        obs, reward, term, trunc, info = env.last()

        if term or trunc:
            env.step(None)
            continue

        current_x = int(obs[0])
        current_y = int(obs[1])
        action = 0
        my_fov = OBSERVER_FOV if agent == "obs_0" else QUAD_FOV

        update_shared_map(shared_map, current_x, current_y, my_fov, env.grid_size)
        explored_ratio = np.sum(shared_map == 1) / total_cells

        draw_ui_and_map(env, shared_map, GLOBAL_STRATEGY, TARGET_RATIO, explored_ratio, target_found, arrive_phase_triggered)

        tx, ty = env.target
        dist = np.linalg.norm([current_x - tx, current_y - ty])

        if not target_found and dist <= my_fov:
            print(f"\n[타겟 관측!] {agent}가 타겟을 발견했습니다! (위치: {tx}, {ty})")
            target_found = True
            known_target_loc = (tx, ty)

        if target_found and not arrive_phase_triggered:
            trigger_arrive = False
            
            if GLOBAL_STRATEGY == "fast":
                print("\n[FAST 전략] 타겟 발견 즉시 도착 페이즈로 전환합니다!")
                trigger_arrive = True
            elif GLOBAL_STRATEGY in ["broad", "safe"] and explored_ratio >= TARGET_RATIO:
                print(f"\n[{GLOBAL_STRATEGY.upper()} 전략] 목표 탐색률 달성! (현재 {explored_ratio*100:.1f}%) 도착 페이즈로 전환합니다!")
                trigger_arrive = True
                
            if trigger_arrive:
                arrive_phase_triggered = True
                q0_pos = env.agent_positions["quad_0"]
                q1_pos = env.agent_positions["quad_1"]
                o_pos  = env.agent_positions["obs_0"]
                
                generate_arrive_problem(q0_pos, q1_pos, o_pos, known_target_loc, env.grid_size)
                run_planner()
                
                multi_agent_plans = {"quad_0": [], "quad_1": [], "obs_0": []}
                multi_agent_plans.update(parse_plan())
                step_indices = {"quad_0": 0, "quad_1": 0, "obs_0": 0}

        my_plan = multi_agent_plans[agent]
        my_idx = step_indices[agent]

        if my_idx < len(my_plan):
            target_loc = my_plan[my_idx]
            wx, wy = loc_to_grid(target_loc)
            if current_x == wx and current_y == wy:
                step_indices[agent] += 1
                my_idx += 1
            
            if my_idx < len(my_plan):
                action = grid_to_discrete_action(current_x, current_y, my_plan[my_idx])
        else:
            action = 0
            
            if not arrive_phase_triggered:
                new_goal = get_best_unexplored_waypoint(shared_map, env.grid_size, my_fov, current_x, current_y)
                reserve_shared_map(shared_map, new_goal[0], new_goal[1], my_fov, env.grid_size)
                current_goals[agent] = new_goal
                
                q0_pos = env.agent_positions["quad_0"]
                q1_pos = env.agent_positions["quad_1"]
                o_pos  = env.agent_positions["obs_0"]
                
                generate_multi_search_problem(
                    q0_pos, q1_pos, o_pos, 
                    current_goals["quad_0"], current_goals["quad_1"], current_goals["obs_0"], 
                    env.grid_size
                )
                run_planner()
                
                new_plans = parse_plan()
                for a in ["quad_0", "quad_1", "obs_0"]:
                    multi_agent_plans[a] = new_plans.get(a, [])
                    step_indices[a] = 0
                
                if multi_agent_plans[agent]:
                    action = grid_to_discrete_action(current_x, current_y, multi_agent_plans[agent][0])

        env.step(action)

        if env._agent_selector.is_last() and pygame is not None:
            time.sleep(0.05)

    print("\n시뮬레이션 종료. 창을 닫으려면 게임 창의 X 버튼을 누르세요.")
    
    if pygame is not None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            env.clock.tick(10)
            
    env.close()
"""Example script to run trained models/baseline and vizualize test episodes on HeMAC."""

import os
import glob
import numpy as np

from stable_baselines3 import PPO, SAC, DQN

from hemac import HeMAC_v0

from stable_baselines3.common.logger import configure

import datetime


def generate_unique_log_dir(base_dir, mode, algorithm):
    """Generate a unique log directory based on the mode (train/eval), algorithm, and timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(base_dir, f"{mode}_{algorithm}_{timestamp}")


def eval(
    env_fn,
    num_games: int = 100,
    render_mode: str | None = None,
    baseline: bool = False,
    algorithm: str = "PPO",
    log_dir="./tensorboard_logs/eval",
    task="long_map",
    **env_kwargs,
):
    """Evaluate trained policy or rule-based baseline."""
    log_dir = generate_unique_log_dir("./tensorboard_logs", "eval", algorithm)
    print(f"Evaluation log directory: {log_dir}")

    logger = configure(log_dir, ["tensorboard", "stdout"])
    logger.set_level(1)

    # instantiate env
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print(f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})")

    if not baseline:
        try:
            models = {}
            for agent in env.possible_agents:
                algorithm_specific_policies = glob.glob(f"{task}_{algorithm}_{agent}*.zip")

                if not algorithm_specific_policies:
                    raise ValueError("No policies found for the selected algorithm.")

                latest_policy = max(algorithm_specific_policies, key=os.path.getctime)
                print(f"evaluating policy : {latest_policy}")

                if algorithm == "PPO":
                    models[agent] = PPO.load(latest_policy)
                elif algorithm == "SAC":
                    models[agent] = SAC.load(latest_policy)
                elif algorithm == "DQN":
                    models[agent] = DQN.load(latest_policy)
                else:
                    raise ValueError(f"Unsupported algorithm: {algorithm}")

        except ValueError as e:
            print(f"Policy not found: {e}")
            exit(0)

    total_rewards_per_episode = []
    episode_lengths = []
    rewards = {agent: 0 for agent in env.possible_agents}
    reward_window = []

    step = 0

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        episode_reward = 0
        steps = 0

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            steps += 1

            for a in env.agents:
                rewards[a] += env.rewards[a]
            episode_reward += reward

            if termination or truncation:
                break
            else:
                if baseline:
                    # New baseline: simple strategy to follow the POI
                    if algorithm == "sanity_check":
                        if "drone" in agent:
                            comm = obs[0:2]
                            if obs[5] < 10 or obs[6] < 10 or obs[7] < 10 or obs[8] < 10:
                                act = np.array([0, 0, np.random.randint(2)], dtype=np.float32)
                            else:
                                act = np.array([comm[0], comm[1], np.random.randint(2)], dtype=np.float32)
                            act = np.clip(act, -10, 10)
                        elif "observer" in agent:
                            act = 1  # go in circle
                        else:
                            act = env.action_space(agent).sample()
                    else:
                        # For any other algo, use a random action (or adapt as needed)
                        act = env.action_space(agent).sample()
                else:
                    act = models[agent].predict(obs, deterministic=True)[0]
                    if "drone" in agent:
                        print(f"drone action: {act}")
                    elif "observer" in agent:
                        print(f"observer action: {act}")

            env.step(act)

        total_rewards_per_episode.append(episode_reward)
        episode_lengths.append(steps)
        reward_window.append(episode_reward)

        # Calculate average episode length
        avg_episode_length = np.mean(episode_lengths)
        avg_reward_per_episode = np.mean(total_rewards_per_episode)

        logger.record("eval/episode_reward", episode_reward, step)
        logger.record("eval/episode_length", steps, step)
        logger.record("eval/average_episode_length", avg_episode_length, step)
        logger.record("eval/avg_reward_per_episode", avg_reward_per_episode, step)

        logger.dump(step)

        print(f"Episode {i + 1}/{num_games}, Reward: {episode_reward}, Steps: {steps}")

        step += 1

    env.close()

    logger.dump(num_games)

    print(f"\nAverage reward per episode: {avg_reward_per_episode}")
    print(f"Average episode length: {avg_episode_length}")

    return avg_reward_per_episode, avg_episode_length


if __name__ == "__main__":
    env_fn = HeMAC_v0

    env_kwargs_level1 = dict(
        time_factor=0.5,
        area_size=(500, 500),
        max_cycles=600,
        render_ratio=0.6,
        n_observers=3,
        n_drones=6,
        n_provisioners=0,
        min_obstacles=1,
        max_obstacles=2,
        rescuing_targets=True,
        observer_comm_range=300,
        patrol_config={
            "benchmark": True,
            "area": [(100, 100), (250, 100), (820, 480), (600, 800), (100, 620)],  # pentagon
        },
        poi_config=[{"speed": 1.0, "dimension": [8, 8], "spawn_mode": "random"}],
        drone_config={
            "drones_starting_pos": [],
            "drone_ui_dimension": 16,
            "drone_max_speed": 10,
            "drone_max_charge": 9999,
            "discrete_action_space": False,  # True if QMIX else false
        },
        drone_sensor={"model": "RoundCamera", "params": {"sensing_range": 30}},
        observer_sensor={
            "model": "ForwardFacingCamera",
            "params": {
                "hfov": np.pi / 6,  # rad
                "sensing_range": 100,
            },
        },
        provisioner_sensor={
            "model": "ForwardFacingCamera",
            "params": {
                "hfov": np.pi / 2,  # rad
                "sensing_range": 30,
            },
        },
    )
    # Watch a game in the normal environment
    eval(
        env_fn,
        num_games=10,
        render_mode="human",
        algorithm="sanity_check",
        baseline=True,
        task="pentagon",
        **env_kwargs_level1,
    )

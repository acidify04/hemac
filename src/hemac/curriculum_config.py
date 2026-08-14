"""Shared HeMAC environment-difficulty definitions."""

OBSTACLE_CURRICULUM_LEVELS = (
    {
        "min_obstacles": 3,
        "max_obstacles": 4,
        "obstacle_min_speed": 1,
        "obstacle_max_speed": 3,
        "n_static_obstacles": 2,
        "goal_min_base_distance": 475.0,
        "goal_max_base_distance": 600.0,
    },
    {
        "min_obstacles": 4,
        "max_obstacles": 5,
        "obstacle_min_speed": 2,
        "obstacle_max_speed": 3,
        "n_static_obstacles": 2,
        "goal_min_base_distance": 550.0,
        "goal_max_base_distance": 675.0,
    },
    {
        "min_obstacles": 4,
        "max_obstacles": 6,
        "obstacle_min_speed": 2,
        "obstacle_max_speed": 5,
        "n_static_obstacles": 3,
        "goal_min_base_distance": 625.0,
        "goal_max_base_distance": 750.0,
    },
    {
        "min_obstacles": 5,
        "max_obstacles": 7,
        "obstacle_min_speed": 2,
        "obstacle_max_speed": 6,
        "n_static_obstacles": 3,
        "goal_min_base_distance": 700.0,
        "goal_max_base_distance": 825.0,
    },
    {
        "min_obstacles": 6,
        "max_obstacles": 8,
        "obstacle_min_speed": 3,
        "obstacle_max_speed": 7,
        "n_static_obstacles": 3,
        "goal_min_base_distance": 775.0,
        "goal_max_base_distance": 925.0,
    },
    {
        "min_obstacles": 7,
        "max_obstacles": 9,
        "obstacle_min_speed": 3,
        "obstacle_max_speed": 7,
        "n_static_obstacles": 3,
        "goal_min_base_distance": 850.0,
        "goal_max_base_distance": 1000.0,
    },
)


def get_obstacle_curriculum_level(stage_number: int) -> dict:
    """Return a copy of one 1-based obstacle curriculum level."""
    stage_number = int(stage_number)
    if not 1 <= stage_number <= len(OBSTACLE_CURRICULUM_LEVELS):
        raise ValueError(
            f"Difficulty must be between 1 and {len(OBSTACLE_CURRICULUM_LEVELS)}, "
            f"got {stage_number}."
        )
    return dict(OBSTACLE_CURRICULUM_LEVELS[stage_number - 1])

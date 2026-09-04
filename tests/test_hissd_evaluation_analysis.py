from src.skill_discovery.analyze_hissd_evaluation import analyze_results


def test_primary_success_uses_selected_task_success() -> None:
    episodes = [
        {
            "controller": controller,
            "difficulty": 3,
            "seed": 10,
            "success": task_success,
            "mission_success": mission_success,
            "drone_task_success": task_success,
            "goal_found": task_success,
            "drone_goal_found": task_success,
            "fatal_crash": False,
            "drone_crash": False,
            "observer_crash": False,
            "coverage_ratio": 0.6,
            "cycles": 20,
        }
        for controller, task_success, mission_success in (
            ("hissd", True, False),
            ("mappo", False, True),
        )
    ]

    result = analyze_results(episodes, bootstrap_samples=20, seed=1)
    comparison = result["difficulty_3/hissd_vs_mappo"]

    assert comparison["success"]["difference"] == 1.0
    assert comparison["mission_success"]["difference"] == -1.0

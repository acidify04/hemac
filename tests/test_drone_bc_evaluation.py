"""Tests for offline BC rollout metric aggregation."""

from src.skill_discovery.evaluate_drone_bc import EpisodeResult, summarize


def test_summarize_drone_bc_episode_results() -> None:
    """Rollout rates and continuous means should aggregate per episode."""
    results = [
        EpisodeResult("bc", 1, 1, 10, True, True, True, False, False, False, 0.8),
        EpisodeResult("bc", 1, 2, 20, False, True, False, True, True, False, 0.4),
    ]

    summary = summarize(results)

    assert summary["episodes"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["goal_found_rate"] == 1.0
    assert summary["drone_goal_found_rate"] == 0.5
    assert summary["drone_crash_rate"] == 0.5
    assert summary["mean_coverage_ratio"] == 0.6
    assert summary["mean_cycles"] == 15.0

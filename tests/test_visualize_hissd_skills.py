"""Tests for episode- and difficulty-level HiSSD skill comparisons."""

import numpy as np

from src.skill_discovery.visualize_hissd_skills import (
    episode_means,
    skill_comparison_statistics,
)


def test_episode_means_keep_difficulty_labels() -> None:
    values = np.asarray([[0.0, 0.0], [2.0, 0.0], [10.0, 2.0]])
    difficulties = np.asarray([1, 1, 2])
    groups = np.asarray([11, 11, 21])

    means, labels, unique_groups = episode_means(values, difficulties, groups)

    np.testing.assert_allclose(means, [[1.0, 0.0], [10.0, 2.0]])
    np.testing.assert_array_equal(labels, [1, 2])
    np.testing.assert_array_equal(unique_groups, [11, 21])


def test_skill_comparison_statistics_separate_difficulties() -> None:
    base = np.asarray(
        [[0.0, 0.0], [0.2, 0.0], [0.0, 0.2], [3.0, 0.0], [3.2, 0.0], [3.0, 0.2]],
        dtype=np.float32,
    )
    embeddings = {
        "common": base,
        "task": base * 2.0,
        "contrastive": base * 0.5,
        "difficulty": np.asarray([1, 1, 1, 2, 2, 2]),
        "episode_group": np.asarray([10, 10, 11, 20, 20, 21]),
    }

    statistics = skill_comparison_statistics(embeddings, within_difficulty=1)

    assert statistics["difficulties"] == [1, 2]
    for result in statistics["representations"].values():
        assert np.asarray(result["centroid_distance_matrix"]).shape == (2, 2)
        assert result["between_within_ratio"] > 1.0
        assert result["per_difficulty"]["1"]["episodes"] == 2

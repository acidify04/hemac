from src.train.train import (
    CURRICULUM_STABILITY_WINDOW,
    CoverageCurriculum,
    ObstacleDifficultyCurriculum,
)


def _assert_five_consecutive_evaluations_required(curriculum):
    for _ in range(CURRICULUM_STABILITY_WINDOW - 1):
        assert not curriculum.record_success(0.8)

    assert not curriculum.record_success(0.75)

    for _ in range(CURRICULUM_STABILITY_WINDOW - 1):
        assert not curriculum.record_success(0.9)

    assert curriculum.record_success(0.9)
    assert curriculum.stage_number == 2
    assert not curriculum.recent_success_rates


def test_coverage_curriculum_requires_five_consecutive_successes():
    assert CURRICULUM_STABILITY_WINDOW == 5
    curriculum = CoverageCurriculum(
        [0.3, 0.4],
        promotion_success_rate=0.8,
        stability_window=CURRICULUM_STABILITY_WINDOW,
    )
    _assert_five_consecutive_evaluations_required(curriculum)


def test_obstacle_curriculum_requires_five_consecutive_successes():
    curriculum = ObstacleDifficultyCurriculum(
        [
            {"min_obstacles": 1, "max_obstacles": 2},
            {"min_obstacles": 2, "max_obstacles": 3},
        ],
        promotion_success_rate=0.8,
        stability_window=CURRICULUM_STABILITY_WINDOW,
    )
    _assert_five_consecutive_evaluations_required(curriculum)

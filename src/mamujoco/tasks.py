"""Task suites for HalfCheetah 6x1 zero-shot transfer experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


AGENT_IDS = tuple(f"agent_{index}" for index in range(6))
JOINT_NAMES = (
    "back_thigh",
    "back_shin",
    "back_foot",
    "front_thigh",
    "front_shin",
    "front_foot",
)
JOINT_TO_AGENT = dict(zip(JOINT_NAMES, AGENT_IDS))
SUPPORTED_SUITES = ("difficulty", "dynamics", "joint_disable")
DEFAULT_DIFFICULTY_STRENGTHS = (1.0, 0.8, 0.6, 0.4)
DEFAULT_SOURCE_DIFFICULTIES = ("D1", "D2")
DEFAULT_TARGET_DIFFICULTIES = ("D3", "D4")


@dataclass(frozen=True)
class TaskSpec:
    """One source or unseen target MDP in a transfer suite."""

    suite: str
    name: str
    split: str
    disabled_agent: str | None = None
    mass_scale: float = 1.0
    friction_scale: float = 1.0
    actuator_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.suite not in SUPPORTED_SUITES:
            raise ValueError(f"Unknown task suite: {self.suite}")
        if self.split not in ("source", "target"):
            raise ValueError(f"Unknown split: {self.split}")
        if self.disabled_agent is not None and self.disabled_agent not in AGENT_IDS:
            raise ValueError(f"Unknown disabled agent: {self.disabled_agent}")
        for field_name in ("mass_scale", "friction_scale", "actuator_scale"):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"{field_name} must be positive")

    @property
    def disabled_joint(self) -> str | None:
        if self.disabled_agent is None:
            return None
        return JOINT_NAMES[int(self.disabled_agent.rsplit("_", 1)[1])]

    @property
    def active_agents(self) -> tuple[str, ...]:
        return tuple(agent for agent in AGENT_IDS if agent != self.disabled_agent)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["disabled_joint"] = self.disabled_joint
        return result


_JOINT_DISABLE_TASKS = (
    TaskSpec("joint_disable", "complete", "source"),
    TaskSpec("joint_disable", "back_thigh", "source", "agent_0"),
    TaskSpec("joint_disable", "back_foot", "source", "agent_2"),
    TaskSpec("joint_disable", "front_thigh", "source", "agent_3"),
    TaskSpec("joint_disable", "front_shin", "source", "agent_4"),
    TaskSpec("joint_disable", "back_shin", "target", "agent_1"),
    TaskSpec("joint_disable", "front_foot", "target", "agent_5"),
)

# This suite keeps the observation/action interface fixed while holding out an
# entire dynamics factor (actuator strength) during source-task training.
_DYNAMICS_TASKS = (
    TaskSpec("dynamics", "nominal", "source"),
    TaskSpec("dynamics", "mass_075", "source", mass_scale=0.75),
    TaskSpec("dynamics", "mass_125", "source", mass_scale=1.25),
    TaskSpec("dynamics", "friction_075", "source", friction_scale=0.75),
    TaskSpec("dynamics", "friction_125", "source", friction_scale=1.25),
    TaskSpec("dynamics", "actuator_075", "target", actuator_scale=0.75),
    TaskSpec("dynamics", "actuator_125", "target", actuator_scale=1.25),
)


def build_difficulty_tasks(
    strengths: tuple[float, ...] = DEFAULT_DIFFICULTY_STRENGTHS,
    source_ids: tuple[str, ...] = DEFAULT_SOURCE_DIFFICULTIES,
    target_ids: tuple[str, ...] = DEFAULT_TARGET_DIFFICULTIES,
) -> tuple[TaskSpec, ...]:
    """Build the configurable same-task actuator-strength protocol."""
    strengths = tuple(float(value) for value in strengths)
    if len(strengths) != 4 or any(value <= 0 for value in strengths):
        raise ValueError("difficulty strengths must contain four positive values")
    difficulty_ids = tuple(f"D{index}" for index in range(1, 5))
    source_ids = tuple(source_ids)
    target_ids = tuple(target_ids)
    if len(source_ids) != 2 or len(target_ids) != 2:
        raise ValueError("difficulty protocol requires two source and two target IDs")
    if set(source_ids) & set(target_ids):
        raise ValueError("source and target difficulties must be disjoint")
    if set(source_ids) | set(target_ids) != set(difficulty_ids):
        raise ValueError("source and target difficulties must partition D1..D4")
    return tuple(
        TaskSpec(
            "difficulty",
            difficulty_id,
            "source" if difficulty_id in source_ids else "target",
            actuator_scale=strength,
        )
        for difficulty_id, strength in zip(difficulty_ids, strengths)
    )


_DIFFICULTY_TASKS = build_difficulty_tasks()

TASK_SUITES = {
    "difficulty": _DIFFICULTY_TASKS,
    "joint_disable": _JOINT_DISABLE_TASKS,
    "dynamics": _DYNAMICS_TASKS,
}


def list_tasks(
    suite: str,
    split: str | None = None,
    *,
    difficulty_strengths: tuple[float, ...] | None = None,
    source_difficulties: tuple[str, ...] | None = None,
    target_difficulties: tuple[str, ...] | None = None,
) -> tuple[TaskSpec, ...]:
    """Return tasks in stable experiment order."""
    if suite not in TASK_SUITES:
        raise ValueError(
            f"Unknown suite {suite!r}; choose from {', '.join(SUPPORTED_SUITES)}"
        )
    if suite == "difficulty" and any(
        value is not None
        for value in (
            difficulty_strengths,
            source_difficulties,
            target_difficulties,
        )
    ):
        tasks = build_difficulty_tasks(
            difficulty_strengths or DEFAULT_DIFFICULTY_STRENGTHS,
            source_difficulties or DEFAULT_SOURCE_DIFFICULTIES,
            target_difficulties or DEFAULT_TARGET_DIFFICULTIES,
        )
    else:
        tasks = TASK_SUITES[suite]
    if split is None:
        return tasks
    if split not in ("source", "target"):
        raise ValueError("split must be 'source', 'target', or None")
    return tuple(task for task in tasks if task.split == split)


def get_task(suite: str, name: str, **task_options) -> TaskSpec:
    """Resolve one task by suite and stable CLI name."""
    for task in list_tasks(suite, **task_options):
        if task.name == name:
            return task
    available = ", ".join(task.name for task in list_tasks(suite, **task_options))
    raise ValueError(f"Unknown {suite} task {name!r}; choose from {available}")


def task_names(
    suite: str, split: str | None = None, **task_options
) -> tuple[str, ...]:
    return tuple(task.name for task in list_tasks(suite, split, **task_options))

"""Offline multi-task learning experiments on maintained MaMuJoCo."""

from .tasks import TaskSpec, get_task, list_tasks

__all__ = ["TaskSpec", "get_task", "list_tasks"]

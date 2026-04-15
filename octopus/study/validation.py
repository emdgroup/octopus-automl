"""Validation functions for OctoStudy attributes."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from attrs import Attribute

from octopus.modules import Task
from octopus.modules.mrmr import Mrmr
from octopus.types import FIComputeMethod, MLType, MRMRRelevance

if TYPE_CHECKING:
    from octopus.study.core import OctoStudy


def validate_workflow(_instance: "OctoStudy", attribute: Attribute, value: Sequence[Task]) -> None:
    """Validate the `workflow` attribute.

    Ensures that the value is a non-empty list where each item is an
    instance of `Task`, and that the workflow meets specified
    conditions.

    Conditions:
    - The first task must have `task_id=0`.
    - All items with `depends_on=None` must be at the start of the list,
      before any items with `depends_on` set.
    - All elements in the list must be in increasing order of `task_id`.
    - For elements with `depends_on` set, the value must
      refer to a `task_id` that comes before them in the list.
    - All `task_id`s should form a complete integer sequence with no
      missing values between the minimum and maximum `task_id`.

    Args:
        _instance: The instance that is being validated (unused).
        attribute: The attribute that is being validated.
        value: The value of the attribute to validate.

    Raises:
        TypeError: If any item in the list is not an instance of
            `Task`.
        ValueError: If the list is empty or does not meet the specified
            conditions.
    """
    # Condition 1: Non-Empty List
    if not value:
        raise ValueError(f"'{attribute.name}' must contain at least one Task.")

    # Condition 2: All Items are Instances of Task
    for item in value:
        if not isinstance(item, Task):
            raise TypeError(
                f"Each item in '{attribute.name}' must be an instance of 'Task', but got '{type(item).__name__}'."
            )

    # Condition 2.5: First Item Must Have task_id=0
    if value[0].task_id != 0:
        raise ValueError(f"The first task must have 'task_id=0', but got 'task_id={value[0].task_id}'.")

    # Build mapping of task_id to index/task and collect task_ids
    task_id_to_index = {}
    task_id_to_task: dict[int, Task] = {}
    task_ids = []
    previous_task_id = None
    for idx, item in enumerate(value):
        # Ensure that task_ids are in increasing order
        if previous_task_id is not None and item.task_id <= previous_task_id:
            raise ValueError(
                f"Item at position {idx + 1} has 'task_id' {item.task_id}, "
                "which is not greater than the previous "
                f"'task_id' {previous_task_id}. "
                "All 'task_id's must be in increasing order in the list."
            )
        previous_task_id = item.task_id

        if item.task_id in task_id_to_index:
            raise ValueError(f"Duplicate 'task_id' {item.task_id} found in the workflow.")
        task_id_to_index[item.task_id] = idx
        task_id_to_task[item.task_id] = item
        task_ids.append(item.task_id)

    # Condition 3: All task_ids form a complete integer sequence with no missing
    # values between min and max
    min_task_id = min(task_ids)
    max_task_id = max(task_ids)
    expected_task_ids = set(range(min_task_id, max_task_id + 1))
    actual_task_ids = set(task_ids)
    if expected_task_ids != actual_task_ids:
        missing_ids = expected_task_ids - actual_task_ids
        extra_ids = actual_task_ids - expected_task_ids
        message = "All 'task_id's must form a complete integer sequence with no missing values between the minimum and maximum 'task_id'."
        if missing_ids:
            message += f" Missing task_ids: {sorted(missing_ids)}."
        if extra_ids:
            message += f" Unexpected task_ids: {sorted(extra_ids)}."
        raise ValueError(message)

    # Condition 4: All items with depends_on=None must be at the start of the list
    reached_dependent = False
    for idx, item in enumerate(value):
        if item.depends_on is None:
            if reached_dependent:
                raise ValueError(
                    f"Item at position {idx + 1} has 'depends_on=None' but"
                    " appears after items with 'depends_on' set. All items with "
                    "'depends_on=None' must be at the start of the list."
                )
        else:
            reached_dependent = True

    # Condition 6: For elements with depends_on set, it must
    # refer to an item that comes before them
    for idx, item in enumerate(value):
        depends_on = item.depends_on
        if depends_on is not None:
            if depends_on not in task_id_to_index:
                raise ValueError(
                    f"Item '{item.description}' (position {idx + 1}) has 'depends_on={depends_on}', which does not correspond to any 'task_id' in the workflow."
                )
            depends_on_idx = task_id_to_index[depends_on]
            if depends_on_idx >= idx:
                raise ValueError(
                    f"Item '{item.description}' (position {idx + 1}) has "
                    f"'depends_on={depends_on}', which refers to an item"
                    " that comes after it in the workflow. 'depends_on' must"
                    " refer to a preceding 'task_id'."
                )

    # Condition 7: MRMR with FROM_DEPENDENCY relevance requires a dependency
    for item in value:
        if isinstance(item, Mrmr) and item.relevance_type == MRMRRelevance.FROM_DEPENDENCY and item.depends_on is None:
            raise ValueError(
                f"MRMR task (task_id={item.task_id}) uses FROM_DEPENDENCY relevance "
                "but has no dependency. It requires feature importance from an upstream task."
            )

    # Condition 8: MRMR fi_method must be available from the upstream task
    for item in value:
        if not (isinstance(item, Mrmr) and item.relevance_type == MRMRRelevance.FROM_DEPENDENCY):
            continue
        if item.depends_on is None:
            continue  # already caught by condition 7
        upstream = task_id_to_task[item.depends_on]
        requested = item.feature_importance_method
        available = _get_fi_methods(upstream)
        if available is not None and requested not in available:
            raise ValueError(
                f"MRMR task (task_id={item.task_id}) requests fi_method={requested.value!r} "
                f"but upstream task (task_id={upstream.task_id}) only produces: "
                f"{[m.value for m in available]}."
            )

    # Condition 9: MRMR with F_STATISTICS relevance is not supported for time-to-event
    if _instance.ml_type == MLType.TIMETOEVENT:
        for item in value:
            if isinstance(item, Mrmr) and item.relevance_type == MRMRRelevance.F_STATISTICS:
                raise ValueError(
                    f"MRMR task (task_id={item.task_id}) uses F_STATISTICS relevance "
                    "which is not supported for time-to-event studies. Use FROM_DEPENDENCY instead."
                )


def _get_fi_methods(task: Task) -> list[FIComputeMethod] | None:
    """Return the FI methods a task produces, or None if unknown."""
    from octopus.modules.autogluon import AutoGluon  # noqa: PLC0415
    from octopus.modules.tako import Tako  # noqa: PLC0415

    if isinstance(task, Tako):
        return task.fi_methods
    if isinstance(task, AutoGluon):
        return [FIComputeMethod.PERMUTATION]
    return None

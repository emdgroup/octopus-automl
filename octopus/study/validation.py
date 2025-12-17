"""Validation functions for OctoStudy attributes."""

from typing import TYPE_CHECKING

from attrs import Attribute

from octopus.task import Task

if TYPE_CHECKING:
    from octopus.study.core import OctoStudy


def validate_start_with_empty_study(instance: "OctoStudy", attribute: Attribute, value: bool) -> None:
    """Validate if start_with_empty_study is consistent with workflow tasks.

    Args:
        instance: The OctoStudy instance being validated.
        attribute: The attribute being validated.
        value: The value of start_with_empty_study.

    Raises:
        ValueError: If start_with_empty_study=True but workflow contains tasks with load_task=True.
    """
    if value:  # start_with_empty_study=True
        for task in instance.workflow:
            if task.load_task:
                raise ValueError(
                    f"Cannot set start_with_empty_study=True when workflow contains tasks with load_task=True. "
                    f"Task '{task.description}' (task_id={task.task_id}) has load_task=True. "
                    f"Either set start_with_empty_study=False or remove load_task from all workflow tasks."
                )


def validate_workflow(_instance: "OctoStudy", attribute: Attribute, value: list[Task]) -> None:
    """Validate the `workflow` attribute.

    Ensures that the value is a non-empty list where each item is an
    instance of `Task`, and that the workflow meets specified
    conditions.

    Conditions:
    - The first task must have `task_id=0`.
    - All items with `depends_on_task=-1` must be at the start of the list,
      before any other items with `depends_on_task >= 0`.
    - All elements in the list must be in increasing order of `task_id`.
    - For elements with `depends_on_task >= 0`, their `depends_on_task` must
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

    # Build mapping of task_id to index and collect task_ids
    task_id_to_index = {}
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

    # Condition 4: All items with depends_on_task=-1 must be at the start of the list
    reached_non_negative_depends_on_task = False
    for idx, item in enumerate(value):
        if item.depends_on_task == -1:
            if reached_non_negative_depends_on_task:
                raise ValueError(
                    f"Item at position {idx + 1} has 'depends_on_task=-1' but"
                    " appears after items with 'depends_on_task>=0'. All items with "
                    "'depends_on_task=-1' must be at the start of the list."
                )
        else:
            reached_non_negative_depends_on_task = True

    # Condition 6: For elements with depends_on_task >= 0, depends_on_task must
    # refer to an item that comes before them
    for idx, item in enumerate(value):
        depends_on_task = item.depends_on_task
        if depends_on_task >= 0:
            if depends_on_task not in task_id_to_index:
                raise ValueError(
                    f"Item '{item.description}' (position {idx + 1}) has "
                    f"'depends_on_task={depends_on_task}', which does not"
                    " correspond to any 'task_id' in the workflow."
                )
            depends_on_taskx = task_id_to_index[depends_on_task]
            if depends_on_taskx >= idx:
                raise ValueError(
                    f"Item '{item.description}' (position {idx + 1}) has "
                    f"'depends_on_task={depends_on_task}', which refers to an item"
                    " that comes after it in the workflow. 'depends_on_task' must"
                    " refer to a preceding 'task_id'."
                )

import importlib

import inspect
from typing import List

from .task import Task
from .modality import Modality, Text, is_modal, Audio, Motion

task_lib = importlib.import_module('.task_lib', package=__name__)
modalities = importlib.import_module('.modality', package=__name__)

all_tasks = [
    getattr(task_lib, name) for name in dir(task_lib)
    if inspect.isclass(getattr(task_lib, name)) and issubclass(getattr(task_lib, name), Task)
]

all_modalities: List[Modality] = [
    getattr(modalities, name) for name in dir(modalities)
    if inspect.isclass(getattr(modalities, name)) and issubclass(getattr(modalities, name), Modality)
       and getattr(modalities, name) not in [Modality, Text]
]
locatable_modalities = [modal for modal in all_modalities if modal.locatable()]
abbr_task_mapping = {task.abbr: task for task in all_tasks}
name_modal_mapping = {modal.name: modal for modal in all_modalities}


def abbr_list_to_task_list(abbr: List[str]) -> List[Task]:
    if isinstance(abbr, str):
        abbr = [abbr]
    assert isinstance(abbr, list), f'input list please, got {abbr}'
    return [abbr_task_mapping[a] for a in abbr]

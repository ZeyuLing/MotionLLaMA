from abc import ABC


class Task(ABC):
    abbr = None
    description = 'description of the task'
    templates = None
    num_persons = 1
    input_modality = []
    output_modality = []

    @classmethod
    def all_modality(cls):
        return cls.input_modality + cls.output_modality


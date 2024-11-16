from random import sample
from typing import Dict, List

from mmotion.utils.task import Task
from mmotion.utils.task.prompt.prompt_template.system_instructions import sample_sys_template
from mmotion.utils.task.prompt.prompt_template.single_text_motion.n2m import N2M_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.single_text_motion.t2m import T2M_TEMPLATE
from mmotion.utils.task.prompt.prompt_template.completion.pred import PRED_TEMPLATE


def sample_template(task: Task):
    templates = task.templates
    assert isinstance(templates, list), (templates, task)
    template = sample(templates, 1)[0]
    input, output = template.get('input', None), template['output']
    if input is not None:
        message = [
            # {'role': 'system', 'content': sample_sys_template()},
            {'role': 'user', 'content': input},
            {'role': 'assistant', 'content': output}
        ]
    else:
        message = [{'role': 'assistant', 'content': output}]
    return message


def get_system_user(message: List[Dict]):
    system_user = [role_content for role_content in message if role_content['role'] in ['system', 'user']]
    if len(system_user) == 0:
        return [{'role': 'system', 'content': ''},
                {'role': 'user', 'content': ''}]
    return system_user


def get_assistant(message: List[Dict]):
    assistant = [role_content for role_content in message if role_content['role'] == 'assistant']

    return assistant

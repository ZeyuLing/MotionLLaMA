import json
from typing import List, Dict, Union


def write_json(save_path: str, data):
    """
    :param save_path: save path of json
    :param data: object need to write to json
    :return: None
    """
    with open(save_path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)


def read_json(path: str) -> Union[List, Dict]:
    """
    :param save_path: save path of json
    :return: object read from json
    """
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data

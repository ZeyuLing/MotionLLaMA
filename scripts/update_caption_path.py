from os.path import exists, join
from typing import Union, List, Dict

import fire
import json

from tqdm import tqdm


def read_json(path: str) -> Union[List, Dict]:
    """
    :param save_path: save path of json
    :return: object read from json
    """
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data
def write_json(save_path: str, data):
    """
    :param save_path: save path of json
    :param data: object need to write to json
    :return: None
    """
    with open(save_path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)

def main(json_path:str='data/motionhub/all.json',
         data_root:str='data/motionhub'):
    data_info = read_json(json_path)
    for key, info in tqdm(data_info['data_list'].items()):
        if 'caption_path' in info:
            continue
        smplx_path = info['smplx_path']
        caption_path = smplx_path.replace('standard_smplx', 'caption').replace('.npz', '.txt')
        if exists(join(data_root, caption_path)):
            print(caption_path)
            info['caption_path'] = caption_path
            data_info['data_list'][key]=info
            write_json(json_path, data_info)

if __name__=='__main__':
    fire.Fire(main)
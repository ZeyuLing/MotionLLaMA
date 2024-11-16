import numpy as np
import os
import sys
from collections import defaultdict
from os.path import join, dirname, relpath
from typing import Dict

sys.path.append(os.curdir)
from mmotion.utils.files_io.json import write_json
from mmotion.utils.files_io.txt import read_list_txt


def get_anno(data_dir: str = 'data/motionhub/cs3d'):
    """
    :param data_dir: cs3d dataset path
    :return: a train.json and a test.json file for cs3d dataset
    """

    train_subset = process_subset('data/motionhub/cs3d/splits/train.txt')
    val_subset = process_subset('data/motionhub/cs3d/splits/val.txt')
    test_subset = process_subset('data/motionhub/cs3d/splits/test.txt')

    write_json(join(data_dir, 'train.json'), train_subset)
    write_json(join(data_dir, 'val.json'), val_subset)
    write_json(join(data_dir, 'test.json'), test_subset)


def process_subset(split_path: str) -> Dict:
    """
    :param split_path: path of motionx dataset
    :return:
    """
    data_root = dirname(dirname(dirname(split_path)))
    data_info = {
        'meta_info': {'dataset': 'cs3d',
                      'version': 'v1.0'},
        'data_list': defaultdict(),
    }

    index_list = read_list_txt(split_path)
    for index in index_list:
        smpl_path = join(data_root, f'cs3d/pkl/{index}.pkl')
        data = np.load(smpl_path, allow_pickle=True)
        num_frames = len(data['smpl_trans'])
        duration = num_frames / 20
        audio_npy_path = smpl_path.replace('.pkl', '.mp4').replace('pkl', 'audio')
        sample_info = {
            'duration': duration,
            'smpl_path': relpath(smpl_path, data_root),
            'fps': 20,
            'audio_path': relpath(audio_npy_path, data_root),
            'has_hand': False,
            'sr': 16000
        }
        data_info['data_list'][index] = sample_info
    return data_info


if __name__ == '__main__':
    get_anno()

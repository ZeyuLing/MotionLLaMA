import numpy as np
import os
import sys
from collections import defaultdict
from os.path import join, dirname, relpath, exists
from typing import Dict

from tqdm import tqdm

sys.path.append(os.curdir)
from mmotion.utils.files_io.txt import read_list_txt
from mmotion.utils.files_io.json import write_json
invalid_list = [

]


def get_anno(data_dir: str = 'data/motionhub/motionx'):
    """
    :param data_dir: motionx dataset path
    :return: a train.json and a test.json file for motionx dataset
    """

    train_subset = process_subset('data/motionhub/motionx/splits/train.txt')
    val_subset = process_subset('data/motionhub/motionx/splits/val.txt')
    test_subset = process_subset('data/motionhub/motionx/splits/test.txt')

    write_json(join(data_dir, 'train.json'), train_subset)
    write_json(join(data_dir, 'val.json'), val_subset)
    write_json(join(data_dir, 'test.json'), test_subset)


def process_subset(split_path: str) -> Dict:
    """
    :param split_path: path of motionx dataset
    :return:
    """
    # data/motionhub
    missing_list = []
    data_root = dirname(dirname(dirname(split_path)))
    data_info = {
        'meta_info': {'dataset': 'motionx',
                      'version': 'v1.1'},
        'data_list': defaultdict(),
    }

    index_list = read_list_txt(split_path)
    for index in tqdm(index_list):
        smplx_path = join(data_root, 'motionx/motion_data/standard_smplx', f'{index}.npz')
        if not os.path.exists(smplx_path):
            missing_list.append(smplx_path)
            continue
        data = np.load(smplx_path, allow_pickle=True)
        num_frames = data['num_frames']

        face_smplx_path = smplx_path.replace('motion_data', 'face_motion_data')
        txt_path = smplx_path.replace('motion_data/standard_smplx', 'caption').replace('.npz', '.txt')

        sample_info = {
            'duration': num_frames / 30,
            'smplx_path': relpath(smplx_path, data_root),
            'raw_path': relpath(smplx_path, data_root).replace('standard_smplx', 'smplx_322'),
            'fps': 30,
            'caption_path': relpath(txt_path, data_root),
            'has_hand': True,
            'face_smplx_path': relpath(face_smplx_path, data_root),
            'hm3d_path': relpath(smplx_path, data_root).replace('standard_smplx', 'humanml3d').replace('.npz', '.npy'),
            'interhuman_path': relpath(smplx_path, data_root).replace('standard_smplx', 'interhuman').replace('.npz', '.npy'),
            'uniform_joints_path': relpath(smplx_path, data_root).replace('standard_smplx', 'uniform_joints').replace('.npz', '.npy'),
            'global_uniform_joints_path': relpath(smplx_path, data_root).replace('standard_smplx', 'global_uniform_joints').replace('.npz', '.npy'),

        }

        hm3d_path = join(data_root, sample_info['hm3d_path'])
        if exists(hm3d_path):
            hm3d = np.load(hm3d_path)
        if not exists(hm3d_path) or np.any(np.isnan(hm3d)):
            sample_info['invalid'] = True
        data_info['data_list'][index] = sample_info
    print('missing_list:', missing_list, f'{len(missing_list)} missed in total')
    return data_info


def get_train_test_dict(split_csv):
    return split_csv.set_index('id')['type'].to_dict()


if __name__ == '__main__':
    get_anno()

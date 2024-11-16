import numpy as np
import os
import sys
from collections import defaultdict
from os.path import join, dirname, relpath
from typing import Dict

from tqdm import tqdm

sys.path.append(os.curdir)
from mmotion.utils.files_io.json import write_json
from mmotion.utils.files_io.txt import read_list_txt

invalid_list = ['G039T006A021R006']


def get_anno(data_dir: str = 'data/motionhub/interx'):
    """
    :param data_dir: interx dataset path
    :return: a train.json and a test.json file for interx dataset
    """

    train_subset = process_subset('data/motionhub/interx/splits/train.txt')
    val_subset = process_subset('data/motionhub/interx/splits/val.txt')
    test_subset = process_subset('data/motionhub/interx/splits/test.txt')

    write_json(join(data_dir, 'train.json'), train_subset)
    write_json(join(data_dir, 'val.json'), val_subset)
    write_json(join(data_dir, 'test.json'), test_subset)


def process_subset(split_path: str) -> Dict:
    """
    :param split_path: path of motionx dataset
    :return:
    """
    interact_order = np.load('data/motionhub/interx/annots/interaction_order.pkl', allow_pickle=True)
    data_root = dirname(dirname(dirname(split_path)))
    data_info = {
        'meta_info': {'dataset': 'InterX',
                      'version': 'v1.0'},
        'data_list': defaultdict(),
    }

    index_list = read_list_txt(split_path)
    for index in tqdm(index_list):
        smplx_path_p1 = join(data_root, f'interx/standard_smplx/{index}/P1.npz')
        smplx_path_p2 = join(data_root, f'interx/standard_smplx/{index}/P2.npz')
        union_caption_path = f'interx/texts/{index}.txt'
        smplx_p1 = np.load(smplx_path_p1, allow_pickle=True)
        num_frames = smplx_p1['num_frames']
        fps = 120
        interact = interact_order[index]
        p1_info = {
            'duration': num_frames / fps,
            'num_frames': num_frames,
            'smplx_path': relpath(smplx_path_p1, data_root),
            'raw_path': relpath(smplx_path_p1, data_root).replace('standard_smplx', 'motions'),
            'caption_path': relpath(smplx_path_p1, data_root).replace('standard_smplx', 'sep_texts').replace('.npz',
                                                                                                             '.txt'),
            'union_caption_path': union_caption_path,
            'fps': fps,
            'has_hand': True,
            'actor': True if interact == 0 else False,
            'interactor_key': f'{index}_p2',
            'hm3d_path': relpath(smplx_path_p1, data_root).replace('standard_smplx', 'humanml3d').replace('.npz',
                                                                                                          '.npy'),
            'interhuman_path': relpath(smplx_path_p1, data_root).replace('standard_smplx', 'interhuman').replace('.npz',
                                                                                                                 '.npy'),
            'uniform_joints_path': relpath(smplx_path_p1, data_root).replace('standard_smplx',
                                                                             'uniform_joints').replace('.npz', '.npy'),
            'global_uniform_joints_path': relpath(smplx_path_p1, data_root).replace('standard_smplx',
                                                                                    'global_uniform_joints').replace(
                '.npz', '.npy'),
        }

        p2_info = {
            'duration': num_frames / fps,
            'num_frames': num_frames,
            'smplx_path': relpath(smplx_path_p2, data_root),
            'raw_path': relpath(smplx_path_p2, data_root).replace('standard_smplx', 'motions'),
            'caption_path': relpath(smplx_path_p2, data_root).replace('standard_smplx', 'sep_texts').replace('.npz',
                                                                                                             '.txt'),
            'union_caption_path': union_caption_path,
            'fps': fps,
            'has_hand': False,
            'actor': True if interact == 1 else False,
            'interactor_key': f'{index}_p1',
            'hm3d_path': relpath(smplx_path_p2, data_root).replace('standard_smplx', 'humanml3d').replace('.npz',
                                                                                                          '.npy'),
            'interhuman_path': relpath(smplx_path_p2, data_root).replace('standard_smplx', 'interhuman').replace('.npz',
                                                                                                                 '.npy'),
            'uniform_joints_path': relpath(smplx_path_p2, data_root).replace('standard_smplx',
                                                                             'uniform_joints').replace('.npz', '.npy'),
            'global_uniform_joints_path': relpath(smplx_path_p2, data_root).replace('standard_smplx',
                                                                                    'global_uniform_joints').replace(
                '.npz', '.npy'),
        }
        if index in invalid_list:
            p1_info['invalid'] = True
            p2_info['invalid'] = True
        data_info['data_list'][f'{index}_p1'] = p1_info
        data_info['data_list'][f'{index}_p2'] = p2_info

    return data_info


if __name__ == '__main__':
    get_anno()

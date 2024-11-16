import numpy as np
import os
import sys
from os.path import join, dirname, normpath, isdir, exists
from typing import List, Dict, Union

import fire
from tqdm import tqdm

sys.path.append(os.curdir)
from mmotion.utils.files_io.pickle import save_pickle

from mmotion.utils.files_io.json import read_json


def calculate_statistics(vec: np.ndarray, num_joints: int = 52):
    """
    :param vec: concatenated all vec, [t,c]
    :return:
    """
    print('Wait a moment, calculating statistics...')
    position_end = num_joints * 3
    vel_begin = num_joints * 3
    vel_end = num_joints * 6
    cont6d_begin = num_joints * 6
    cont6d_end = num_joints * 12

    mean = vec.mean(axis=0)
    pos_min = vec[..., :position_end].min(axis=0)
    pos_max = vec[..., :position_end].max(axis=0)
    pos_mean = mean[:position_end]
    vel_min = vec[..., vel_begin:vel_end].min(axis=0)
    vel_max = vec[..., vel_begin:vel_end].max(axis=0)
    vel_mean = mean[vel_begin:vel_end]
    rot_min = vec[..., cont6d_begin:cont6d_end].min(axis=0)
    rot_max = vec[..., cont6d_begin:cont6d_end].max(axis=0)
    rot_mean = mean[cont6d_begin:cont6d_end]
    fc_mean = mean[-4:]
    raw_std = vec.std(axis=0)


    pos_std = raw_std[: position_end]
    vel_std = raw_std[vel_begin:vel_end]
    # root always rotates around y-axis, some channels are always 0.
    rot_std = raw_std[cont6d_begin: cont6d_end].mean(keepdims=True).repeat(cont6d_end-cont6d_begin)

# feet contact
    fc_std = raw_std[-4:].mean(keepdims=True).repeat(4)

    # split pos, rot, vel into body and hands
    stats = {
        'mean': mean,
        'raw_std': raw_std,
        'pos_std': pos_std,
        'rot_std': rot_std,
        'vel_std': vel_std,
        'fc_std': fc_std,
        'pos_min': pos_min,
        'pos_max': pos_max,
        'pos_mean':pos_mean,
        'vel_min': vel_min,
        'vel_max': vel_max,
        'vel_mean': vel_mean,
        'rot_min': rot_min,
        'rot_max': rot_max,
        'rot_mean': rot_mean,
        'fc_mean': fc_mean
    }
    if num_joints > 24:
        pos_body_std = raw_std[: 22 * 3]
        pos_hand_std = raw_std[22 * 3:position_end]
        vel_body_std = raw_std[vel_begin: vel_begin + 22 * 3]
        vel_hand_std = raw_std[vel_begin + 22 * 3: vel_end]
        rot_body_std = raw_std[cont6d_begin: cont6d_begin + 22 * 6]
        rot_hand_std = raw_std[cont6d_begin + 22 * 6:cont6d_end]

        stats.update({
            'pos_body_std': pos_body_std,
            'pos_hand_std': pos_hand_std,
            'vel_body_std': vel_body_std,
            'vel_hand_std': vel_hand_std,
            'rot_body_std': rot_body_std,
            'rot_hand_std': rot_hand_std
        })

    return stats


def get_vec(data_list: Dict, data_root):
    vec_list = []
    for sample in tqdm(data_list.values(), 'loading all vec'):
        vec_path = join(data_root, sample['interhuman_path'])
        if not exists(vec_path):
            print('{} does not exist.'.format(vec_path))
            continue
        vec = np.load(vec_path)
        if np.any(np.isnan(vec)):
            print(f'{vec_path} is nan')
            continue
        vec_list.append(vec)
    return vec_list


def merge_vec(dict_list: List):
    return np.concatenate(dict_list, axis=0)


def load_data_list(anno_file: Union[str, List[str]]):
    if isinstance(anno_file, str):
        if isdir(anno_file):
            train_anno = join(anno_file, 'train.json')
            val_anno = join(anno_file, 'val.json')
            test_anno = join(anno_file, 'test.json')
            anno_file = []
            for anno in [train_anno, val_anno, test_anno]:
                if exists(anno):
                    anno_file.append(anno)
        else:
            anno_file = [anno_file]
    data_list = {}
    for anno in anno_file:
        anno = read_json(anno)['data_list']
        data_list.update(anno)

    return data_list


def main(anno_file: Union[str, List[str]] = ['data/motionhub/train.json',
                                             'data/motionhub/val.json',
                                             'data/motionhub/test.json'],
         data_root: str = 'data/motionhub'):
    data_list = load_data_list(anno_file)
    vec_list = get_vec(data_list, data_root)

    vec = merge_vec(vec_list)
    if isinstance(anno_file, list):
        save_root = dirname(anno_file[0])
    else:
        if isdir(anno_file):
            save_root = anno_file
        else:
            save_root = dirname(anno_file)
    save_root = join(save_root, 'statistics')
    os.makedirs(save_root, exist_ok=True)

    stat = calculate_statistics(vec)
    save_pickle(stat, join(save_root, 'interhuman.pkl'))
    print(stat)


if __name__ == '__main__':
    fire.Fire(main)

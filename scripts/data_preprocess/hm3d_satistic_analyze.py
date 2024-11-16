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


def calculate_statistics(hm3d: np.ndarray, num_joints: int = 52):
    """
    :param hm3d: concatenated all hm3d, [t,c]
    :return:
    """
    print('Wait a moment, calculating statistics...')
    position_begin = 4
    position_end = 4 + (num_joints - 1) * 3
    cont6d_begin = 4 + (num_joints - 1) * 3
    cont6d_end = 4 + (num_joints - 1) * 9
    vel_begin = 4 + (num_joints - 1) * 9
    vel_end = 4 + (num_joints - 1) * 9 + num_joints * 3

    mean = hm3d.mean(axis=0)
    root_min = hm3d[..., :4].min(axis=0)
    root_max = hm3d[..., :4].max(axis=0)
    pos_min = hm3d[..., position_begin:position_end].min(axis=0)
    pos_max = hm3d[..., position_begin:position_end].max(axis=0)
    vel_min = hm3d[..., vel_begin:vel_end].min(axis=0)
    vel_max = hm3d[..., vel_begin:vel_end].max(axis=0)
    rot_min = hm3d[..., cont6d_begin:cont6d_end].min(axis=0)
    rot_max = hm3d[..., cont6d_begin:cont6d_end].max(axis=0)
    raw_std = hm3d.std(axis=0)

    # angular speed
    root_a_std = raw_std[0:1]
    # root position
    root_xz_std = raw_std[1:3].mean(keepdims=True).repeat(2)
    root_y_std = raw_std[3:4]

    pos_std = raw_std[position_begin: position_end].mean(keepdims=True).repeat(position_end - position_begin)
    rot_std = raw_std[cont6d_begin: cont6d_end].mean(keepdims=True).repeat(cont6d_end - cont6d_begin)
    vel_std = raw_std[vel_begin:vel_end].mean(keepdims=True).repeat(vel_end - vel_begin)

    # feet contact
    fc_std = raw_std[-4:].mean(keepdims=True).repeat(4)

    # split pos, rot, vel into body and hands
    stats = {
        'mean': mean,
        'raw_std': raw_std,
        'root_a_std': root_a_std,
        'root_xz_std': root_xz_std,
        'root_y_std': root_y_std,
        'pos_std': pos_std,
        'rot_std': rot_std,
        'vel_std': vel_std,
        'fc_std': fc_std,
        'root_min': root_min,
        'root_max': root_max,
        'pos_min': pos_min,
        'pos_max': pos_max,
        'vel_min': vel_min,
        'vel_max': vel_max,
        'rot_min': rot_min,
        'rot_max': rot_max
    }
    if num_joints > 24:
        pos_body_std = raw_std[position_begin: position_begin + 21 * 3].mean(keepdims=True).repeat(21 * 3)
        pos_hand_std = raw_std[position_begin + 21 * 3:position_end].mean(keepdims=True).repeat(30 * 3)
        vel_body_std = raw_std[vel_begin: vel_begin + 22 * 3].mean(keepdims=True).repeat(22 * 3)
        vel_hand_std = raw_std[vel_begin + 22 * 3:vel_end].mean(keepdims=True).repeat(30 * 3)
        rot_body_std = raw_std[cont6d_begin: cont6d_begin + 21 * 6].mean(keepdims=True).repeat(21 * 6)
        rot_hand_std = raw_std[cont6d_begin + 21 * 6:cont6d_end].mean(keepdims=True).repeat(30 * 6)

        stats.update({
            'pos_body_std': pos_body_std,
            'pos_hand_std': pos_hand_std,
            'vel_body_std': vel_body_std,
            'vel_hand_std': vel_hand_std,
            'rot_body_std': rot_body_std,
            'rot_hand_std': rot_hand_std
        })

    return stats


def get_hm3d(data_list: Dict, data_root):
    hm3d_list = []
    for sample in tqdm(data_list.values(), 'loading all hm3d'):
        hm3d_path = join(data_root, sample['hm3d_path'])
        if not exists(hm3d_path):
            print('{} does not exist.'.format(hm3d_path))
            continue
        hm3d_vec = np.load(hm3d_path)
        if np.any(np.isnan(hm3d_vec)):
            print(f'{hm3d_path} is nan')
            continue
        hm3d_list.append(hm3d_vec)
    return hm3d_list


def merge_hm3d(dict_list: List):
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
    hm3d_list = get_hm3d(data_list, data_root)

    hm3d = merge_hm3d(hm3d_list)
    if isinstance(anno_file, list):
        save_root = dirname(anno_file[0])
    else:
        if isdir(anno_file):
            save_root = anno_file
        else:
            save_root = dirname(anno_file)
    save_root = join(save_root, 'statistics')
    os.makedirs(save_root, exist_ok=True)

    stat = calculate_statistics(hm3d)
    save_pickle(stat, join(save_root, 'humanml3d.pkl'))
    print(stat)


if __name__ == '__main__':
    fire.Fire(main)

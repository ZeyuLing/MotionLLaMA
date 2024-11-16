import numpy as np
import os
import sys
from collections import defaultdict
from glob import glob
from os.path import join, dirname, normpath, isdir, exists
from typing import List, Dict, Union

import fire
import pandas as pd
import torch
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.curdir)
from mmotion.utils.files_io.pickle import save_pickle
from mmotion.utils.geometry.rotation_convert import rot_convert

from mmotion.utils.smpl_utils.smpl_key_const import ROTATION_KEYS, TRANSL, JOINTS, SMPLX_KEYS, NUM_JOINTS_OF_ROT, \
    LEFT_HAND_POSE, RIGHT_HAND_POSE

from mmotion.utils.files_io.json import read_json


def calculate_statistics(data_dict):
    stats = defaultdict()

    for key, array in tqdm(data_dict.items()):
        if len(array > 0):
            if key in ROTATION_KEYS or key == TRANSL or key == JOINTS:
                if key in ROTATION_KEYS:
                    j = NUM_JOINTS_OF_ROT[key]
                    array = rearrange(array, 'b (j c) -> b j c', j=j)
                axis = tuple(range(np.ndim(array)))[:-1]
                stats[key] = {
                    'mean': np.mean(array, axis=axis),
                    'std': np.std(array, axis=axis),
                    'min': np.min(array, axis=axis),
                    'max': np.max(array, axis=axis)
                }
        else:
            stats[key] = {
                'mean': 0,
                'std': 1.,
                'min': 0.,
                'max': 0.
            }

    return stats


def merge_annos(anno_file: List[str]):
    data_list = {}
    for path in anno_file:
        data_list.update(read_json(path)['data_list'])
    return data_list


def get_smplx(data_list: Dict, data_root):
    smplx_dict_list = []
    for sample in tqdm(data_list.values(), 'merging smplx'):
        has_hand = sample['has_hand']
        smplx_path = join(data_root, sample['smplx_path'])
        smplx_dict = np.load(smplx_path, allow_pickle=True)
        smplx_dict['has_hand'] = has_hand
        smplx_dict_list.append(smplx_dict)
    return smplx_dict_list


def merge_smplx(dict_list: List):
    keys = ROTATION_KEYS + [TRANSL, JOINTS]
    merged_smplx = {key: [] for key in keys}
    for smplx_dict in tqdm(dict_list, 'merging smplx'):
        for key, value in smplx_dict.items():
            if key not in keys:
                continue
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            if key in [LEFT_HAND_POSE, RIGHT_HAND_POSE] and not smplx_dict['has_hand']:
                continue
            merged_smplx[key].append(value)
    for key, value in merged_smplx.items():
        if len(value) > 0:
            merged_smplx[key] = np.concatenate(value, axis=0)

    return merged_smplx


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


def transfer_rot_type(smplx_aa: Dict, rot_type: str):
    """

    :param smplx_aa: every rot param is in shape t,c where t is merged from all samples
    :return: smplx dict in rot_type
    """

    smplx_new_type = {}
    for key, value in smplx_aa.items():
        if key not in ROTATION_KEYS:
            smplx_new_type[key] = value
        else:
            value = rearrange(value, 't (j c) -> t j c', c=3)
            value = rot_convert(value, 'axis_angle', rot_type)
            value = rearrange(value, 't j c -> t (j c)')
            smplx_new_type[key] = value
    return smplx_new_type


def main(anno_file: Union[str, List[str]] = ['data/motionhub/train.json',
                                             'data/motionhub/val.json',
                                             'data/motionhub/test.json'],
         data_root: str = 'data/motionhub'):
    data_list = load_data_list(anno_file)
    smplx_list = get_smplx(data_list, data_root)

    merge_data_axis_angle = merge_smplx(smplx_list)
    merge_data_rotation_6d = transfer_rot_type(merge_data_axis_angle, 'rotation_6d')
    merge_data_quaternion = transfer_rot_type(merge_data_axis_angle, 'quaternion')

    if isinstance(anno_file, list):
        save_root = dirname(anno_file[0])
    else:
        if isdir(anno_file):
            save_root = anno_file
        else:
            save_root = dirname(anno_file)
    save_root = join(save_root, 'statistics')
    os.makedirs(save_root, exist_ok=True)

    stat_aa = calculate_statistics(merge_data_axis_angle)
    save_pickle(stat_aa, join(save_root, 'axis_angle.pkl'))
    print(stat_aa)

    stat_q = calculate_statistics(merge_data_quaternion)
    save_pickle(stat_q, join(save_root, 'quaternion.pkl'))
    print(stat_q)

    stat_rot6d = calculate_statistics(merge_data_rotation_6d)
    save_pickle(stat_rot6d, join(save_root, 'rotation_6d.pkl'))
    print(stat_rot6d)


if __name__ == '__main__':
    fire.Fire(main)

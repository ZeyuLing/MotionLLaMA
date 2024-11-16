import numpy as np
import os
import sys
from collections import defaultdict
from glob import glob
from os.path import join
from typing import List

import pandas as pd
import torch
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.curdir)
from mmotion.utils.smpl_utils.smpl_key_const import ROTATION_KEYS, TRANSL, JOINTS


def merge_smplx(dict_list: List):
    merged_dict = defaultdict()
    example = dict_list[0]
    for key, value in example.items():
        if isinstance(value, np.ndarray):
            merged_dict[key] = np.concatenate([sample[key] for sample in dict_list], axis=0)
        elif isinstance(value, torch.Tensor):
            merged_dict[key] = np.concatenate([sample[key].numpy() for sample in dict_list], axis=0)
    return merged_dict


def calculate_statistics(data_dict):

    stats_df = pd.DataFrame(columns=['mean', 'std', 'min', 'max'], index=ROTATION_KEYS + [TRANSL, JOINTS])

    for key, array in tqdm(data_dict.items()):
        if key in ROTATION_KEYS or key == TRANSL or key == JOINTS:
            if array.shape[-1] != 3 and key in ROTATION_KEYS:
                array = rearrange(array, 'b (j c) -> b j c', c=3)
            axis = tuple(range(np.ndim(array)))[:-1]
            stats_df.at[key, 'mean'] = np.mean(array, axis=axis)
            stats_df.at[key, 'std'] = np.std(array, axis=axis)
            stats_df.at[key, 'min'] = np.min(array, axis=axis)
            stats_df.at[key, 'max'] = np.max(array, axis=axis)

    return stats_df


def analyze(subset_path: str):
    smplx_list = []
    for root, dirs, files in os.walk(subset_path):
        for file in tqdm(files):
            if file.endswith(".npz"):
                smplx_dict = np.load(join(root, file), allow_pickle=True)
                smplx_list.append(smplx_dict)
    merged_smplx = merge_smplx(smplx_list)
    stat_df = calculate_statistics(merged_smplx)
    stat_df.to_csv(join(subset_path, 'statistics.csv'))


if __name__ == '__main__':
    standard_smplx_root = 'data/motionhub/motionx/motion_data/standard_smplx'
    for subset in glob(join(standard_smplx_root, '*')):
        analyze(subset)

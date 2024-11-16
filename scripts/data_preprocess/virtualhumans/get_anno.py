import sys
from collections import defaultdict
from glob import glob
from os.path import join, basename, relpath, dirname

import fire
import os

import numpy as np

import torch
from tqdm import tqdm

sys.path.append(os.curdir)
from mmotion.utils.files_io.json import write_json


def get_ids(subset_path: str):
    id_list = []
    for filename in os.listdir(subset_path):
        id_list.append(filename[:-4])
    return id_list


def main(data_root: str = 'data/motionhub/virtualhumans'):
    train_data_info = {
        'meta_info': {'dataset': 'virtualhumans',
                      'version': 'v1'},
        'data_list': defaultdict()
    }
    val_data_info = {
        'meta_info': {'dataset': 'virtualhumans',
                      'version': 'v1'},
        'data_list': defaultdict()
    }
    test_data_info = {
        'meta_info': {'dataset': 'virtualhumans',
                      'version': 'v1'},
        'data_list': defaultdict()
    }
    train_list = get_ids(join(data_root, 'train'))
    val_list = get_ids(join(data_root, 'validation'))
    test_list = get_ids(join(data_root, 'test'))

    for smplx_path in tqdm(glob(join(data_root, 'standard_smplx/*.npz'))):
        idx = basename(smplx_path)[:-4]
        rel_smplx_path = relpath(smplx_path, dirname(data_root))
        smplx = np.load(smplx_path, allow_pickle=True)
        num_frames = smplx['num_frames']
        sample = {
            'duration': num_frames / 30,
            'num_frames': num_frames,
            'smplx_path': rel_smplx_path,
            'fps': 30,
            'has_hand': False,
            'caption_path': rel_smplx_path.replace('standard_smplx', 'caption').replace('.npz', '.txt'),
            'hm3d_path': rel_smplx_path.replace('standard_smplx', 'humanml3d').replace('.npz', '.npy'),
            'interhuman_path': rel_smplx_path.replace('standard_smplx', 'interhuman').replace('.npz', '.npy'),
            'uniform_joints_path': rel_smplx_path.replace('standard_smplx', 'uniform_joints').replace('.npz', '.npy'),
            'global_uniform_joints_path': rel_smplx_path.replace('standard_smplx', 'global_uniform_joints').replace(
                '.npz', '.npy'),
        }
        for key, value in smplx.items():
            if isinstance(value, torch.Tensor):
                if torch.any(torch.isnan(value)) or torch.any(torch.isinf(value)):
                    sample['invalid'] = True
                    print(idx)
                    break
        if idx in train_list:
            train_data_info['data_list'][idx] = sample
        elif idx in val_list:
            val_data_info['data_list'][idx] = sample
        elif idx in test_list:
            test_data_info['data_list'][idx] = sample
    write_json(join(data_root, 'train.json'), train_data_info)
    write_json(join(data_root, 'val.json'), val_data_info)
    write_json(join(data_root, 'test.json'), test_data_info)


if __name__ == '__main__':
    fire.Fire(main)

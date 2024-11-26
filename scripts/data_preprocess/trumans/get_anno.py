import numpy as np
import os
import sys
from collections import defaultdict
from glob import glob
from os.path import join, relpath, dirname, basename, exists

import torch
from datasets import tqdm

sys.path.append(os.curdir)

from mmotion.utils.files_io.json import write_json

ignore_list = []


def get_anno(data_dir: str = 'data/motionhub/trumans'):
    train_data_info = {
        'meta_info': {'dataset': 'trumans',
                      'version': 'v1'},
        'data_list': defaultdict()
    }

    for smplx_path in tqdm(glob(join(data_dir, 'standard_smplx', '*.npz'))):
        rel_smplx_path = relpath(smplx_path, dirname(data_dir))
        filename = basename(smplx_path).split('.')[0]
        smplx = np.load(smplx_path, allow_pickle=True)
        num_frames = smplx['num_frames']
        sample = {
            'duration': num_frames / 30.,
            'num_frames': num_frames,
            'smplx_path': rel_smplx_path,
            'fps': 30,
            'has_hand': True,
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
                    print(filename)
                    break

        if filename in ignore_list:
            sample['invalid'] = True
        vis_path = smplx_path.replace('standard_smplx', 'vis_video').replace('.npz', '.mp4')
        if exists(vis_path):
            sample['vis_path'] = rel_smplx_path.replace('standard_smplx', 'vis_video').replace('.npz', '.mp4')
        train_data_info['data_list'][filename] = sample

    write_json(join(data_dir, 'train.json'), train_data_info)


if __name__ == '__main__':
    get_anno()

import numpy as np
import os
import sys
from collections import defaultdict
from glob import glob
from os.path import join, relpath, dirname, basename, exists

from datasets import tqdm

sys.path.append(os.curdir)

from mmotion.utils.files_io.json import write_json

ignore_list = ['152']
test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092",
             "120", "037", "109", "204", "144"]


def get_anno(data_dir: str = 'data/motionhub/finedance'):
    train_data_info = {
        'meta_info': {'dataset': 'finedance',
                      'version': 'v1'},
        'data_list': defaultdict()
    }

    test_data_info = {
        'meta_info': {'dataset': 'finedance',
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
            'music_path': rel_smplx_path.replace('standard_smplx', 'music_wav').replace('.npz', '.wav'),
            'raw_path': rel_smplx_path.replace('standard_smplx', 'motion').replace('.npz', '.npy'),
            'fps': 30,
            'has_hand': True,
            'interhuman_path': rel_smplx_path.replace('standard_smplx', 'interhuman').replace('.npz', '.npy'),
            'sr': 48000,
            'caption_path': rel_smplx_path.replace('standard_smplx', 'caption').replace('.npz', '.txt'),
        }

        if filename in ignore_list:
            sample['invalid'] = True
        vis_path = smplx_path.replace('standard_smplx', 'vis_video').replace('.npz', '.mp4')
        if exists(vis_path):
            sample['vis_path'] = rel_smplx_path.replace('standard_smplx', 'vis_video').replace('.npz', '.mp4')
        if filename not in test_list:
            train_data_info['data_list'][f'finedance_{filename}'] = sample
        else:
            test_data_info['data_list'][f'finedance_{filename}'] = sample

    write_json(join(data_dir, 'train.json'), train_data_info)
    write_json(join(data_dir, 'test.json'), test_data_info)


if __name__ == '__main__':
    get_anno()

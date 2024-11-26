import os
import sys
from collections import defaultdict
from glob import glob
from os.path import join, relpath, dirname, basename, exists

import numpy as np
from tqdm import tqdm

sys.path.append(os.curdir)
from mmotion.utils.files_io.json import write_json
from mmotion.utils.files_io.txt import read_list_txt

ignore_list = []

genres_dict = {
    "gBR": "Break",
    "gPO": "Pop",
    "gLO": "Lock",
    "gMH": "Middle Hip-hop",
    "gLH": "LA style Hip-hop",
    "gHO": "House",
    "gWA": "Waack",
    "gKR": "Krump",
    "gJS": "Street Jazz",
    "gJB": "Ballet Jazz"
}


def get_information_from_name(filename: str):
    filename = filename.rstrip('.pkl')
    genre, situation, camera_id, dancer_id, music_id, choreo_id = filename.split('_')
    genre = genres_dict[genre]
    return genre


def get_anno(data_dir: str = 'data/motionhub/aist'):
    train_data_info = {
        'meta_info': {'dataset': 'aist++ train subset',
                      'version': 'v1'},
        'data_list': defaultdict()
    }

    test_data_info = {
        'meta_info': {'dataset': 'aist++ test subset',
                      'version': 'v1'},
        'data_list': defaultdict()
    }
    test_list = read_list_txt(join(data_dir, 'splits/test.txt'))

    for smplx_path in tqdm(glob(join(data_dir, 'standard_smplx', '*.npz'))):
        rel_smplx_path = relpath(smplx_path, dirname(data_dir))
        filename = basename(smplx_path).split('.')[0]

        smplx = np.load(smplx_path, allow_pickle=True)
        num_frames = smplx['num_frames']

        sample = {
            'duration': num_frames / 60.,
            'num_frames': num_frames,
            'smplx_path': rel_smplx_path,
            'music_path': rel_smplx_path.replace('standard_smplx', 'raw_music').replace('.npz', '.wav'),
            'raw_path': rel_smplx_path.replace('standard_smplx', 'motion').replace('.npz', '.npy'),
            'fps': 60.,
            'has_hand': False,
            'interhuman_path': rel_smplx_path.replace('standard_smplx', 'interhuman').replace('.npz', '.npy'),
            'caption_path': rel_smplx_path.replace('standard_smplx', 'caption').replace('.npz', '.txt'),
            'genre': get_information_from_name(filename),
            'sr': 48000
        }

        if filename in ignore_list:
            sample['invalid'] = True
        vis_path = smplx_path.replace('standard_smplx', 'vis_video').replace('.npz', '.mp4')
        if exists(vis_path):
            sample['vis_path'] = rel_smplx_path.replace('standard_smplx', 'vis_video').replace('.npz', '.mp4')
        if filename in test_list:
            test_data_info['data_list'][filename] = sample
        else:
            train_data_info['data_list'][filename] = sample

    write_json(join(data_dir, 'train.json'), train_data_info)
    write_json(join(data_dir, 'test.json'), test_data_info)

    print(len(train_data_info['data_list']) + len(test_data_info['data_list']))


if __name__ == '__main__':
    get_anno()

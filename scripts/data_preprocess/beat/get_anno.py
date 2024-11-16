import numpy as np
import os
import re
import sys
from collections import defaultdict
from glob import glob
from os.path import join, basename, dirname, relpath
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm
sys.path.append(os.curdir)
from mmotion.utils.dataset_utils.beat_utils import filename_to_speaker_id
from mmotion.utils.files_io.json import write_json


def get_anno(data_dir: str = 'data/motionhub/beat_v2.0.0'):
    """
    :param data_dir: beatv2 dataset path
    :return: a train.json and a test.json file for beatv2 dataset
    """
    train_data_info = {
        'meta_info': {'dataset': 'beat',
                      'version': 'v2'},
        'data_list': defaultdict(),
    }
    test_data_info = {
        'meta_info': {'dataset': 'beat',
                      'version': 'v2'},
        'data_list': defaultdict(),
    }
    for subset in glob(join(data_dir, '*')):
        if os.path.isdir(subset):
            train_subset, test_subset = process_subset(subset)
            train_data_info['data_list'].update(train_subset['data_list'])
            test_data_info['data_list'].update(test_subset['data_list'])

    print(f"{len(train_data_info['data_list'])} samples in train subset")
    print(f"{len(test_data_info['data_list'])} samples intest subset")
    write_json(join(data_dir, 'train.json'), train_data_info)
    write_json(join(data_dir, 'test.json'), test_data_info)


def process_subset(subset_path: str) -> Tuple[Dict, Dict]:
    """
    :param subset_path: path of beatv2 dataset
    :return:
    """
    data_root = dirname(dirname(subset_path))
    train_test_split = pd.read_csv(join(subset_path, 'train_test_split.csv'))
    train_test_split: Dict = get_train_test_dict(train_test_split)
    train_data_info = {
        'meta_info': {'dataset': 'beat',
                      'version': 'v2'},
        'data_list': defaultdict(),
    }
    test_data_info = {
        'meta_info': {'dataset': 'beat',
                      'version': 'v2'},
        'data_list': defaultdict(),
    }
    for smplx_path in tqdm(glob(join(subset_path, 'standard_smplx', '*.npz'))):
        data = np.load(smplx_path, allow_pickle=True)
        num_frames = data['num_frames']
        rel_smplx_path = relpath(smplx_path, data_root)
        filename = basename(smplx_path).split('.')[0]
        speaker_idx = filename_to_speaker_id(filename)
        key = 'beatv2_' + filename
        language = re.search(r'beat_(\w+)_v2\.0\.0', smplx_path).group(1)
        sample = {
            'duration': num_frames / 30,
            'num_frames': num_frames,
            'smplx_path': rel_smplx_path,
            'raw_path': rel_smplx_path.replace('standard_smplx', 'smplxflame_30'),
            'fps': 30,
            'speech_script_path': rel_smplx_path.replace('standard_smplx', 'text').replace('.npz', '.txt'),
            'audio_path': rel_smplx_path.replace('standard_smplx', 'wave16k').replace('.npz', '.wav'),
            'sr': 16000,
            'script_path': rel_smplx_path.replace('standard_smplx', 'textgrid').replace('.npz', '.TextGrid'),
            'has_hand': True,
            'hm3d_path': rel_smplx_path.replace('standard_smplx', 'humanml3d').replace('.npz', '.npy'),
            'interhuman_path': rel_smplx_path.replace('standard_smplx', 'interhuman').replace('.npz', '.npy'),
            'caption_path': rel_smplx_path.replace('standard_smplx', 'caption').replace('.npz', '.txt'),
            'language': language,
            'speaker_id': speaker_idx

        }

        if train_test_split.get(filename) == 'test':
            test_data_info['data_list'][key] = sample
        else:
            train_data_info['data_list'][key] = sample

    write_json(join(subset_path, 'train.json'), train_data_info)
    write_json(join(subset_path, 'test.json'), test_data_info)
    return train_data_info, test_data_info


def get_train_test_dict(split_csv):
    return split_csv.set_index('id')['type'].to_dict()


if __name__ == '__main__':
    get_anno()

import numpy as np
import os
import sys
from collections import defaultdict
from glob import glob
from os.path import join, relpath, dirname, basename, exists

from datasets import tqdm
sys.path.append(os.curdir)
from mmotion.utils.task import tasks_for_t2m_data, tasks_for_inter_data
from mmotion.utils.files_io.json import write_json
from mmotion.utils.task.task_lib import (
    PRETRAIN_RANDOM_MOTION_GENERATION, \
    PRETRAIN_RANDOM_INTER_MOTION_GENERATION,
    PRETRAIN_RANDOM_TEXT_MOTION_GENERATION,
    PRETRAIN_RANDOM_INTER_SEPARATE_TEXT_MOTION_GENERATION, PRETRAIN_RANDOM_INTER_UNION_TEXT_MOTION_GENERATION)


def get_anno(data_dir: str = 'data/motionhub/chi3d'):
    train_data_info = {
        'meta_info': {'dataset': 'chi3d',
                      'version': 'v1'},
        'data_list': defaultdict()
    }

    for smplx_path in tqdm(glob(join(data_dir, 'train/s*/standard_smplx/*/*.npz'))):
        rel_smplx_path = relpath(smplx_path, dirname(data_dir))
        txt_path = rel_smplx_path.replace('standard_smplx', 'caption').replace('.npz', '.txt')
        union_caption_path = (txt_path.replace('caption', 'union_caption')
                              .replace('/P1.txt', '.txt').replace('/P2.txt', '.txt'))
        subset_name = basename(dirname(dirname(dirname(smplx_path))))
        index = basename(dirname(smplx_path))
        person_id = basename(smplx_path).split('.')[0]
        smplx = np.load(smplx_path, allow_pickle=True)
        num_frames = smplx['num_frames']
        interactor_key = f'{subset_name}_{index}_P1' if person_id == 'P2' else f'{subset_name}_{index}_P2'
        sample = {
            'duration': num_frames / 50,
            'num_frames': num_frames,
            'smplx_path': rel_smplx_path,
            'caption_path': txt_path,
            'union_caption_path': union_caption_path,
            'raw_path': rel_smplx_path.replace('standard_smplx', 'smplx').replace('.npz', '.pkl'),
            'fps': 50,
            'has_hand': True,
            'interactor_key': interactor_key,
            'hm3d_path': rel_smplx_path.replace('standard_smplx', 'humanml3d').replace('.npz', '.npy'),
            'interhuman_path': rel_smplx_path.replace('standard_smplx', 'interhuman').replace('.npz', '.npy'),
            'uniform_joints_path': rel_smplx_path.replace('standard_smplx', 'uniform_joints').replace('.npz', '.npy'),
            'global_uniform_joints_path': rel_smplx_path.replace('standard_smplx', 'global_uniform_joints').replace('.npz', '.npy'),
            'task': tasks_for_t2m_data + tasks_for_inter_data,
            'pretrain_task': [PRETRAIN_RANDOM_MOTION_GENERATION,
                              PRETRAIN_RANDOM_TEXT_MOTION_GENERATION,
                              PRETRAIN_RANDOM_INTER_MOTION_GENERATION,
                              PRETRAIN_RANDOM_INTER_UNION_TEXT_MOTION_GENERATION,
                              PRETRAIN_RANDOM_INTER_SEPARATE_TEXT_MOTION_GENERATION]

        }
        vis_path = smplx_path.replace('standard_smplx', 'vis_video').replace('.npz', '.mp4')
        if exists(vis_path):
            sample['vis_path'] = rel_smplx_path.replace('standard_smplx', 'vis_video').replace('.npz', '.mp4')
        train_data_info['data_list'][f'{subset_name}_{index}_{person_id}'] = sample

    write_json(join(data_dir, 'train.json'), train_data_info)


if __name__ == '__main__':
    get_anno()

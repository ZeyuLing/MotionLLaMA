# TODO: finish it
import numpy as np
import os
import sys
from collections import defaultdict
from glob import glob
from os.path import join, dirname, basename, relpath

from tqdm import tqdm
sys.path.append(os.curdir)

from mmotion.utils.files_io.json import write_json


def main(data_dir: str = 'data/motionhub/hi4d'):
    fps = 30
    train_data_info = {
        'meta_info': {'dataset': 'hi4d',
                      'version': 'v1'},
        'data_list': defaultdict()
    }

    for motion_path in tqdm(glob(join(data_dir, 'pair*/*/standard_smplx'))):
        action = basename(dirname(motion_path))
        pair = basename(dirname(dirname(motion_path)))

        p1_path = join(motion_path, 'P1.npz')
        p2_path = join(motion_path, 'P2.npz')

        smpl_p1 = np.load(p1_path, allow_pickle=True)
        smpl_p2 = np.load(p2_path, allow_pickle=True)

        p1_index = f'hi4d_{pair}_{action}_p1'
        p2_index = f'hi4d_{pair}_{action}_p2'

        p1_info = {
            'duration': len(smpl_p1['transl']) / fps,
            'smplx_path': relpath(p1_path, dirname(data_dir)),
            'raw_path': relpath(p1_path, dirname(data_dir)).replace('standard_smplx', 'smpl'),
            'caption_path': relpath(p1_path, dirname(data_dir)).replace('standard_smplx', 'caption').replace('.npz','.txt'),
            'union_caption_path': relpath(p1_path, dirname(data_dir)).replace('standard_smplx', 'union_caption')
            .replace('/P1.txt', '.txt').replace('/P2.txt', '.txt'),
            'fps': fps,
            'has_hand': False,
            'interactor_key': p2_index,
            'hm3d_path': relpath(p1_path, dirname(data_dir)).replace('standard_smplx', 'humanml3d').replace('.npz', '.npy'),
            'vis_path': relpath(p1_path, dirname(data_dir)).replace('standard_smplx', 'vis_video').replace('.npz', '.mp4'),
            'interhuman_path': relpath(p1_path, dirname(data_dir)).replace('standard_smplx', 'interhuman').replace('.npz', '.npy'),
            'uniform_joints_path': relpath(p1_path, dirname(data_dir)).replace('standard_smplx', 'uniform_joints').replace('.npz', '.npy'),
            'global_uniform_joints_path': relpath(p1_path, dirname(data_dir)).replace('standard_smplx', 'global_uniform_joints').replace('.npz', '.npy'),

        }

        p2_info = {
            'duration': len(smpl_p2['transl']) / fps,
            'smplx_path': relpath(p2_path, dirname(data_dir)),
            'raw_path': relpath(p2_path, dirname(data_dir)).replace('standard_smplx', 'smpl').replace('.pkl', '.npz'),
            'caption_path': relpath(p2_path, dirname(data_dir)).replace('standard_smplx', 'caption').replace('.npz', '.txt'),
            'union_caption_path': relpath(p2_path, dirname(data_dir)).replace('standard_smplx','union_caption')
            .replace('/P1.txt','.txt').replace('/P2.txt', '.txt'),
            'fps': fps,
            'has_hand': False,
            'interactor_key': p1_index,
            'hm3d_path': relpath(p2_path, dirname(data_dir)).replace('standard_smplx', 'humanml3d').replace('.npz',
                                                                                                            '.npy'),
            'vis_path': relpath(p2_path, dirname(data_dir)).replace('standard_smplx', 'vis_video').replace('.npz',
                                                                                                           '.mp4'),
            'interhuman_path': relpath(p2_path, dirname(data_dir)).replace('standard_smplx', 'interhuman').replace('.npz','.npy'),
            'uniform_joints_path': relpath(p2_path, dirname(data_dir)).replace('standard_smplx', 'uniform_joints').replace('.npz','.npy'),
            'global_uniform_joints_path': relpath(p2_path, dirname(data_dir)).replace('standard_smplx', 'global_uniform_joints').replace('.npz','.npy'),
        }

        train_data_info['data_list'][p1_index] = p1_info
        train_data_info['data_list'][p2_index] = p2_info
    write_json(join(data_dir, 'train.json'), train_data_info)


if __name__ == '__main__':
    main()

import numpy as np
import os
import sys
from collections import defaultdict
from os.path import join, dirname, relpath, exists
from typing import Dict

from tqdm import tqdm
sys.path.append(os.curdir)

from mmotion.utils.files_io.json import write_json
from mmotion.utils.files_io.txt import read_list_txt, read_txt

invalid_list = ['7543', '7561', '4434']
def merge_subset(train_set, val_set, test_set):
    all_set = train_set
    all_set['data_list'].update(val_set['data_list'])
    all_set['data_list'].update(test_set['data_list'])
    return all_set


def get_anno(data_dir: str = 'data/motionhub/interhuman'):
    """
    :param data_dir: interhuman dataset path
    :return: a train.json and a test.json file for interhuman dataset
    """

    train_subset = process_subset('data/motionhub/interhuman/split/train.txt')
    val_subset = process_subset('data/motionhub/interhuman/split/val.txt')
    test_subset = process_subset('data/motionhub/interhuman/split/test.txt')
    all_subset = merge_subset(train_subset, val_subset, test_subset)

    write_json(join(data_dir, 'train.json'), train_subset)
    write_json(join(data_dir, 'val.json'), val_subset)
    write_json(join(data_dir, 'test.json'), test_subset)
    write_json(join(data_dir, 'all.json'), all_subset)



def process_subset(split_path: str) -> Dict:
    """
    :param split_path: path of motionx dataset
    :return:
    """
    data_root = dirname(dirname(dirname(split_path)))
    data_info = {
        'meta_info': {'dataset': 'interhuman',
                      'version': 'v1.0'},
        'data_list': defaultdict(),
    }

    index_list = read_list_txt(split_path)
    for index in tqdm(index_list):
        smpl_path_p1 = join(data_root, f'interhuman/standard_smplx/{index}/P1.npz')
        smpl_path_p2 = join(data_root, f'interhuman/standard_smplx/{index}/P2.npz')
        union_caption_path =  f'interhuman/annots/{index}.txt'
        try:
            union_caption = read_txt(join(data_root, union_caption_path))
        except:
            assert False, union_caption # manually modify it
        if len(union_caption)==0:
            invalid_list.append(index)
        if not exists(smpl_path_p1):
            invalid_list.append(index)
            continue
        smpl_p1 = np.load(smpl_path_p1, allow_pickle=True)
        smpl_p2 = np.load(smpl_path_p2, allow_pickle=True)
        fps = 60.
        num_frames = smpl_p1['num_frames']
        p1_info = {
            'duration': num_frames / fps,
            'num_frames': num_frames,
            'smplx_path': relpath(smpl_path_p1, data_root),
            'raw_path': relpath(smpl_path_p1, data_root).replace('standard_smplx', 'sep_motions').replace('.npz',
                                                                                                          '.pkl'),
            'caption_path': relpath(smpl_path_p1, data_root).replace('standard_smplx', 'sep_annots').replace('.npz',
                                                                                                             '.txt'),
            'union_caption_path': union_caption_path,
            'fps': fps,
            'has_hand': False,
            'interactor_key': f'interhuman_{index}_p2',
            'hm3d_path': relpath(smpl_path_p1, data_root).replace('standard_smplx', 'humanml3d').replace('.npz', '.npy'),
            'interhuman_path': relpath(smpl_path_p1, data_root).replace('standard_smplx', 'interhuman').replace('.npz', '.npy'),
            'uniform_joints_path': relpath(smpl_path_p1, data_root).replace('standard_smplx', 'uniform_joints').replace('.npz', '.npy'),
            'global_uniform_joints_path': relpath(smpl_path_p1, data_root).replace('standard_smplx', 'global_uniform_joints').replace('.npz', '.npy'),
        }
        num_frames = smpl_p2['num_frames']
        p2_info = {
            'duration': num_frames / fps,
            'num_frames': num_frames,
            'smplx_path': relpath(smpl_path_p2, data_root),
            'raw_path': relpath(smpl_path_p2, data_root).replace('standard_smplx', 'sep_motions').replace('.npz',
                                                                                                          '.pkl'),
            'caption_path': relpath(smpl_path_p2, data_root).replace('standard_smplx', 'sep_annots').replace('.npz',
                                                                                                             '.txt'),
            'union_caption_path': union_caption_path,
            'fps': fps,
            'has_hand': False,
            'interactor_key': f'interhuman_{index}_p1',
            'hm3d_path': relpath(smpl_path_p2, data_root).replace('standard_smplx', 'humanml3d').replace('.npz', '.npy'),
            'interhuman_path': relpath(smpl_path_p2, data_root).replace('standard_smplx', 'interhuman').replace('.npz', '.npy'),
            'uniform_joints_path': relpath(smpl_path_p2, data_root).replace('standard_smplx', 'uniform_joints').replace('.npz', '.npy'),
            'global_uniform_joints_path': relpath(smpl_path_p2, data_root).replace('standard_smplx', 'global_uniform_joints').replace('.npz', '.npy'),
        }
        if index in invalid_list:
            p1_info['invalid']=True
            p2_info['invalid']=True
        data_info['data_list'][f'interhuman_{index}_p1'] = p1_info
        data_info['data_list'][f'interhuman_{index}_p2'] = p2_info
    print(f'{len(invalid_list)} missed ', invalid_list)

    return data_info


if __name__ == '__main__':
    get_anno()

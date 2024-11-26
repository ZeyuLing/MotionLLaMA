"""
    The Hi4D smpl data is given in frame-level. Each frame contains 2 persons.
    This script separate the persons and merge the frames.
"""
import numpy as np
import os
import sys
from glob import glob
from os.path import join
from typing import List

from tqdm import tqdm
sys.path.append(os.curdir)
from mmotion.utils.files_io.pickle import save_pickle


def reformat_motion(frame_list: List[str]):
    p1 = {
        'transl': None,
        'body_pose': None,
        'global_orient': None,
        'betas': None
    }

    p2 = {
        'transl': None,
        'body_pose': None,
        'global_orient': None,
        'betas': None
    }

    for key in p1.keys():
        p1[key] = np.stack([np.load(frame, allow_pickle=True)[key][0] for frame in frame_list], axis=0)
    for key in p2.keys():
        p2[key] = np.stack([np.load(frame, allow_pickle=True)[key][1] for frame in frame_list], axis=0)

    return p1, p2


def main(data_dir: str = 'data/motionhub/hi4d'):
    for pair in tqdm(glob(join(data_dir, 'pair**'))):
        motion_path_list = glob(join(pair, '*/smpl'))
        for motion_path in motion_path_list:
            frame_file_list = sorted(glob(join(motion_path, '*.npz')))
            p1, p2 = reformat_motion(frame_file_list)
            save_pickle(p1, join(motion_path, 'P1.pkl'))
            save_pickle(p2, join(motion_path, 'P2.pkl'))


if __name__ == '__main__':
    main()

import numpy as np
import os
import sys
from glob import glob
from os.path import join, basename

from tqdm import tqdm
sys.path.append(os.curdir)
from mmotion.utils.files_io.pickle import save_pickle

if __name__ == '__main__':
    data_root = 'data/motionhub/interhuman'
    sep_motion_root = join(data_root, 'sep_motions')

    os.makedirs(sep_motion_root, exist_ok=True)
    for motion in tqdm(glob(join(data_root, 'motions/*.pkl'))):
        idx = basename(motion).split('.')[0]
        data_dict = np.load(motion, allow_pickle=True)
        person1 = data_dict['person1']
        person2 = data_dict['person2']

        person1['mocap_framerate'] = data_dict['mocap_framerate']
        person2['mocap_framerate'] = data_dict['mocap_framerate']

        person1['frames'] = data_dict['frames']
        person2['frames'] = data_dict['frames']

        person1['global_orient'] = person1.pop('root_orient')
        person2['global_orient'] = person2.pop('root_orient')

        person1['body_pose'] = person1.pop('pose_body')
        person2['body_pose'] = person2.pop('pose_body')

        os.makedirs(join(sep_motion_root, f'{idx}'), exist_ok=True)

        save_pickle(person1, join(sep_motion_root, f'{idx}', 'P1.pkl'))
        save_pickle(person2, join(sep_motion_root, f'{idx}', 'P2.pkl'))

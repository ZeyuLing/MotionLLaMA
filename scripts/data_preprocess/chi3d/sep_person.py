import numpy as np
import os
import pickle
import sys
from glob import glob
from os.path import join, basename

from tqdm import tqdm
sys.path.append(os.curdir)
from mmotion.utils.files_io.json import read_json


def separate(data_dir: str = 'data/motionhub/chi3d'):
    for smplx_path in tqdm(glob(join(data_dir, 'train/s*/smplx', '*.json'))):
        save_dir = smplx_path.strip('.json')
        os.makedirs(save_dir, exist_ok=True)
        persons_data = read_json(smplx_path)
        p1_smplx = {}
        p2_smplx = {}
        for key, value in persons_data.items():
            p1_smplx[key] = np.asarray(value[0])
            p2_smplx[key] = np.asarray(value[1])
        with open(join(save_dir, 'P1.pkl'), 'wb') as fp:
            pickle.dump(p1_smplx, fp)
        with open(join(save_dir, 'P2.pkl'), 'wb') as fp:
            pickle.dump(p2_smplx, fp)


if __name__ == '__main__':
    separate()

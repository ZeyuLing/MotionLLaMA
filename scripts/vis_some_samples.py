import os
import random
import sys
from os.path import join

import fire

sys.path.append(os.curdir)
from mmotion.utils.files_io.json import read_json
from scripts.visualize.vis_smplx import vis_smplx


def main(anno: str = 'data/motionhub/train.json', num_samples=10):
    data_list = read_json(anno)['data_list']
    samples = random.sample(list(data_list.keys()), k=min(num_samples, len(data_list.keys())))
    for key in samples:
        sample = data_list[key]
        npz_path = join('data/motionhub', sample['smplx_path'])
        vis_smplx(npz_path, save_dir=f'vis_results/{key}')


if __name__ == '__main__':
    fire.Fire(main)

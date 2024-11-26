from os.path import isdir, join

import numpy as np
import os
import sys

import fire
from glob import glob

sys.path.append(os.curdir)
from mmotion.utils.bvh.joints2bvh_converter import Joint2BVHConvertor


def transform(joints_path, bvh_path, convertor):
    joints = np.load(joints_path)
    convertor.convert(joints, bvh_path)


def main(joints_path: str, bvh_template='data/motionhub/template.bvh'):
    convertor = Joint2BVHConvertor(template_file=bvh_template)
    if isdir(joints_path):
        joints_path = glob(join(joints_path, '*.npy'))
    if isinstance(joints_path, str):
        joints_path = [joints_path]
    for path in joints_path:

        bvh_path = path.replace('.npy', '.bvh')
        print(path, bvh_path)
        transform(path, bvh_path, convertor)


if __name__ == '__main__':
    fire.Fire(main)

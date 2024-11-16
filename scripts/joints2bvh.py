import numpy as np
import os
import sys

import fire
sys.path.append(os.curdir)
from mmotion.utils.bvh.joints2bvh_converter import Joint2BVHConvertor


def main(joints_path: str, bvh_path: str, bvh_template='data/motionhub/template.bvh'):
    convertor = Joint2BVHConvertor(template_file=bvh_template)
    joints = np.load(joints_path)
    convertor.convert(joints, bvh_path)

if __name__=='__main__':
    fire.Fire(main)

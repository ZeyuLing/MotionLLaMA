"""
    Visualize the motion vectors
"""
from os.path import normpath, isfile

import fire
import math
import os.path
import sys

sys.path.append(os.curdir)
from mmotion.core.visualization import visualize_kp3d


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


import numpy as np

def main(
    vec_path:str,
    title:str = 'some_motion',
    save_path:str = 'vis_results',
    fps:int = 30
):
    vec_path = normpath(vec_path)
    if not isfile(save_path):
        save_path = os.path.join(save_path, os.path.basename(vec_path).split('.')[0] + '.mp4')

    # t j 3 for each item
    joints = np.load(vec_path)

    visualize_kp3d(
        joints,
        frame_names=title,
        output_path=save_path,
        convention='blender',
        resolution=(1024, 1024),
        data_source='smplh',
        show_axis=False,
        show_floor=True,
        fps=fps
    )
if __name__ == '__main__':
    fire.Fire(main)
"""
    Visualize the motion vectors
"""
from glob import glob
from os.path import isfile, normpath, isdir, join

import fire
import math
import os.path
import sys

import torch
from einops import rearrange

sys.path.append(os.curdir)
from mmotion.core.conventions.keypoints_mapping import keypoint_index_mapping
from mmotion.motion_representation.param_utils import t2m_raw_offsets, t2m_body_hand_kinematic_chain
from mmotion.motion_representation import Skeleton, interhuman2joints, interhuman2joints_from_rot

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



num_joints_dict = {
    137: 22,  # wo rot, body
    263: 22,  # with rot, body
    313: 52,  # wo rot, whole body
    623: 52  # with rot, whole body
}

def main(
    vec_path:str,
    title:str = 'some_motion',
    save_path:str = 'vis_results',
    use_rot:bool = False,
    reference_file:str = 'data/motionhub/motionx/motion_data/standard_smplx/humanml/000021.npz',
    fps:int = 30
):
    vec_path = normpath(vec_path)
    if not isfile(save_path):
        save_path = os.path.join(save_path, os.path.basename(vec_path).split('.')[0] + '.mp4')

    if isdir(vec_path):
        # for multi-person visualization
        vec_path = glob(join(vec_path, '*.npy'))
    else:
        assert isfile(vec_path)
        vec_path = [vec_path]
    # t j 3 for each item
    if not use_rot:
        joints = np.stack([interhuman2joints(np.load(p), num_joints=52) for p in vec_path], axis=1)
    else:
        smplh_index = keypoint_index_mapping('smplh', 'smplx')

        reference_data = np.load(reference_file, allow_pickle=True)['joints']
        reference_data = reference_data[:, smplh_index]
        reference_data = reference_data.reshape(len(reference_data), -1, 3)

        tgt_skel = Skeleton(torch.from_numpy(t2m_raw_offsets),
                            t2m_body_hand_kinematic_chain, 'cpu')
        _ = tgt_skel.get_offsets_joints(reference_data[0])

        joints = np.stack([
            interhuman2joints_from_rot(torch.from_numpy(np.load(p)).to(torch.float32), skeleton=tgt_skel).numpy()
            for p in vec_path], axis=1)


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
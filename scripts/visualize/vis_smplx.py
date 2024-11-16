"""
    Supporting visualization of standard smplx dict file(pkl or npz)
"""
import numpy as np
import os
import sys
from os import makedirs
from os.path import join, basename, isdir, normpath
from typing import Union

import fire

sys.path.append(os.curdir)
from mmotion.utils.smpl_utils.smpl_key_const import JOINTS,  extract_smplx_keys, merge_pose_keys

from mmotion.utils.smpl_utils.merge import merge_pose, merge_persons


from mmotion.core.visualization import visualize_kp3d
from mmotion.core.visualization.visualize_smpl import render_smpl
from mmotion.registry import MODELS, TRANSFORMS

transform = TRANSFORMS.build({'type':'LoadSmpl'})
def vis_smplx(
        smplx_file: Union[str],
        save_dir: str = 'vis_results',
        pretrained_smplx: str = 'checkpoints/smpl_models/smplx/',
        mesh_only=False,
        joints_only=False
):
    smplx_file = normpath(smplx_file)
    assert not (mesh_only and joints_only), 'visualize at list one format!'
    smplx_model = MODELS.build(dict(model_path=pretrained_smplx, type='SMPLX', gender='neutral', create_betas=False,
                                    create_transl=False, create_expression=False, create_body_pose=False,
                                    create_reye_pose=False, create_jaw_pose=False, create_leye_pose=False,
                                    create_left_hand_pose=False, create_right_hand_pose=False,
                                    create_global_orient=False,
                                    use_pca=False, use_face_contour=False, keypoint_src='smplx_no_contour'))
    if isdir(smplx_file):
        P1_path = join(smplx_file, 'P1.npz')
        P2_path = join(smplx_file, 'P2.npz')

        smplx_dict_p1 = np.load(P1_path, allow_pickle=True)
        smplx_dict_p2 = np.load(P2_path, allow_pickle=True)
        smplx_dict_list = [merge_pose(smplx_dict_p1),
                           merge_pose(smplx_dict_p2)]
        joints_list = [
            np.concatenate([smplx_dict_p1[JOINTS][:, :22], smplx_dict_p1[JOINTS][:, 25:55]], axis=1),
            np.concatenate([smplx_dict_p2[JOINTS][:, :22], smplx_dict_p2[JOINTS][:, 25:55]], axis=1)
        ]
        smplx_dict = merge_persons(smplx_dict_list)
        joints = np.stack(joints_list, axis=1)
        palette = ['yellow', 'white']
    else:
        smplx_dict = np.load(smplx_file, allow_pickle=True)
        joints = smplx_dict['joints']
        joints = np.concatenate([joints[:, :22], joints[:, 25:55]], axis=1)
        palette = ['white']

    save_path = join(save_dir, basename(smplx_file).split('.')[0])
    makedirs(save_path, exist_ok=True)
    mesh_path = join(save_path, 'mesh.mp4')
    joints_path = join(save_path, 'joints.mp4')
    if not mesh_only:
        visualize_kp3d(
            joints,
            joints_path,
            convention='blender',
            data_source='smplh',
            resolution=(1024, 1024)
        )
    if not joints_only:
        render_smpl(
            **smplx_dict,
            output_path=mesh_path,
            convention='blender',
            body_model=smplx_model,
            resolution=(1024, 1024),
            overwrite=True,
            palette=palette
        )

if __name__ == '__main__':
    fire.Fire(vis_smplx)

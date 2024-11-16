"""
    For motionx 322-smplx npy file visualization
"""
import numpy as np
import os
import sys
from os.path import join, basename, dirname
from typing import Union, List, Dict

import fire
import torch
from einops import rearrange

sys.path.append(os.curdir)
from mmotion.utils.geometry.rotation_convert import rot_convert
from mmotion.core.visualization.visualize_smpl import visualize_smpl_pose

DEFAULT_SAVE: str = 'vis_results/motionx/'


def load_smplx_file(smplx_file: str) -> Dict:
    smplx_322 = torch.from_numpy(np.load(smplx_file))
    num_frames = smplx_322.shape[0]
    zero_pose = torch.zeros((num_frames, 3), dtype=smplx_322.dtype, device=smplx_322.device)
    # when determine dict keys, we reference to smplx repo.

    smplx_full_pose = torch.cat([
        smplx_322[..., :66],  # body
        smplx_322[..., 66 + 90:66 + 93],  # jaw
        zero_pose,  # leye
        zero_pose,  # reye
        smplx_322[..., 66: 66 + 90]
    ], dim=-1)
    smplx_full_pose = rearrange(smplx_full_pose, 't (j c) -> t j c', c=3)
    smplx_full_pose = rot_convert(smplx_full_pose, 'axis_angle', 'quaternion')
    smplx_full_pose = rot_convert(smplx_full_pose, 'quaternion', 'axis_angle')
    smplx_full_pose = rearrange(smplx_full_pose, 't j c -> t (j c)')
    return dict(
        poses=smplx_full_pose,
        transl=smplx_322[..., 309:309 + 3],
        betas=smplx_322[..., 312:]
    )


def main(
        smplx_file: Union[str, List[str]],
        save_path: Union[str, List[str]] = None,
        pretrained_smplx: str = 'smpl_models',
):
    body_model_config = dict(model_path=pretrained_smplx, type='SMPLX', gender='neutral', create_betas=False,
                             create_transl=False, create_expression=False, create_body_pose=False,
                             create_reye_pose=False, create_jaw_pose=False, create_leye_pose=False,
                             create_left_hand_pose=False, create_right_hand_pose=False, create_global_orient=False,
                             use_pca=False, use_face_contour=False, keypoint_src='smplx_no_contour')
    if not isinstance(smplx_file, List):
        smplx_file = [smplx_file]

    if save_path is None:
        save_path = []
        for path in smplx_file:
            save_name = basename(path).replace('.npy', '.mp4')
            sp = join(DEFAULT_SAVE, save_name)
            save_path.append(sp)
            os.makedirs(dirname(sp), exist_ok=True)
    elif isinstance(save_path, str):
        save_path = [save_path]

    batch_smplx_list = [load_smplx_file(path) for path in smplx_file]

    for smplx_dict, path in zip(batch_smplx_list, save_path):
        visualize_smpl_pose(
            **smplx_dict,
            output_path=path,
            resolution=(1024, 1024),
            body_model_config=body_model_config,
            overwrite=True,
            device='cpu',
            # T=np.zeros([3])
        )


if __name__ == '__main__':
    fire.Fire(main)
